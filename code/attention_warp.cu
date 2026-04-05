/*
 * attention_warp.cu
 *
 * Phase 5 — Alternative optimization: warp-level softmax reduction.
 *
 * Motivation from Week 4 NCU analysis:
 *   softmax_fixed_grid_kernel uses block size = 1 thread.
 *   Only 1 warp per block, 31 of 32 threads idle.
 *   Each warp performs O(N) sequential reduction alone.
 *
 * This variant uses block size = 32 threads (one full warp per block).
 *   - All 32 threads in the warp collaboratively scan the row
 *   - Warp shuffle (__shfl_down_sync) computes max and sum in
 *     O(log2(32)) = 5 steps instead of O(N) sequential steps
 *   - No __syncthreads() needed — warp is implicitly synchronized
 *   - Same grid: (N, B) — one block per row, all SMs fully utilized
 *
 * Pairs with the tiled QK kernel from Phase 3 (unchanged).
 *
 * Build:  cmake .. && make
 * Run:    ./attention_warp --batch 32 --seq 1024 --dim 64
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

#define TILE_SIZE   16
#define WARP_SIZE   32
#define FULL_MASK   0xffffffff

/* ── Tiled QKᵀ kernel (Phase 3, unchanged) ─────────────────────────────── */

__global__ void qk_dot_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float*       __restrict__ S,
    int B, int N, int d
) {
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];

    int b = blockIdx.z;
    int i = blockIdx.y * TILE_SIZE + threadIdx.y;
    int j = blockIdx.x * TILE_SIZE + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (d + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_col  = t * TILE_SIZE + tx;
        int j_load = blockIdx.x * TILE_SIZE + ty;
        Qs[ty][tx] = (i < N && k_col < d) ? Q[b*N*d + i*d + k_col] : 0.f;
        Ks[ty][tx] = (j_load < N && k_col < d) ? K[b*N*d + j_load*d + k_col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) acc += Qs[ty][k] * Ks[tx][k];
        __syncthreads();
    }
    if (b < B && i < N && j < N)
        S[b*N*N + i*N + j] = acc / sqrtf((float)d);
}

/* ── Warp-level softmax kernel ──────────────────────────────────────────────
 *
 * Grid:  (N, B)   — one warp (block of 32 threads) per row
 * Block: (32, 1)  — exactly one warp per block
 *
 * Each warp owns one row S[b, i, :] of length N.
 * Threads stride across the row: thread t handles j = t, t+32, t+64, ...
 *
 * Reduction uses __shfl_down_sync for warp-wide max and sum:
 *   Step 1: each thread finds local max over its strided elements
 *   Step 2: warp reduction → global row max in 5 shuffle steps
 *   Step 3: each thread computes exp(x - max) for its elements, sums locally
 *   Step 4: warp reduction → global sum in 5 shuffle steps
 *   Step 5: each thread normalizes its elements
 *
 * No shared memory needed — all reductions happen in registers via shuffles.
 * No __syncthreads() needed — warp execution is implicitly synchronized.
 */
__global__ void softmax_warp_kernel(
    float* __restrict__ S,
    int B, int N
) {
    int b = blockIdx.y;
    int i = blockIdx.x;   /* one block (= one warp) per row */
    int t = threadIdx.x;  /* thread index within warp: 0..31 */

    if (b >= B || i >= N) return;

    float* row = S + b * N * N + i * N;

    /* ── Pass 1: find local max over strided elements ── */
    float local_max = -1e38f;
    for (int j = t; j < N; j += WARP_SIZE)
        local_max = fmaxf(local_max, row[j]);

    /* ── Warp reduction: max across all 32 threads ── */
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_max = fmaxf(local_max,
                          __shfl_down_sync(FULL_MASK, local_max, offset));
    float row_max = __shfl_sync(FULL_MASK, local_max, 0); /* broadcast */

    /* ── Pass 2: exp(x - max) and local sum ── */
    float local_sum = 0.f;
    for (int j = t; j < N; j += WARP_SIZE) {
        row[j] = expf(row[j] - row_max);
        local_sum += row[j];
    }

    /* ── Warp reduction: sum across all 32 threads ── */
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, offset);
    float row_sum = __shfl_sync(FULL_MASK, local_sum, 0); /* broadcast */

    /* ── Pass 3: normalize ── */
    for (int j = t; j < N; j += WARP_SIZE)
        row[j] /= row_sum;
}

/* ── Tiled softmax variant (TILE_SIZE=32) for comparison ───────────────────
 *
 * Investigates whether a larger tile (32×32 = 2KB shared mem per side,
 * 8KB total) improves the QK step over TILE_SIZE=16 (2KB total).
 * Shared memory per block: 2 × 32×32×4 = 8192 bytes = 8KB.
 * Register usage may increase; occupancy impact assessed via NCU.
 */
#define TILE_SIZE_32 32

__global__ void qk_dot_tiled32_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float*       __restrict__ S,
    int B, int N, int d
) {
    __shared__ float Qs[TILE_SIZE_32][TILE_SIZE_32];
    __shared__ float Ks[TILE_SIZE_32][TILE_SIZE_32];

    int b  = blockIdx.z;
    int i  = blockIdx.y * TILE_SIZE_32 + threadIdx.y;
    int j  = blockIdx.x * TILE_SIZE_32 + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    float acc = 0.f;

    for (int t = 0; t < (d + TILE_SIZE_32 - 1) / TILE_SIZE_32; t++) {
        int k_col  = t * TILE_SIZE_32 + tx;
        int j_load = blockIdx.x * TILE_SIZE_32 + ty;
        Qs[ty][tx] = (i < N && k_col < d) ? Q[b*N*d + i*d + k_col] : 0.f;
        Ks[ty][tx] = (j_load < N && k_col < d) ? K[b*N*d + j_load*d + k_col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE_32; k++) acc += Qs[ty][k] * Ks[tx][k];
        __syncthreads();
    }
    if (b < B && i < N && j < N)
        S[b*N*N + i*N + j] = acc / sqrtf((float)d);
}

/* ── attn_output kernel (unchanged) ─────────────────────────────────────── */

__global__ void attn_output_kernel(
    const float* __restrict__ S,
    const float* __restrict__ V,
    float*       __restrict__ out,
    int B, int N, int d
) {
    int b = blockIdx.z;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || i >= N || k >= d) return;
    const float* s_row      = S + b*N*N + i*N;
    const float* v_col_base = V + b*N*d + k;
    float acc = 0.f;
    for (int j = 0; j < N; j++) acc += s_row[j] * v_col_base[j*d];
    out[b*N*d + i*d + k] = acc;
}

/* ── Host runner ─────────────────────────────────────────────────────────── */

void run_attention_warp(
    int B, int N, int d,
    const char* variant,   /* "warp_softmax" or "tiled32" */
    const float* h_Q, const float* h_K, const float* h_V,
    float* h_out, float* latency_ms,
    int warmup, int iters
) {
    size_t qkv_bytes   = (size_t)B*N*d*sizeof(float);
    size_t score_bytes = (size_t)B*N*N*sizeof(float);

    float *d_Q, *d_K, *d_V, *d_S, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q,   qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K,   qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V,   qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_S,   score_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, qkv_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));

    int use_tiled32 = (strcmp(variant, "tiled32") == 0);

    /* tiled16 + warp softmax */
    dim3 blk_qk16(TILE_SIZE, TILE_SIZE);
    dim3 grd_qk16((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE, B);

    /* tiled32 QK */
    dim3 blk_qk32(TILE_SIZE_32, TILE_SIZE_32);
    dim3 grd_qk32((N+TILE_SIZE_32-1)/TILE_SIZE_32,
                  (N+TILE_SIZE_32-1)/TILE_SIZE_32, B);

    /* warp softmax: one warp (32 threads) per row */
    dim3 blk_warp(WARP_SIZE);
    dim3 grd_warp(N, B);

    dim3 blk_out(16, 16);
    dim3 grd_out((d+15)/16, (N+15)/16, B);

    auto forward = [&]() {
        if (use_tiled32) {
            qk_dot_tiled32_kernel<<<grd_qk32, blk_qk32>>>(d_Q, d_K, d_S, B, N, d);
        } else {
            qk_dot_tiled_kernel<<<grd_qk16, blk_qk16>>>(d_Q, d_K, d_S, B, N, d);
        }
        softmax_warp_kernel<<<grd_warp, blk_warp>>>(d_S, B, N);
        attn_output_kernel<<<grd_out, blk_out>>>(d_S, d_V, d_out, B, N, d);
    };

    for (int i = 0; i < warmup; i++) forward();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) forward();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float total_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));
    *latency_ms = total_ms / iters;

    CUDA_CHECK(cudaMemcpy(h_out, d_out, qkv_bytes, cudaMemcpyDeviceToHost));
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ── Helpers ─────────────────────────────────────────────────────────────── */

void fill_random(float* arr, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        arr[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --variant V  'warp_softmax' or 'tiled32'  (default: warp_softmax)\n"
        "  --batch  B   Batch size                    (default: 1)\n"
        "  --seq    N   Sequence length               (default: 512)\n"
        "  --dim    d   Head dimension                (default: 64)\n"
        "  --warmup W   Warmup iterations             (default: 5)\n"
        "  --iters  I   Measured iterations           (default: 100)\n"
        "  --csv    F   Append results to CSV\n", prog);
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int B=1, N=512, d=64, warmup=5, iters=100;
    const char* variant  = "warp_softmax";
    const char* csv_path = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i],"--variant") && i+1<argc) variant  = argv[++i];
        else if (!strcmp(argv[i],"--batch")   && i+1<argc) B        = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seq")     && i+1<argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--dim")     && i+1<argc) d        = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")  && i+1<argc) warmup   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--iters")   && i+1<argc) iters    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--csv")     && i+1<argc) csv_path = argv[++i];
        else if (!strcmp(argv[i],"--help"))   { print_usage(argv[0]); return 0; }
    }

    printf("=== Attention Warp (Phase 5) | variant=%s ===\n", variant);
    printf("B=%d  N=%d  d=%d  warmup=%d  iters=%d\n\n", B, N, d, warmup, iters);

    size_t qkv_n = (size_t)B*N*d;
    float* h_Q   = (float*)malloc(qkv_n*sizeof(float));
    float* h_K   = (float*)malloc(qkv_n*sizeof(float));
    float* h_V   = (float*)malloc(qkv_n*sizeof(float));
    float* h_out = (float*)malloc(qkv_n*sizeof(float));

    fill_random(h_Q, qkv_n, 42);
    fill_random(h_K, qkv_n, 43);
    fill_random(h_V, qkv_n, 44);

    float latency_ms = 0.f;
    run_attention_warp(B, N, d, variant, h_Q, h_K, h_V, h_out,
                       &latency_ms, warmup, iters);

    double tps = (double)B*N / (latency_ms/1000.0);
    printf("Latency:    %.4f ms\n", latency_ms);
    printf("Throughput: %.2f k tokens/sec\n", tps/1e3);

    int has_nan = 0;
    for (size_t i = 0; i < qkv_n; i++)
        if (isnan(h_out[i])||isinf(h_out[i])) { has_nan=1; break; }
    printf("Output NaN/Inf check: %s\n", has_nan ? "FAILED ❌" : "PASSED ✅");

    printf("\nFirst 8 output values:\n");
    for (int i = 0; i < 8 && i < (int)qkv_n; i++)
        printf("  out[%d] = %.6f\n", i, h_out[i]);

    if (csv_path) {
        FILE* f = fopen(csv_path, "a");
        if (f) {
            fseek(f, 0, SEEK_END);
            if (ftell(f)==0)
                fprintf(f, "variant,B,N,d,latency_ms,"
                           "throughput_k_tokens_per_sec\n");
            fprintf(f, "%s,%d,%d,%d,%.4f,%.2f\n",
                    variant, B, N, d, latency_ms, tps/1e3);
            fclose(f);
        }
    }

    free(h_Q); free(h_K); free(h_V); free(h_out);
    return 0;
}