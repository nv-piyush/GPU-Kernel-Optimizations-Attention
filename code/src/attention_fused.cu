/*
 * attention_fused.cu
 *
 * Phase 4 — Two optimizations addressing the softmax bottleneck
 * identified in Week 3 profiling:
 *
 * Variant A: "softmax_fixed" — tiled QK + fixed-grid softmax + attn_output
 *   Fixes the 2-block grid by assigning one thread per row,
 *   grid = (N, B). Isolates the cost of grid underutilization.
 *
 * Variant B: "fused" — fused QK+softmax kernel + attn_output
 *   Each thread block computes one full row of S (QK^T/sqrt(d)),
 *   applies softmax in registers, and writes the result — eliminating
 *   the intermediate [B,N,N] DRAM round-trip between QK and softmax.
 *
 * Both variants keep attn_output unchanged from baseline/tiled.
 *
 * Build:  see CMakeLists.txt
 * Run:    ./attention_fused --variant softmax_fixed --batch 4 --seq 512 --dim 64
 *         ./attention_fused --variant fused         --batch 4 --seq 512 --dim 64
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

#define TILE_SIZE 16

/* ═══════════════════════════════════════════════════════════════════════════
 * VARIANT A KERNELS
 * Tiled QK (from Phase 3) + Fixed-grid softmax
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Tiled QKᵀ kernel — identical to Phase 3 */
__global__ void qk_dot_tiled_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float*       __restrict__ S,
    int B, int N, int d
) {
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];

    int b  = blockIdx.z;
    int i  = blockIdx.y * TILE_SIZE + threadIdx.y;
    int j  = blockIdx.x * TILE_SIZE + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (d + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_col  = t * TILE_SIZE + tx;
        int j_load = blockIdx.x * TILE_SIZE + ty;

        Qs[ty][tx] = (i < N && k_col < d) ? Q[b*N*d + i*d + k_col]      : 0.f;
        Ks[ty][tx] = (j_load < N && k_col < d) ? K[b*N*d + j_load*d + k_col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) acc += Qs[ty][k] * Ks[tx][k];
        __syncthreads();
    }
    if (b < B && i < N && j < N)
        S[b*N*N + i*N + j] = acc / sqrtf((float)d);
}

/* ── Variant A: Fixed-grid softmax ──────────────────────────────────────────
 *
 * ROOT CAUSE OF BOTTLENECK (Week 3):
 *   Old grid: (ceil(N/256), B) with block=(256)
 *   → at N=512: grid=(2,1) → only 2 SMs active out of 36
 *
 * Fix: one thread per row, grid=(N, B), block=(1)
 *   → at N=512: grid=(512,1) → 512 blocks → all 36 SMs active
 *   Each thread still does the sequential 3-pass reduction over N columns,
 *   but now rows are fully parallelized across SMs.
 *
 * This isolates the grid underutilization cost from fusion complexity.
 */
__global__ void softmax_fixed_grid_kernel(
    float* __restrict__ S,
    int B, int N
) {
    int b = blockIdx.y;
    int i = blockIdx.x;   /* one block per row — grid=(N, B) */

    if (b >= B || i >= N) return;

    float* row = S + b * N * N + i * N;

    float mx = row[0];
    for (int j = 1; j < N; j++) mx = fmaxf(mx, row[j]);

    float s = 0.f;
    for (int j = 0; j < N; j++) { row[j] = expf(row[j] - mx); s += row[j]; }
    for (int j = 0; j < N; j++) row[j] /= s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * VARIANT B KERNELS
 * Fused QK + Softmax kernel
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── Variant B: Fused QK + Softmax ─────────────────────────────────────────
 *
 * Each thread block handles one output row i of one batch b.
 * The block computes the full row of dot products S[b,i,:] in shared memory,
 * applies softmax in-place in shared memory, then writes the result to S.
 *
 * This eliminates the DRAM round-trip of the full [B,N,N] score matrix
 * between the QK kernel and the softmax kernel:
 *   - Baseline/tiled: QK writes S to DRAM → softmax reads S from DRAM
 *   - Fused: S[b,i,:] stays in shared memory throughout
 *
 * At N=1024, d=64, B=32: score matrix = 32 × 1024 × 1024 × 4B = 128 MB
 * Eliminating this write+read saves significant DRAM bandwidth.
 *
 * Grid:  (N, B)        — one block per (row, batch)
 * Block: (BLOCK_N)     — threads collaborate on one row
 *
 * Shared memory: BLOCK_N floats for the score row
 */
#define FUSED_BLOCK 256   /* threads per block; must be >= N or use loops */

__global__ void qk_softmax_fused_kernel(
    const float* __restrict__ Q,   // [B, N, d]
    const float* __restrict__ K,   // [B, N, d]
    float*       __restrict__ S,   // [B, N, N] output (softmax weights)
    int B, int N, int d
) {
    extern __shared__ float smem[];  /* dynamic: N floats per block */

    int b = blockIdx.y;
    int i = blockIdx.x;             /* this block owns row i of batch b */

    if (b >= B || i >= N) return;

    const float* q_row = Q + b * N * d + i * d;

    /* ── Step 1: compute all dot products for row i ── */
    /* Each thread computes a strided subset of j values */
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        const float* k_row = K + b * N * d + j * d;
        float acc = 0.f;
        for (int k = 0; k < d; k++) acc += q_row[k] * k_row[k];
        smem[j] = acc / sqrtf((float)d);
    }
    __syncthreads();

    /* ── Step 2: parallel row max (tree reduction) ── */
    __shared__ float smax[FUSED_BLOCK];
    float local_max = -1e38f;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        local_max = fmaxf(local_max, smem[j]);
    smax[threadIdx.x] = local_max;
    __syncthreads();

    /* reduce within block */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x],
                                       smax[threadIdx.x + stride]);
        __syncthreads();
    }
    float row_max = smax[0];

    /* ── Step 3: exp(x - max) and parallel sum ── */
    __shared__ float ssum[FUSED_BLOCK];
    float local_sum = 0.f;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        smem[j] = expf(smem[j] - row_max);
        local_sum += smem[j];
    }
    ssum[threadIdx.x] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride)
            ssum[threadIdx.x] += ssum[threadIdx.x + stride];
        __syncthreads();
    }
    float row_sum = ssum[0];

    /* ── Step 4: normalize and write to global memory ── */
    float* s_row = S + b * N * N + i * N;
    for (int j = threadIdx.x; j < N; j += blockDim.x)
        s_row[j] = smem[j] / row_sum;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SHARED: attn_output kernel (unchanged from baseline)
 * ═══════════════════════════════════════════════════════════════════════════ */

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

    const float* s_row      = S + b * N * N + i * N;
    const float* v_col_base = V + b * N * d + k;
    float acc = 0.f;
    for (int j = 0; j < N; j++) acc += s_row[j] * v_col_base[j * d];
    out[b * N * d + i * d + k] = acc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * HOST RUNNER
 * ═══════════════════════════════════════════════════════════════════════════ */

void run_attention_fused(
    int B, int N, int d,
    const char* variant,          /* "softmax_fixed" or "fused" */
    const float* h_Q, const float* h_K, const float* h_V,
    float* h_out,
    float* latency_ms,
    int warmup_iters, int measure_iters
) {
    size_t qkv_bytes   = (size_t)B * N * d * sizeof(float);
    size_t score_bytes = (size_t)B * N * N * sizeof(float);

    float *d_Q, *d_K, *d_V, *d_S, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q,   qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_K,   qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_V,   qkv_bytes));
    CUDA_CHECK(cudaMalloc(&d_S,   score_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, qkv_bytes));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));

    /* launch configs */
    dim3 blk_qk(TILE_SIZE, TILE_SIZE);
    dim3 grd_qk((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE, B);

    /* fixed-grid softmax: one block per row */
    dim3 blk_sm_fixed(1);
    dim3 grd_sm_fixed(N, B);

    /* fused kernel: one block per row, FUSED_BLOCK threads */
    dim3 blk_fused(FUSED_BLOCK);
    dim3 grd_fused(N, B);
    size_t smem_fused = (size_t)N * sizeof(float);

    dim3 blk_out(16, 16);
    dim3 grd_out((d+15)/16, (N+15)/16, B);

    int use_fused = (strcmp(variant, "fused") == 0);

    auto forward = [&]() {
        if (use_fused) {
            /* Variant B: fused QK+softmax */
            qk_softmax_fused_kernel<<<grd_fused, blk_fused, smem_fused>>>(
                d_Q, d_K, d_S, B, N, d);
        } else {
            /* Variant A: tiled QK + fixed-grid softmax */
            qk_dot_tiled_kernel<<<grd_qk, blk_qk>>>(d_Q, d_K, d_S, B, N, d);
            softmax_fixed_grid_kernel<<<grd_sm_fixed, blk_sm_fixed>>>(d_S, B, N);
        }
        attn_output_kernel<<<grd_out, blk_out>>>(d_S, d_V, d_out, B, N, d);
    };

    for (int i = 0; i < warmup_iters; i++) forward();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < measure_iters; i++) forward();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float total_ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, t0, t1));
    *latency_ms = total_ms / measure_iters;

    CUDA_CHECK(cudaMemcpy(h_out, d_out, qkv_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

/* ─── Helpers ────────────────────────────────────────────────────────────── */

void fill_random(float* arr, size_t n, unsigned int seed) {
    srand(seed);
    for (size_t i = 0; i < n; i++)
        arr[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --variant V  'softmax_fixed' or 'fused'   (default: fused)\n"
        "  --batch  B   Batch size                    (default: 1)\n"
        "  --seq    N   Sequence length               (default: 512)\n"
        "  --dim    d   Head dimension                (default: 64)\n"
        "  --warmup W   Warmup iterations             (default: 5)\n"
        "  --iters  I   Measured iterations           (default: 100)\n"
        "  --csv    F   Append results to CSV file\n"
        "  --dump   D   Dump Q/K/V/out binaries to directory D\n",
        prog);
}

/* ─── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int B = 1, N = 512, d = 64, warmup = 5, iters = 100;
    const char* variant  = "fused";
    const char* csv_path = NULL;
    const char* dump_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--variant") && i+1<argc) variant  = argv[++i];
        else if (!strcmp(argv[i], "--batch")   && i+1<argc) B        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq")     && i+1<argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim")     && i+1<argc) d        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup")  && i+1<argc) warmup   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")   && i+1<argc) iters    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")     && i+1<argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--dump")    && i+1<argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--help"))   { print_usage(argv[0]); return 0; }
    }

    printf("=== Attention Fused (Phase 4) | variant=%s ===\n", variant);
    printf("B=%d  N=%d  d=%d  warmup=%d  iters=%d\n\n", B, N, d, warmup, iters);

    size_t qkv_n = (size_t)B * N * d;
    float* h_Q   = (float*)malloc(qkv_n * sizeof(float));
    float* h_K   = (float*)malloc(qkv_n * sizeof(float));
    float* h_V   = (float*)malloc(qkv_n * sizeof(float));
    float* h_out = (float*)malloc(qkv_n * sizeof(float));

    fill_random(h_Q, qkv_n, 42);
    fill_random(h_K, qkv_n, 43);
    fill_random(h_V, qkv_n, 44);

    float latency_ms = 0.f;
    run_attention_fused(B, N, d, variant, h_Q, h_K, h_V, h_out,
                        &latency_ms, warmup, iters);

    double tokens_per_sec = (double)B * N / (latency_ms / 1000.0);
    printf("Latency:    %.4f ms\n", latency_ms);
    printf("Throughput: %.2f k tokens/sec\n", tokens_per_sec / 1e3);

    int has_nan = 0;
    for (size_t i = 0; i < qkv_n; i++)
        if (isnan(h_out[i]) || isinf(h_out[i])) { has_nan = 1; break; }
    printf("Output NaN/Inf check: %s\n", has_nan ? "FAILED ❌" : "PASSED ✅");

    printf("\nFirst 8 output values:\n");
    for (int i = 0; i < 8 && i < (int)qkv_n; i++)
        printf("  out[%d] = %.6f\n", i, h_out[i]);

    if (csv_path) {
        FILE* f = fopen(csv_path, "a");
        if (f) {
            fseek(f, 0, SEEK_END);
            if (ftell(f) == 0)
                fprintf(f, "variant,B,N,d,latency_ms,"
                           "throughput_k_tokens_per_sec\n");
            fprintf(f, "%s,%d,%d,%d,%.4f,%.2f\n",
                    variant, B, N, d, latency_ms, tokens_per_sec / 1e3);
            fclose(f);
            printf("\nResults appended to %s\n", csv_path);
        }
    }

    if (dump_dir) {
        char path[512]; FILE* f; size_t n_elem = (size_t)B * N * d;
        #define DUMP(name, arr) \
            snprintf(path, sizeof(path), "%s/%s.bin", dump_dir, name); \
            f = fopen(path, "wb"); \
            if (f) { fwrite(arr, sizeof(float), n_elem, f); fclose(f); } \
            printf("Dumped: %s\n", path);
        DUMP("Q", h_Q) DUMP("K", h_K) DUMP("V", h_V) DUMP("out", h_out)
        #undef DUMP
        snprintf(path, sizeof(path), "%s/meta.txt", dump_dir);
        f = fopen(path, "w");
        if (f) { fprintf(f, "B=%d\nN=%d\nd=%d\n", B, N, d); fclose(f); }
    }

    free(h_Q); free(h_K); free(h_V); free(h_out);
    return 0;
}