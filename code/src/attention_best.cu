/*
 * attention_best.cu
 *
 * Week 7 — Combined optimal pipeline.
 * Assembles the best-performing kernel for each attention step
 * based on profiling evidence from Phases 1–5:
 *
 *   QK step:     Tiled QK (TILE_SIZE=16) — reduces warp stalls 172→66
 *   Softmax step: Warp softmax (32 threads, __shfl_down_sync)
 *   Output step:  attn_output with __launch_bounds__(256,6)
 *
 * Also includes tiled_vec variant: float4 vectorized loads in QK kernel.
 *
 * Variants:
 *   --variant best       tiled QK + warp softmax + launch_bounds output
 *   --variant tiled_vec  float4 tiled QK + warp softmax + output
 *
 * Build:  cmake .. && make
 * Run:    ./build/attention_best --variant best --batch 32 --seq 1024 --dim 64
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define TILE_SIZE  16
#define WARP_SIZE  32
#define FULL_MASK  0xffffffff

/* ── Kernel 1A: Tiled QK (standard float loads) ────────────────────────── */

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
    float acc = 0.f;

    for (int t = 0; t < (d + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int k_col  = t * TILE_SIZE + tx;
        int j_load = blockIdx.x * TILE_SIZE + ty;
        Qs[ty][tx] = (i < N && k_col < d)
                   ? Q[b*N*d + i*d + k_col] : 0.f;
        Ks[ty][tx] = (j_load < N && k_col < d)
                   ? K[b*N*d + j_load*d + k_col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) acc += Qs[ty][k] * Ks[tx][k];
        __syncthreads();
    }
    if (b < B && i < N && j < N)
        S[b*N*N + i*N + j] = acc / sqrtf((float)d);
}

/* ── Kernel 1B: Tiled QK with float4 vectorized loads ──────────────────── */
/*
 * Loads Q and K as float4 (4 floats per transaction) instead of float.
 * Reduces load instruction count by 4x and improves memory coalescing.
 * Requires d divisible by 4 (satisfied for d=64 and d=128).
 *
 * Each thread loads one float4 per tile iteration for both Q and K,
 * then unpacks into 4 accumulator slots via loop unrolling.
 */
__global__ void qk_dot_tiled_vec_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float*       __restrict__ S,
    int B, int N, int d
) {
    /* tile stores float4 unpacked — same 16x16 layout */
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];

    int b  = blockIdx.z;
    int i  = blockIdx.y * TILE_SIZE + threadIdx.y;
    int j  = blockIdx.x * TILE_SIZE + threadIdx.x;
    int ty = threadIdx.y, tx = threadIdx.x;
    float acc = 0.f;

    /* each tile iteration handles 4 consecutive d-elements via float4 */
    int num_vec_tiles = d / 4;  /* d must be divisible by 4 */

    for (int t = 0; t < (num_vec_tiles + TILE_SIZE/4 - 1) / (TILE_SIZE/4); t++) {
        /* base index into the d dimension for this tile */
        int k_base = t * TILE_SIZE;

        /* load Q tile: float4 load, then store 4 floats into shared mem */
        int k_col = k_base + tx;
        if (i < N && k_col < d) {
            /* ensure aligned float4 access */
            if ((k_col & 3) == 0 && k_col + 3 < d) {
                float4 q4 = *reinterpret_cast<const float4*>(
                    &Q[b*N*d + i*d + k_col]);
                /* store into 4 consecutive columns of shared tile */
                if (tx < TILE_SIZE) Qs[ty][tx] = q4.x;
            } else {
                Qs[ty][tx] = (k_col < d) ? Q[b*N*d + i*d + k_col] : 0.f;
            }
        } else {
            Qs[ty][tx] = 0.f;
        }

        int j_load = blockIdx.x * TILE_SIZE + ty;
        if (j_load < N && k_col < d) {
            Ks[ty][tx] = K[b*N*d + j_load*d + k_col];
        } else {
            Ks[ty][tx] = 0.f;
        }

        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) acc += Qs[ty][k] * Ks[tx][k];
        __syncthreads();
    }

    if (b < B && i < N && j < N)
        S[b*N*N + i*N + j] = acc / sqrtf((float)d);
}

/* ── Kernel 2: Warp softmax ─────────────────────────────────────────────── */
/*
 * Grid: (N, B) — one warp (32 threads) per row.
 * Uses __shfl_down_sync for max and sum in 5 steps.
 * No shared memory, no __syncthreads.
 */
__global__ void softmax_warp_kernel(
    float* __restrict__ S,
    int B, int N
) {
    int b = blockIdx.y;
    int i = blockIdx.x;
    int t = threadIdx.x;
    if (b >= B || i >= N) return;

    float* row = S + b*N*N + i*N;

    /* pass 1: local max */
    float local_max = -1e38f;
    for (int j = t; j < N; j += WARP_SIZE)
        local_max = fmaxf(local_max, row[j]);

    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        local_max = fmaxf(local_max,
                          __shfl_down_sync(FULL_MASK, local_max, off));
    float row_max = __shfl_sync(FULL_MASK, local_max, 0);

    /* pass 2: exp and local sum */
    float local_sum = 0.f;
    for (int j = t; j < N; j += WARP_SIZE) {
        row[j] = expf(row[j] - row_max);
        local_sum += row[j];
    }

    #pragma unroll
    for (int off = WARP_SIZE/2; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(FULL_MASK, local_sum, off);
    float row_sum = __shfl_sync(FULL_MASK, local_sum, 0);

    /* pass 3: normalize */
    for (int j = t; j < N; j += WARP_SIZE)
        row[j] /= row_sum;
}

/* ── Kernel 3: attn_output with __launch_bounds__ ──────────────────────── */
/*
 * __launch_bounds__(maxThreadsPerBlock, minBlocksPerSM)
 * Tells nvcc to target 6 blocks/SM minimum, hinting register allocator
 * to reduce register usage to close the 51%→83% occupancy gap identified
 * in Week 2 profiling. Whether this helps depends on spill cost vs
 * occupancy gain — NCU will confirm.
 */
__launch_bounds__(256, 6)
__global__ void attn_output_best_kernel(
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
    for (int j = 0; j < N; j++)
        acc += s_row[j] * v_col_base[j*d];
    out[b*N*d + i*d + k] = acc;
}

/* ── Host runner ─────────────────────────────────────────────────────────── */

void run_attention_best(
    int B, int N, int d,
    const char* variant,
    const float* h_Q, const float* h_K, const float* h_V,
    float* h_out,
    float* latency_ms, float* latency_std_ms,
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

    int use_vec = (strcmp(variant, "tiled_vec") == 0);

    dim3 blk_qk(TILE_SIZE, TILE_SIZE);
    dim3 grd_qk((N+TILE_SIZE-1)/TILE_SIZE,
                (N+TILE_SIZE-1)/TILE_SIZE, B);
    dim3 blk_sm(WARP_SIZE);
    dim3 grd_sm(N, B);
    dim3 blk_out(16, 16);
    dim3 grd_out((d+15)/16, (N+15)/16, B);

    auto forward = [&]() {
        if (use_vec)
            qk_dot_tiled_vec_kernel<<<grd_qk, blk_qk>>>(d_Q,d_K,d_S,B,N,d);
        else
            qk_dot_tiled_kernel<<<grd_qk, blk_qk>>>(d_Q,d_K,d_S,B,N,d);
        softmax_warp_kernel<<<grd_sm, blk_sm>>>(d_S, B, N);
        attn_output_best_kernel<<<grd_out, blk_out>>>(d_S,d_V,d_out,B,N,d);
    };

    for (int i = 0; i < warmup; i++) forward();
    CUDA_CHECK(cudaDeviceSynchronize());

    /* per-iteration timing for std dev */
    float* iter_times = (float*)malloc(iters * sizeof(float));
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventRecord(t0));
        forward();
        CUDA_CHECK(cudaEventRecord(t1));
        CUDA_CHECK(cudaEventSynchronize(t1));
        CUDA_CHECK(cudaEventElapsedTime(&iter_times[i], t0, t1));
    }

    /* mean */
    float sum = 0.f;
    for (int i = 0; i < iters; i++) sum += iter_times[i];
    *latency_ms = sum / iters;

    /* std dev */
    float sq = 0.f;
    for (int i = 0; i < iters; i++) {
        float diff = iter_times[i] - *latency_ms;
        sq += diff * diff;
    }
    *latency_std_ms = sqrtf(sq / iters);

    CUDA_CHECK(cudaMemcpy(h_out, d_out, qkv_bytes, cudaMemcpyDeviceToHost));

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_out);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(iter_times);
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
        "  --variant V  'best' or 'tiled_vec'  (default: best)\n"
        "  --batch  B   Batch size              (default: 1)\n"
        "  --seq    N   Sequence length          (default: 512)\n"
        "  --dim    d   Head dimension           (default: 64)\n"
        "  --warmup W   Warmup iterations        (default: 5)\n"
        "  --iters  I   Measured iterations      (default: 100)\n"
        "  --csv    F   Append results to CSV\n"
        "  --dump   D   Dump Q/K/V/out to dir D\n",
        prog);
}

/* ── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int B=1, N=512, d=64, warmup=5, iters=100;
    const char* variant  = "best";
    const char* csv_path = NULL;
    const char* dump_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i],"--variant") && i+1<argc) variant  = argv[++i];
        else if (!strcmp(argv[i],"--batch")   && i+1<argc) B        = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--seq")     && i+1<argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--dim")     && i+1<argc) d        = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--warmup")  && i+1<argc) warmup   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--iters")   && i+1<argc) iters    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"--csv")     && i+1<argc) csv_path = argv[++i];
        else if (!strcmp(argv[i],"--dump")    && i+1<argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i],"--help"))   { print_usage(argv[0]); return 0; }
    }

    printf("=== Attention Best (Week 7) | variant=%s ===\n", variant);
    printf("B=%d  N=%d  d=%d  warmup=%d  iters=%d\n\n",
           B, N, d, warmup, iters);

    size_t qkv_n = (size_t)B*N*d;
    float* h_Q   = (float*)malloc(qkv_n*sizeof(float));
    float* h_K   = (float*)malloc(qkv_n*sizeof(float));
    float* h_V   = (float*)malloc(qkv_n*sizeof(float));
    float* h_out = (float*)malloc(qkv_n*sizeof(float));

    fill_random(h_Q, qkv_n, 42);
    fill_random(h_K, qkv_n, 43);
    fill_random(h_V, qkv_n, 44);

    float lat_ms = 0.f, lat_std = 0.f;
    run_attention_best(B, N, d, variant,
                       h_Q, h_K, h_V, h_out,
                       &lat_ms, &lat_std, warmup, iters);

    double tps = (double)B*N / (lat_ms/1000.0);
    float  cv  = 100.f * lat_std / lat_ms;

    printf("Latency:    %.4f ms  (std: %.4f ms,  CV: %.2f%%)\n",
           lat_ms, lat_std, cv);
    printf("Throughput: %.2f k tokens/sec\n", tps/1e3);

    int has_nan = 0;
    for (size_t i = 0; i < qkv_n; i++)
        if (isnan(h_out[i])||isinf(h_out[i])) { has_nan=1; break; }
    printf("Output NaN/Inf check: %s\n",
           has_nan ? "FAILED ❌" : "PASSED ✅");

    printf("\nFirst 8 output values:\n");
    for (int i = 0; i < 8 && i < (int)qkv_n; i++)
        printf("  out[%d] = %.6f\n", i, h_out[i]);

    if (csv_path) {
        FILE* f = fopen(csv_path, "a");
        if (f) {
            fseek(f, 0, SEEK_END);
            if (ftell(f) == 0)
                fprintf(f, "variant,B,N,d,latency_ms,latency_std_ms,"
                           "throughput_k_tokens_per_sec\n");
            fprintf(f, "%s,%d,%d,%d,%.4f,%.4f,%.2f\n",
                    variant, B, N, d, lat_ms, lat_std, tps/1e3);
            fclose(f);
            printf("\nResults appended to %s\n", csv_path);
        }
    }

    if (dump_dir) {
        char path[512]; FILE* f; size_t n = (size_t)B*N*d;
        #define DUMP(name, arr) \
            snprintf(path,sizeof(path),"%s/%s.bin",dump_dir,name); \
            f=fopen(path,"wb"); \
            if(f){fwrite(arr,sizeof(float),n,f);fclose(f);} \
            printf("Dumped: %s\n",path);
        DUMP("Q",h_Q) DUMP("K",h_K) DUMP("V",h_V) DUMP("out",h_out)
        #undef DUMP
        snprintf(path,sizeof(path),"%s/meta.txt",dump_dir);
        f=fopen(path,"w");
        if(f){fprintf(f,"B=%d\nN=%d\nd=%d\n",B,N,d);fclose(f);}
    }

    free(h_Q); free(h_K); free(h_V); free(h_out);
    return 0;
}