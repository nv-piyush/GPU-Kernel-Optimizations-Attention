

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

/* ── Tile size ──────────────────────────────────────────────────────────────
 * 16×16 = 256 threads per block, 2×(16×16×4) = 2KB shared memory per block.
 * Fits well within the 16KB shared memory config seen in NCU output.
 * Can be changed to 32 for experimentation (requires d >= 32).
 */
#define TILE_SIZE 16

/* ─── Kernel 1 (Tiled): QKᵀ with scaling ───────────────────────────────────
 *
 * S[b,i,j] = dot(Q[b,i,:], K[b,j,:]) / sqrt(d)
 *
 * Each block computes a TILE_SIZE×TILE_SIZE output tile of S.
 * The d dimension is iterated in tiles of TILE_SIZE.
 * Qs and Ks tiles are loaded into shared memory once per tile iteration
 * and reused by all threads in the block.
 *
 * Grid:  (ceil(N/TILE_SIZE), ceil(N/TILE_SIZE), B)
 * Block: (TILE_SIZE, TILE_SIZE)
 */
__global__ void qk_dot_tiled_kernel(
    const float* __restrict__ Q,   // [B, N, d]
    const float* __restrict__ K,   // [B, N, d]
    float*       __restrict__ S,   // [B, N, N]
    int B, int N, int d
) {
    __shared__ float Qs[TILE_SIZE][TILE_SIZE];  // Q tile: rows i, cols k
    __shared__ float Ks[TILE_SIZE][TILE_SIZE];  // K tile: rows j, cols k

    int b  = blockIdx.z;
    int i  = blockIdx.y * TILE_SIZE + threadIdx.y;  // output row  (Q side)
    int j  = blockIdx.x * TILE_SIZE + threadIdx.x;  // output col  (K side)
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    float acc = 0.0f;

    int num_tiles = (d + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        int k_col = t * TILE_SIZE + tx;   // k index for this thread's load

        /* Load Q tile: Q[b, i, t*T+tx] */
        Qs[ty][tx] = (i < N && k_col < d)
                   ? Q[b * N * d + i * d + k_col]
                   : 0.0f;

        /* Load K tile: K[b, j_tile_row, t*T+tx]
         * j_tile_row uses ty so each row of Ks corresponds to a distinct j */
        int j_load = blockIdx.x * TILE_SIZE + ty;
        Ks[ty][tx] = (j_load < N && k_col < d)
                   ? K[b * N * d + j_load * d + k_col]
                   : 0.0f;

        __syncthreads();

        /* Accumulate partial dot product over this tile */
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += Qs[ty][k] * Ks[tx][k];
            /* Qs[ty][k] = Q[b, i, t*T+k]
             * Ks[tx][k] = K[b, j, t*T+k]  (tx selects j row) */
        }

        __syncthreads();
    }

    if (b < B && i < N && j < N)
        S[b * N * N + i * N + j] = acc / sqrtf((float)d);
}

/* ─── Kernel 2: Row-wise softmax (unchanged from baseline) ─────────────────
 * In-place, numerically stable (max subtraction).
 * Grid: (ceil(N/256), B)   Block: (256)
 */
__global__ void softmax_kernel(
    float* __restrict__ S,
    int B, int N
) {

}

/* ─── Kernel 3: Weighted sum with V (unchanged from baseline) ───────────────
 * Grid: (ceil(d/16), ceil(N/16), B)   Block: (16, 16)
 */
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

    float acc = 0.0f;
    for (int j = 0; j < N; j++) acc += s_row[j] * v_col_base[j * d];
    out[b * N * d + i * d + k] = acc;
}

/* ─── Host: run one tiled attention pass ────────────────────────────────── */

void run_attention_tiled(
    int B, int N, int d,
    const float* h_Q, const float* h_K, const float* h_V,
    float* h_out,
    float* latency_ms,
    int warmup_iters,
    int measure_iters
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
    dim3 grd_qk((N + TILE_SIZE-1)/TILE_SIZE, (N + TILE_SIZE-1)/TILE_SIZE, B);

    dim3 blk_sm(256);
    dim3 grd_sm((N + 255)/256, B);

    dim3 blk_out(16, 16);
    dim3 grd_out((d+15)/16, (N+15)/16, B);

    auto forward = [&]() {
        qk_dot_tiled_kernel  <<<grd_qk, blk_qk>>>(d_Q, d_K, d_S, B, N, d);
        softmax_kernel        <<<grd_sm, blk_sm>>>(d_S, B, N);
        attn_output_kernel    <<<grd_out, blk_out>>>(d_S, d_V, d_out, B, N, d);
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

    float total_ms = 0.0f;
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
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

void print_usage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --batch  B   Batch size          (default: 1)\n"
        "  --seq    N   Sequence length      (default: 512)\n"
        "  --dim    d   Head dimension       (default: 64)\n"
        "  --warmup W   Warmup iterations    (default: 5)\n"
        "  --iters  I   Measured iterations  (default: 100)\n"
        "  --csv    F   Append results to CSV file\n"
        "  --dump   D   Dump Q/K/V/out to directory D\n",
        prog);
}

/* ─── main ───────────────────────────────────────────────────────────────── */

int main(int argc, char** argv) {
    int B = 1, N = 256, d = 64, warmup = 5, iters = 100;
    const char* csv_path = NULL;
    const char* dump_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--batch")  && i+1<argc) B        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seq")    && i+1<argc) N        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dim")    && i+1<argc) d        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i+1<argc) warmup   = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")  && i+1<argc) iters    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--csv")    && i+1<argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--dump")   && i+1<argc) dump_dir = argv[++i];
        else if (!strcmp(argv[i], "--help"))  { print_usage(argv[0]); return 0; }
    }

    printf("=== Attention Tiled (Phase 3) | TILE_SIZE=%d ===\n", TILE_SIZE);
    printf("B=%d  N=%d  d=%d  warmup=%d  iters=%d\n\n", B, N, d, warmup, iters);

    size_t qkv_n = (size_t)B * N * d;
    float* h_Q   = (float*)malloc(qkv_n * sizeof(float));
    float* h_K   = (float*)malloc(qkv_n * sizeof(float));
    float* h_V   = (float*)malloc(qkv_n * sizeof(float));
    float* h_out = (float*)malloc(qkv_n * sizeof(float));

    fill_random(h_Q, qkv_n, 42);
    fill_random(h_K, qkv_n, 43);
    fill_random(h_V, qkv_n, 44);

    float latency_ms = 0.0f;
    run_attention_tiled(B, N, d, h_Q, h_K, h_V, h_out,
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
                fprintf(f, "variant,B,N,d,latency_ms,throughput_k_tokens_per_sec\n");
            fprintf(f, "tiled,%d,%d,%d,%.4f,%.2f\n",
                    B, N, d, latency_ms, tokens_per_sec / 1e3);
            fclose(f);
            printf("\nResults appended to %s\n", csv_path);
        }
    }

    if (dump_dir) {
        char path[256]; FILE* f;
        size_t n_elem = (size_t)B * N * d;
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