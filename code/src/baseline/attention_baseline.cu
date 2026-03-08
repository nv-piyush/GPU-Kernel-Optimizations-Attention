#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define CHECK_CUDA(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    }

__global__
void qk_matmul_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    int B, int N, int d)
{
    int b = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < B && row < N && col < N) {
        float sum = 0.0f;

        for (int i = 0; i < d; ++i) {
            float q_val = Q[b * N * d + row * d + i];
            float k_val = K[b * N * d + col * d + i];
            sum += q_val * k_val;
        }

        S[b * N * N + row * N + col] = sum / sqrtf((float)d);
    }
}

void launch_qk(
    const float* Q,
    const float* K,
    float* S,
    int B, int N, int d)
{
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (N + blockDim.y - 1) / blockDim.y,
        B);

    qk_matmul_kernel<<<gridDim, blockDim>>>(Q, K, S, B, N, d);
    CHECK_CUDA(cudaDeviceSynchronize());
}