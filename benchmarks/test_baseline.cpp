#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

extern void launch_qk(const float*, const float*, float*, int, int, int);

int main() {
    int B = 1;
    int N = 64;
    int d = 64;

    size_t size_qkv = B * N * d;
    size_t size_s = B * N * N;

    std::vector<float> h_Q(size_qkv);
    std::vector<float> h_K(size_qkv);
    std::vector<float> h_S(size_s);

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& x : h_Q) x = dist(gen);
    for (auto& x : h_K) x = dist(gen);

    float *d_Q, *d_K, *d_S;
    cudaMalloc(&d_Q, size_qkv * sizeof(float));
    cudaMalloc(&d_K, size_qkv * sizeof(float));
    cudaMalloc(&d_S, size_s * sizeof(float));

    cudaMemcpy(d_Q, h_Q.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), size_qkv * sizeof(float), cudaMemcpyHostToDevice);

    launch_qk(d_Q, d_K, d_S, B, N, d);

    cudaMemcpy(h_S.data(), d_S, size_s * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Baseline QK^T computation complete." << std::endl;
    std::cout << "Sample output value: " << h_S[0] << std::endl;

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_S);

    return 0;
}