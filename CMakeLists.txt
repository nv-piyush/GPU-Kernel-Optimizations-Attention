cmake_minimum_required(VERSION 3.18)
project(AttentionBaseline CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

add_executable(test_baseline
    src/baseline/attention_baseline.cu
    benchmarks/test_baseline.cpp
)

set_target_properties(test_baseline PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)