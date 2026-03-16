cmake_minimum_required(VERSION 3.18)
project(attention_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── GPU architecture ────────────────────────────────────────────────────────
# RTX 5060 Ti — Blackwell architecture, compute capability 12.0
set(CMAKE_CUDA_ARCHITECTURES 120)

# ── Compiler flags ──────────────────────────────────────────────────────────
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    --use_fast_math \
    -lineinfo \
    -Xptxas=-v")
# -lineinfo       → keeps source line info for Nsight profiling
# -Xptxas=-v      → prints register/smem usage per kernel at build time
# --use_fast_math → faster FP ops (acceptable for research benchmarking)

# ── Executables ─────────────────────────────────────────────────────────────

add_executable(attention_baseline attention_baseline.cu)
add_executable(attention_tiled     attention_tiled.cu)

foreach(target attention_baseline attention_tiled)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    )
endforeach()