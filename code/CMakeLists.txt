cmake_minimum_required(VERSION 3.18)
project(attention_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── GPU architecture ────────────────────────────────────────────────────────
# RTX 5060 Ti is Blackwell (sm_120), requires CUDA 12.8+
# If your nvcc version is older, lower this to 89 (Ada) or 86 (Ampere)
# Check with: nvcc --version
set(CMAKE_CUDA_ARCHITECTURES 89)

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
add_executable(attention_fused     attention_fused.cu)
add_executable(attention_warp      attention_warp.cu)
add_executable(attention_best      attention_best.cu)

foreach(target attention_baseline attention_tiled attention_fused
                attention_warp attention_best)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
    )
endforeach()