# GPU Kernel Optimizations for Scaled Dot-Product Attention

**Name:** Piyush Jadhav  
**GitHub Username:** nv_piyush  

---

## Abstract

Transformer architectures dominate modern deep learning systems, and scaled dot-product attention is one of their most computationally expensive components during inference. Prior research demonstrates that attention performance on GPUs is often limited by memory bandwidth rather than arithmetic throughput. However, there is limited controlled analysis isolating the impact of specific kernel-level optimization techniques.

This project conducts an ablation study of two GPU kernel optimizations—kernel fusion and shared-memory tiling—applied to scaled dot-product attention during Transformer inference. The primary research questions are:

1. How do kernel fusion and shared-memory tiling affect latency and throughput?
2. Under what workload configurations (sequence length, batch size) do these techniques provide measurable performance gains?
3. What tradeoffs arise in occupancy, memory utilization, and scalability?

The project includes CUDA/Triton implementations, benchmarking scripts, profiling results, and a LaTeX research paper documenting experimental findings.