# NCU Profiling Analysis — Baseline Attention Kernels
**Config: B=1, N=512, d=64 | GPU: RTX 5060 Ti (sm_120)**

---

## Summary Table

| Kernel | Duration (µs) | Achieved Occupancy | Mem Busy | DRAM Throughput | Compute Throughput | Warp Cycles/Inst |
|---|---|---|---|---|---|---|
| `qk_dot_scaled_kernel` | 107.87 | **90.56%** | 95.27% | 0.56% | 21.55% | 172.64 |
| `softmax_kernel` | **300.00** | 16.46% | 5.73% | 0.80% | 0.35% | 124.56 |
| `attn_output_kernel` | 37.12 | 50.95% | 47.14% | 7.24% | 62.85% | 14.15 |

**Total profiled duration: ~445 µs**
Softmax alone accounts for **67% of total kernel time** despite being the simplest operation.

---

## Kernel 1: `qk_dot_scaled_kernel` — L1-Bound, Not DRAM-Bound

### Key numbers
- Achieved occupancy: **90.56%** — excellent, near theoretical 100%
- Mem Busy: **95.27%** but DRAM Throughput: only **0.56%**
- L1/TEX Hit Rate: **97.36%** | L2 Hit Rate: **90.55%**
- Warp Cycles Per Issued Instruction: **172.64** — very high, indicates memory latency stalls
- SM Busy: only **7.63%** despite high occupancy

### Interpretation
The kernel is **L1/TEX cache bound**, not DRAM bound. The very high L1 hit rate (97.36%) means K tiles are being served from L1 cache rather than DRAM — this is because at N=512, d=64, the K matrix (512×64×4 = 128KB) partially fits in L1. However, the 172.64 warp cycles per instruction reveals significant memory latency stalls — warps are waiting on L1 loads even though they hit.

The low SM Busy (7.63%) despite high occupancy means threads are occupied but stalled — they are present on the SM but not issuing instructions. This is a classic **memory latency hiding failure**: warps are available but the L1 pipeline is saturated.

### What changes at larger N
At N=1024, the K matrix grows to 512KB, overflowing L1 and L2, which will push DRAM throughput up sharply. This is exactly why the latency plots showed a steep increase between N=512 and N=1024.

---

## Kernel 2: `softmax_kernel` — **Critical Bottleneck: Severe Underutilization**

### Key numbers
- Grid size: **2 blocks** | Threads: **512** | Waves per SM: **0.01**
- Achieved occupancy: **16.46%** — severely underutilized
- Duration: **300 µs** — longest of the three kernels
- SM Busy: **0.10%** — 36 SMs available, almost none used
- Eligible Warps Per Scheduler: **0.02** — schedulers almost always idle

### Interpretation
This is the most important finding in the entire profile. The softmax kernel launches only **2 thread blocks** for N=512 (one block of 256 threads per 256 rows, grid = ⌈512/256⌉ = 2). This means **only 2 out of 36 SMs are active** — 94% of the GPU sits completely idle for 300 µs.

The kernel logic itself is correct and efficient (100% branch efficiency, no divergence, 95.56% L2 hit rate), but the grid is far too small to saturate the GPU. This is a **launch configuration bug** that becomes the dominant runtime cost at this configuration.

### Fix for Phase 4 (Kernel Fusion)
This is the primary motivation for kernel fusion in Phase 4. Fusing softmax into the QKᵀ kernel eliminates the separate launch entirely. Even before fusion, the grid can be fixed by parallelizing across rows differently.

---

## Kernel 3: `attn_output_kernel` — Register-Limited Occupancy

### Key numbers
- Achieved occupancy: **50.95%** vs theoretical 83.33%
- Block Limit Registers: **5 blocks** — registers are the binding constraint
- Registers Per Thread: **42**
- L1/TEX Hit Rate: **91.49%** | L2 Hit Rate: **64.30%**
- Compute Throughput: **62.85%** — the most compute-efficient kernel
- Warp Cycles Per Issued Instruction: **14.15** — healthy, low stall cycles
- Waves Per SM: **0.71** — less than one full wave

### Interpretation
This is the healthiest kernel of the three. Compute throughput is 62.85% and warp stalls are low (14.15 cycles/instruction vs 172 for QKᵀ). The gap between theoretical (83.33%) and achieved (50.95%) occupancy is caused by **register pressure** — 42 registers per thread limits blocks per SM to 5, while the warp limit would allow 6.

The L2 hit rate drop to 64.30% (vs 90%+ for the other kernels) indicates the V matrix access pattern has lower locality — each output element reads an entire column of V[b, :, k], which is a strided access pattern with stride d=64 floats, reducing cache effectiveness.

---

## Arithmetic Intensity Estimate (for Roofline)

Using FLOPs and observed memory throughput:

| Kernel | FLOPs | Approx Bytes Moved | AI (FLOPs/Byte) | Regime |
|---|---|---|---|---|
| `qk_dot_scaled` | 33.6 MFLOPs | ~128 KB (L1-served) | ~256 | Compute-bound (at cache) |
| `softmax` | 1.3 MFLOPs | ~1 MB (S matrix) | ~1.3 | Memory-bound |
| `attn_output` | 33.6 MFLOPs | ~2 MB (S+V) | ~16 | Memory-bound |

*Note: Full DRAM-based AI will be computed in Phase 5 using exact byte counters from ncu.*

---

## Key Findings for Paper

1. **Softmax is the dominant bottleneck (67% of runtime)** due to a 2-block grid that leaves 94% of SMs idle. This is a launch configuration problem, not an algorithmic one — and is the primary motivation for kernel fusion in Phase 4.

2. **QKᵀ is L1-bound, not DRAM-bound at N=512.** The K matrix fits in L1 cache, masking the expected memory bottleneck. At N=1024 this will change as K overflows cache — consistent with the steep latency increase observed in benchmark plots.

3. **attn_output is the most efficient kernel** with 62.85% compute throughput, but occupancy is capped at 50.95% by register pressure (42 registers/thread). Reducing register usage is a potential optimization target.

4. **No memory spilling across all kernels** (local and shared memory spilling = 0) — the register file is under pressure but not overflowing to local memory.

---

## Action Items for Upcoming Phases

| Phase | Action | Motivated By |
|---|---|---|
| Phase 3 (Tiling) | Tile Q and K into shared memory | L1 saturation in QKᵀ at large N |
| Phase 4 (Fusion) | Fuse softmax into QKᵀ kernel | Softmax 2-block grid wasting 94% of GPU |
| Phase 4 (Fusion) | Fix softmax grid: parallelize over rows | Same issue |
| Phase 5 (Roofline) | Collect exact DRAM byte counts | Confirm memory-bound regime at N=1024 |