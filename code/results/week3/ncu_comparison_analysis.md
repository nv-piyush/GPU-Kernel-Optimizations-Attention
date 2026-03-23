# NCU Profiling Comparison — Baseline vs Tiled
**Config: B=1, N=512, d=64 | GPU: RTX 5060 Ti (sm_120)**

---

## Summary: Kernel Duration

| Kernel | Baseline (µs) | Tiled (µs) | Speedup |
|---|---|---|---|
| `qk_dot_*_kernel` | 107.87 | **32.35** | **3.33×** |
| `softmax_kernel` | 300.00 | 300.38 | 1.00× (unchanged) |
| `attn_output_kernel` | 37.12 | 35.52 | 1.04× (unchanged) |
| **Total** | **~445 µs** | **~368 µs** | **1.21×** |

Tiling delivers a **3.33× speedup on the QK kernel** but overall improvement is only 1.21× because softmax (unchanged, same 2-block grid) still dominates at 300 µs. This directly quantifies why kernel fusion is the next critical step.

---

## Deep Dive: `qk_dot_tiled_kernel` vs `qk_dot_scaled_kernel`

| Metric | Baseline | Tiled | Change |
|---|---|---|---|
| Duration (µs) | 107.87 | 32.35 | **−70%** |
| Achieved Occupancy (%) | 90.56 | **91.25** | +0.69pp |
| Executed Instructions | 2,433,024 | **1,835,008** | −25% |
| Warp Cycles / Instruction | 172.64 | **66.40** | **−62%** |
| Registers / Thread | 32 | 40 | +8 |
| Static Shared Memory / Block | 0 | **2.05 KB** | confirms tiling active |
| L1/TEX Hit Rate (%) | 97.36 | **10.81** | −87pp |
| L2 Hit Rate (%) | 90.55 | **91.60** | +1pp |
| DRAM Throughput (%) | 0.56 | 1.88 | +1.32pp |
| Memory Throughput (Gbyte/s) | 2.47 | **8.27** | **+3.35×** |
| Compute Throughput (%) | 21.55 | **28.31** | +6.76pp |

### What the numbers mean

**Warp stalls dropped 62%.** The most important signal is warp cycles per instruction falling from 172.64 to 66.40. In the baseline, threads stalled waiting on L1 loads even though hit rates were high — the L1 pipeline was saturated. With tiling, shared memory serves the inner loop (the `d`-dimension dot product), which has much lower latency than L1 global loads. Fewer stalls → more instruction throughput → 3.33× faster kernel.

**L1 hit rate dropped from 97% to 11% — and that's correct.** This is the most counterintuitive result. In the baseline, L1 caches K rows naturally because threads in neighboring blocks access overlapping K rows. With tiling, Q and K are loaded into *shared memory* instead of going through L1 for the inner loop — so L1 sees far fewer requests. The 11% hit rate reflects only the initial tile loads from global memory, not the reuse within shared memory (which bypasses L1 entirely). This is expected and correct behavior.

**L2 hit rate stayed at 91.6%.** The tile loads from global memory still hit L2 well, confirming that the working set for each tile fits in L2. Good — this means DRAM pressure remains low at N=512.

**Memory throughput tripled (2.47 → 8.27 GB/s).** With tiling, the memory pipeline is used more efficiently per unit time — fewer but more bandwidth-saturating accesses — even though total data moved is similar. This reflects better memory access coalescing and fewer redundant requests.

**Instructions reduced 25%.** Tiling eliminates redundant global memory load instructions that were being issued per-thread in the baseline for overlapping K rows. Shared memory loads are cheaper and issued once per tile per block.

**Register pressure increased slightly (32→40 regs/thread)** to hold tile indices and the shared memory pointers. Block limit by registers went from 8 to 6, but achieved occupancy actually improved slightly (90.56→91.25%) because the shared memory working set is small enough (2.05 KB/block) that it doesn't reduce block limits.

---

## `softmax_kernel` — Unchanged, Still the Bottleneck

| Metric | Baseline | Tiled | Change |
|---|---|---|---|
| Duration (µs) | 300.00 | 300.38 | ≈ 0 |
| Grid size (blocks) | 2 | 2 | unchanged |
| Achieved Occupancy (%) | 16.46 | 16.77 | ≈ 0 |
| SM Busy (%) | 0.10 | 0.10 | unchanged |

As expected — the softmax kernel is identical in both variants and remains the dominant bottleneck, consuming **82% of total tiled runtime** (300 of 368 µs). The QK optimization is completely masked at the end-to-end level. This is the central motivation for Phase 4 kernel fusion.

---

## `attn_output_kernel` — Marginally Faster

| Metric | Baseline | Tiled | Change |
|---|---|---|---|
| Duration (µs) | 37.12 | 35.52 | −4.3% |
| Achieved Occupancy (%) | 50.95 | 50.30 | ≈ 0 |
| Compute Throughput (%) | 62.85 | 65.64 | +2.79pp |

The minor improvement comes from slightly warmer L2 cache state after the tiled QK pass — more of the S matrix residues are in L2 when attn_output begins. Not a meaningful optimization target.

---

## Key Takeaways for Paper

1. **Tiling achieves a 3.33× speedup on the QK kernel** by reducing warp stalls from 172→66 cycles/instruction via shared memory reuse of K tiles.

2. **End-to-end improvement is only 1.21×** because softmax (unchanged, 2-block grid) consumes 82% of tiled runtime. Tiling alone is insufficient to deliver meaningful wall-clock improvement.

3. **The L1 hit rate drop (97%→11%) is a correct and expected artifact** of traffic moving from the L1 cache path to the shared memory path, not a regression.

4. **Register pressure increased 25% (32→40)** with no occupancy penalty at this configuration, but may become a constraint at larger tile sizes.

5. **Phase 4 kernel fusion is the critical next step.** Fusing softmax into the QK kernel eliminates the 2-block grid bottleneck. Combined with tiling, the expected end-to-end improvement should be substantially higher than 1.21×.
