#!/bin/bash
# benchmarks/run_sweep_week4.sh
#
# Runs all four variants across the full 32-config parameter grid.
# Produces results/week4/full_sweep.csv with columns:
#   variant, B, N, d, latency_ms, throughput_k_tokens_per_sec
#
# Variants: baseline | tiled | softmax_fixed | fused
#
# Usage:
#   chmod +x benchmarks/run_sweep_week4.sh
#   ./benchmarks/run_sweep_week4.sh

set -e

BIN_BASELINE="./build/attention_baseline"
BIN_TILED="./build/attention_tiled"
BIN_FUSED="./build/attention_fused"
OUT="results/week4/full_sweep.csv"
WARMUP=5
ITERS=100

BATCH_SIZES=(1 4 16 32)
SEQ_LENGTHS=(128 256 512 1024)
HEAD_DIMS=(64 128)

mkdir -p results/week4

echo "variant,B,N,d,latency_ms,throughput_k_tokens_per_sec" > "$OUT"

total=$(( ${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]} * ${#HEAD_DIMS[@]} ))
idx=0

for B in "${BATCH_SIZES[@]}"; do
  for N in "${SEQ_LENGTHS[@]}"; do
    for d in "${HEAD_DIMS[@]}"; do
      idx=$((idx + 1))
      echo -n "[$idx/$total]  B=$B  N=$N  d=$d ... "

      $BIN_BASELINE \
        --batch $B --seq $N --dim $d \
        --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_TILED \
        --batch $B --seq $N --dim $d \
        --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_FUSED --variant softmax_fixed \
        --batch $B --seq $N --dim $d \
        --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_FUSED --variant fused \
        --batch $B --seq $N --dim $d \
        --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      echo "done"
    done
  done
done

echo ""
echo "Sweep complete → $OUT"