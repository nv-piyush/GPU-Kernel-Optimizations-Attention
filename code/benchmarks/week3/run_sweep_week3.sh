#!/bin/bash
# benchmarks/run_sweep_week3.sh
#
# Runs both baseline and tiled variants across the full parameter grid
# and appends results to a combined CSV for side-by-side comparison.
#
# Usage:
#   chmod +x benchmarks/run_sweep_week3.sh
#   ./benchmarks/run_sweep_week3.sh

set -e

BINARY_BASELINE="./build/attention_baseline"
BINARY_TILED="./build/attention_tiled"
OUT="results/week3/combined_sweep.csv"
WARMUP=5
ITERS=100

BATCH_SIZES=(1 4 16 32)
SEQ_LENGTHS=(128 256 512 1024)
HEAD_DIMS=(64 128)

mkdir -p results/week3

# Write CSV header
echo "variant,B,N,d,latency_ms,throughput_k_tokens_per_sec" > "$OUT"

total=$(( ${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]} * ${#HEAD_DIMS[@]} ))
idx=0

for B in "${BATCH_SIZES[@]}"; do
  for N in "${SEQ_LENGTHS[@]}"; do
    for d in "${HEAD_DIMS[@]}"; do
      idx=$((idx + 1))
      echo -n "[$idx/$total]  B=$B  N=$N  d=$d ... "

      # Baseline
      $BINARY_BASELINE \
        --batch $B --seq $N --dim $d \
        --warmup $WARMUP --iters $ITERS \
        --csv "$OUT" > /dev/null

      # Tiled
      $BINARY_TILED \
        --batch $B --seq $N --dim $d \
        --warmup $WARMUP --iters $ITERS \
        --csv "$OUT" > /dev/null

      echo "done"
    done
  done
done

echo ""
echo "Sweep complete. Results saved to $OUT"