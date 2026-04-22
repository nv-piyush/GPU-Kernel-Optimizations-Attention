#!/bin/bash
# benchmarks/run_sweep_week7.sh
#
# Runs all variants including the new Week 7 best and tiled_vec.
# Produces results/week7/full_sweep.csv with std dev column.
#
# Variants: baseline | tiled | softmax_fixed | fused |
#           warp_softmax | best | tiled_vec

set -e

BIN_BASE="./build/attention_baseline"
BIN_TILED="./build/attention_tiled"
BIN_FUSED="./build/attention_fused"
BIN_WARP="./build/attention_warp"
BIN_BEST="./build/attention_best"
OUT="results/week7/full_sweep.csv"
WARMUP=5
ITERS=100

BATCH_SIZES=(1 4 16 32)
SEQ_LENGTHS=(128 256 512 1024)
HEAD_DIMS=(64 128)

mkdir -p results/week7
echo "variant,B,N,d,latency_ms,latency_std_ms,throughput_k_tokens_per_sec" \
    > "$OUT"

total=$(( ${#BATCH_SIZES[@]} * ${#SEQ_LENGTHS[@]} * ${#HEAD_DIMS[@]} ))
idx=0

for B in "${BATCH_SIZES[@]}"; do
  for N in "${SEQ_LENGTHS[@]}"; do
    for d in "${HEAD_DIMS[@]}"; do
      idx=$((idx+1))
      echo -n "[$idx/$total] B=$B N=$N d=$d ... "

      $BIN_BASE  --batch $B --seq $N --dim $d \
          --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_TILED --batch $B --seq $N --dim $d \
          --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_FUSED --variant softmax_fixed \
          --batch $B --seq $N --dim $d \
          --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_WARP --variant warp_softmax \
          --batch $B --seq $N --dim $d \
          --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_BEST --variant best \
          --batch $B --seq $N --dim $d \
          --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      $BIN_BEST --variant tiled_vec \
          --batch $B --seq $N --dim $d \
          --warmup $WARMUP --iters $ITERS --csv "$OUT" > /dev/null

      echo "done"
    done
  done
done

echo ""
echo "Sweep complete → $OUT"