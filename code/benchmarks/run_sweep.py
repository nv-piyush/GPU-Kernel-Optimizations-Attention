import argparse
import subprocess
import csv
import os
import sys
import itertools
import re
from datetime import datetime

# ── Parameter grid ────────────────────────────────────────────────────────────

BATCH_SIZES   = [1, 4, 16, 32]
SEQ_LENGTHS   = [128, 256, 512, 1024]
HEAD_DIMS     = [64, 128]

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--binary",  default="./build/attention_baseline")
parser.add_argument("--out",     default="results/week2/baseline_sweep.csv")
parser.add_argument("--warmup",  type=int, default=5)
parser.add_argument("--iters",   type=int, default=100)
parser.add_argument("--dump-dir", default=None,
                    help="If set, dump Q/K/V/out binaries for each config (slow)")
args = parser.parse_args()

# ── Verify binary exists ──────────────────────────────────────────────────────

if not os.path.exists(args.binary):
    print(f"[sweep] Binary not found: {args.binary}")
    print("[sweep] Build first:  mkdir build && cd build && cmake .. && make")
    sys.exit(1)

os.makedirs(os.path.dirname(args.out), exist_ok=True)

# ── CSV setup ─────────────────────────────────────────────────────────────────

FIELDS = ["variant", "B", "N", "d", "latency_ms",
          "throughput_k_tokens_per_sec", "timestamp"]

configs = list(itertools.product(BATCH_SIZES, SEQ_LENGTHS, HEAD_DIMS))
total   = len(configs)

print(f"[sweep] {total} configurations  |  warmup={args.warmup}  iters={args.iters}")
print(f"[sweep] Output → {args.out}\n")

# ── Run sweep ─────────────────────────────────────────────────────────────────

def parse_output(stdout: str):
    """Extract latency and throughput from binary stdout."""
    lat = tput = None
    for line in stdout.splitlines():
        m = re.search(r"Latency:\s+([\d.]+)\s+ms", line)
        if m: lat = float(m.group(1))
        m = re.search(r"Throughput:\s+([\d.]+)\s+k tokens/sec", line)
        if m: tput = float(m.group(1))
    return lat, tput

with open(args.out, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=FIELDS)
    writer.writeheader()

    for idx, (B, N, d) in enumerate(configs, 1):
        cmd = [
            args.binary,
            "--batch",  str(B),
            "--seq",    str(N),
            "--dim",    str(d),
            "--warmup", str(args.warmup),
            "--iters",  str(args.iters),
        ]

        print(f"[{idx:02d}/{total}]  B={B:2d}  N={N:4d}  d={d:3d} ... ", end="", flush=True)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120
            )

            if result.returncode != 0:
                print(f"FAILED\n  stderr: {result.stderr.strip()}")
                continue

            lat, tput = parse_output(result.stdout)

            if lat is None or tput is None:
                print(f"PARSE ERROR\n  stdout: {result.stdout.strip()}")
                continue

            writer.writerow({
                "variant":                    "baseline",
                "B":                          B,
                "N":                          N,
                "d":                          d,
                "latency_ms":                 round(lat,  4),
                "throughput_k_tokens_per_sec": round(tput, 2),
                "timestamp":                  datetime.utcnow().isoformat(),
            })
            csvfile.flush()  # write row immediately in case of crash mid-sweep

            print(f"{lat:.3f} ms  |  {tput:.1f} k tok/s")

        except subprocess.TimeoutExpired:
            print("TIMEOUT (>120s) — skipping")
        except Exception as e:
            print(f"ERROR: {e}")

print(f"\n[sweep] Done. Results saved to {args.out}")