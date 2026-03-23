# ╔══════════════════════════════════════════════════════════════════╗
# ║  Phase 2 — Benchmark Results Analysis (run in Colab)            ║
# ║                                                                  ║
# ║  Upload baseline_sweep.csv when prompted.                       ║
# ║  Produces 3 plots saved as PNGs for your repo and paper.        ║
# ╚══════════════════════════════════════════════════════════════════╝

# ─────────────────────────────────────────────────────────────────
# Cell 1 — Upload CSV
# ─────────────────────────────────────────────────────────────────

from google.colab import files
import pandas as pd
import io

uploaded = files.upload()
fname = list(uploaded.keys())[0]
df = pd.read_csv(io.BytesIO(uploaded[fname]))

print(f"Loaded {len(df)} rows from {fname}")
print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────
# Cell 2 — Latency vs Sequence Length
# ─────────────────────────────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# separate by head dim
for d_val in sorted(df["d"].unique()):
    sub = df[df["d"] == d_val]

    fig, ax = plt.subplots(figsize=(8, 5))

    for B_val in sorted(sub["B"].unique()):
        grp = sub[sub["B"] == B_val].sort_values("N")
        ax.plot(grp["N"], grp["latency_ms"],
                marker="o", label=f"B={B_val}")

    ax.set_xlabel("Sequence Length (N)", fontsize=12)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title(f"Baseline Latency vs Sequence Length  |  d={d_val}", fontsize=13)
    ax.set_xticks(sorted(sub["N"].unique()))
    ax.legend(title="Batch size", loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_yscale("log")           # log scale reveals O(N²) growth clearly
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    fname_out = f"latency_vs_N_d{d_val}.png"
    plt.savefig(fname_out, dpi=150)
    plt.show()
    print(f"Saved: {fname_out}")


# ─────────────────────────────────────────────────────────────────
# Cell 3 — Throughput Heatmap (B × N)
# ─────────────────────────────────────────────────────────────────

for d_val in sorted(df["d"].unique()):
    sub = df[df["d"] == d_val]

    pivot = sub.pivot(index="B", columns="N",
                      values="throughput_k_tokens_per_sec")

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Sequence Length (N)", fontsize=12)
    ax.set_ylabel("Batch Size (B)", fontsize=12)
    ax.set_title(f"Throughput (k tokens/sec)  |  d={d_val}", fontsize=13)

    # annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9,
                    color="black" if val < pivot.values.max() * 0.7 else "white")

    plt.colorbar(im, ax=ax, label="k tokens/sec")
    plt.tight_layout()
    fname_out = f"throughput_heatmap_d{d_val}.png"
    plt.savefig(fname_out, dpi=150)
    plt.show()
    print(f"Saved: {fname_out}")


# ─────────────────────────────────────────────────────────────────
# Cell 4 — Latency Scaling Analysis (verify O(N²) growth)
# ─────────────────────────────────────────────────────────────────

print("\n=== Scaling Analysis (latency ratio when N doubles) ===")
print(f"{'B':>4} {'d':>4} {'N_low':>6} {'N_high':>7} {'lat_low':>9} "
      f"{'lat_high':>10} {'ratio':>7} {'expected':>9}")

for B_val in sorted(df["B"].unique()):
    for d_val in sorted(df["d"].unique()):
        grp = df[(df["B"] == B_val) & (df["d"] == d_val)].sort_values("N")
        lats = grp["latency_ms"].values
        Ns   = grp["N"].values
        for i in range(len(Ns) - 1):
            ratio = lats[i+1] / lats[i]
            print(f"{B_val:>4} {d_val:>4} {Ns[i]:>6} {Ns[i+1]:>7} "
                  f"{lats[i]:>9.3f} {lats[i+1]:>10.3f} {ratio:>7.2f}x "
                  f"{'(~4x expected)':>9}")
        print()

# If ratios are ~4x when N doubles → O(N²) confirmed (attention is quadratic)
# If ratios are much less → memory bandwidth is already saturated at small N


print("\n=== Summary Table (for paper / report) ===\n")
summary = df[["variant","B","N","d","latency_ms","throughput_k_tokens_per_sec"]]
print(summary.to_markdown(index=False))

# Download all generated plots
from google.colab import files as colab_files
import glob
for f in glob.glob("*.png"):
    colab_files.download(f)