# Colab Runtime Environment

All Python analysis for this project runs in Google Colab.
No GPU is required — analysis only reads CSV and binary files produced by the CUDA binary.

---

## Runtime Configuration

| Setting | Value |
|---|---|
| Runtime type | CPU (no GPU needed) |
| Python version | 3.10.x (Colab default) |
| Hardware accelerator | None |

---

## Library Versions

| Library | Version | Purpose |
|---|---|---|
| `torch` | 2.6.0 | PyTorch reference computation for correctness validation |
| `numpy` | 1.26.4 | Array operations and binary file loading |
| `pandas` | 2.2.2 | CSV loading and analysis |
| `matplotlib` | 3.9.1 | Plot generation |
| `tabulate` | 0.9.0 | Markdown table output |

All libraries above are pre-installed in Colab. No `pip install` commands are needed
when running in Colab. To run locally:

```bash
pip install -r colab/requirements.txt
```

---

## How to Reproduce

### Correctness Validation (Week 1)

1. On GPU cluster: run the CUDA binary with `--dump dump/` to write
   `Q.bin`, `K.bin`, `V.bin`, `out.bin`, `meta.txt`
2. Transfer files via `scp` to local machine
3. Open `colab/validate_colab.ipynb` in Google Colab
4. Upload the 5 dump files when prompted
5. Run all cells — correctness report and error plot are generated automatically

### Benchmark Analysis (Week 2)

1. On GPU cluster: run `python benchmarks/run_sweep.py` to produce
   `results/week2/baseline_sweep.csv`
2. Transfer CSV via `scp` to local machine
3. Open `benchmarks/analyze_results.ipynb` in Google Colab
4. Upload `baseline_sweep.csv` when prompted
5. Run all cells — 4 plots are generated and auto-downloaded

---

## Relationship Between `.ipynb` and `.py` Files

| File | Purpose |
|---|---|
| `benchmarks/analyze_results.ipynb` | **Primary analysis file.** Run interactively in Colab. Generates all plots and the summary table. This is what was used to produce the figures in `results/week2/`. |
| `benchmarks/analyze_results.py` | Script version of the same analysis. Equivalent logic, runnable locally without Jupyter if needed. Both files are kept in sync. |
| `colab/validate_colab.ipynb` | Correctness validation notebook. Compares CUDA output against PyTorch reference. Used for Week 1 validation. |

The `.ipynb` is the authoritative version for Colab use. The `.py` was committed
earlier as a reference; the notebook contains the same work and should be considered
the primary deliverable.

---

## Cluster Environment (CUDA Binary)

| Setting | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5060 Ti |
| CUDA Version | 13.1 |
| Driver Version | 590.48.01 |
| Compute Capability | sm_120 (Blackwell) |
| Compiler flags | `-arch=sm_120 --use_fast_math -lineinfo -Xptxas=-v` |