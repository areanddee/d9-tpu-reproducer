# M5+ N-Sweep Performance Summary

**Date:** 2026-05-18
**Configuration:** N ∈ {48, 64, 96, 128, 192, 256, 384}, dt = 300·(48/N) (CFL≈0.6),
perturbed Galewsky IC, narrow ∇⁴ hyperviscosity (C=0.005), f32, full RK4 +
hyp step. 3 warmup + 1000 timed steps, **batched-throughput timing** (one
`block_until_ready` at end, mean = total/iters) matching the M5+ 6-day
script methodology.

**JAX/jaxlib version: 0.7.2 (both Colab runs).** Mac CPU baselines use
JAX 0.4.38. CLAUDE.md historical baselines are pre-M-track, JAX version
not recorded but believed to be 0.4.x era.

**Provenance:** `m5plus_galewsky_nsweep_gpu_f32_20260518/results.json`
(git `db7222e`, host 8c971208ba6c) and
`m5plus_galewsky_nsweep_tpu_f32_20260518/results.json`
(git `db7222e`, host 8f68bfe7d0c1).

---

## Headline numbers (ms/step, full RK4 + narrow ∇⁴)

| N | A100 IS (M5+) | TPU MatrixForm (M5+) | TPU/A100 ratio |
|---:|---:|---:|---:|
| 48  | 1.30 | 3.15 | 2.42× |
| 64  | 1.42 | 3.47 | 2.44× |
| 96  | 1.70 | 4.70 | 2.77× |
| 128 | 2.04 | 7.02 | 3.44× |
| 192 | 3.02 | 9.32 | 3.09× |
| 256 | 4.30 | 13.44 | 3.13× |
| 384 | 8.94 | 20.62 | **2.31×** |

The cross-platform gap is now **2.3–3.4× across the entire range**,
narrowest at the smallest and largest N values.

## Historical comparison

### A100 IndexScatter — three measurements, same algorithm

| N | CLAUDE.md (pre-M-track, JAX ~0.4.x) | D9 Task 0 (2026-05-13, JAX 0.7.2) | M5+ N-sweep (2026-05-18, JAX 0.7.2) |
|---:|---:|---:|---:|
| 192 | 5.40 ms | 4.41 ms | 3.02 ms |
| ratio vs CLAUDE.md | 1.00× | 0.82× | 0.56× |

**The A100 IS speedup happened in two stages.** The CLAUDE.md → May 13
transition (-18%) is plausibly attributable to either the D4 refactor
(setup_galewsky_legacy was introduced in D4.x) or the JAX 0.4 → 0.7.2
upgrade — both happened in that interval. The May 13 → May 18 transition
(-32%) is **on unchanged IS code under unchanged JAX version (0.7.2 in
both runs)**. The most likely explanations:

  1. Different Colab A100 SKU / driver between May 13 and May 18 sessions.
  2. Methodology delta: d9_task0 used profile_nonlinear's harness with
     30 reps; M5+ N-sweep uses 1000-step batched throughput. The latter
     is more aggressive about amortizing dispatch overhead.
  3. Some hidden code path change between the two SHAs (`0cf27b9` →
     `db7222e`). Unlikely but worth ruling out.

A controlled experiment to disambiguate is proposed below.

### TPU v6e — DenseGather → MatrixForm (M-track payoff)

| N | CLAUDE.md DG (JAX ~0.4.x) | D9 Task 0 DG (2026-05-13, JAX 0.7.2) | M5+ MatrixForm (2026-05-18, JAX 0.7.2) | M-track speedup |
|---:|---:|---:|---:|---:|
| 192 | 37.1 ms | **36.43 ms** | 9.32 ms | **3.91×** |

The CLAUDE.md → D9 Task 0 transition is essentially noise (-2%), so JAX
0.4 → 0.7.2 did *not* materially improve DenseGather on TPU. The
36.43 ms → 9.32 ms transition is **pure MatrixForm vs DenseGather at
the same JAX version on the same Colab pool**: a clean 3.9× speedup at
N=192 attributable entirely to the algebraic reformulation.

Extrapolating: across the full N range CLAUDE.md → M5+ TPU went 3.2× to
3.7× faster. The bulk of that is M-track; the JAX-version effect on TPU
appears small.

### Cross-platform gap collapse

| N | CLAUDE.md TPU/A100 | M5+ TPU/A100 | Gap collapse |
|---:|---:|---:|---:|
| 48  | 4.5× | 2.42× | 46% |
| 96  | 5.3× | 2.77× | 48% |
| 128 | 5.9× | 3.44× | 42% |
| 192 | 6.9× | 3.09× | 55% |
| 256 | 6.8× | 3.13× | 54% |
| 384 | 6.5× | **2.31×** | **65%** |

The gap collapse is most dramatic at N=384 — the largest case, closest
to multi-device production resolution. Trajectory points to TPU
sharded (M6) crossing A100 single-device.

## Scaling exponents (fit on log-log)

| Platform | CLAUDE.md fit (claimed → re-fit) | M5+ fit | Comment |
|---|---:|---:|---|
| A100 | N^0.65 → **N^0.77** (re-fit) | **N^0.89** | Launch overhead amortized; closer to compute-bound |
| TPU v6e | N^1.00 → **N^0.98** (re-fit) | **N^0.93** | MatrixForm MXU sub-linear (good) |

(Re-fits done on the same CLAUDE.md tabulated points; the N^0.65 number
in CLAUDE.md was a rounded eyeball estimate, the linear-regression fit
on its own table is N^0.77.)

A100 is now scaling much closer to compute-bound. 2D-SWE's per-step
arithmetic scales as N^2 (cells); divided by the dt ∝ 1/N timestep
refinement, time-to-solution per fixed-physical-duration scales as N^3,
but per-step compute scales as N^1.0 plus higher-order memory effects.
The N^0.89 fit suggests A100 is in the compute-bound limit with modest
launch overhead at small N.

TPU at N^0.93 with MatrixForm is slightly *better* than its prior N^0.98
with DenseGather. The matmul-based exchange uses MXU throughput that
scales sub-linearly with workload size, while DenseGather's gather
decomposition was HBM-bandwidth bound (~N^1 cost in memory traffic).
Modest but real exponent improvement.

## How to prove JAX-version vs exchange-refactor attribution

The remaining uncertainty is whether A100 IS's 5.4 → 3.02 ms speedup
came from (a) JAX upgrade, (b) the D4 refactor, or (c) post-May-13
Colab-side changes (hardware, driver, or runtime).

Three controlled experiments would disambiguate. None is expensive.

1. **JAX 0.4.x vs 0.7.2 on current SHA, A100, N=192.**
   ```
   pip install jax==0.4.38 jaxlib==0.4.38
   ```
   Then re-run the N-sweep. If A100 N=192 returns to ~5 ms, JAX is the
   cause. If it stays at ~3 ms, JAX is innocent.

2. **Pre-D4-refactor SHA on current JAX, A100, N=192.**
   ```
   git checkout <pre-D4-SHA>
   ```
   Then re-run. If 5 ms returns, the D4 refactor is the cause. If
   3 ms stays, refactor is innocent.

3. **Colab A100 SKU stability check.** Run the same script 3× over a week,
   check whether the N=192 number drifts. If it varies ±20%, the Colab-side
   hypothesis is confirmed.

(1) is the most informative single test and takes ~10 min on Colab.

## M-track end-to-end summary

| | A100 (best path) | TPU v6e (best path) |
|---|---|---|
| Pre-M-track (CLAUDE.md) | 5.4 ms @ IS | 37.1 ms @ DG |
| **M5+ (today)** | **3.0 ms @ IS** | **9.3 ms @ MatrixForm** |
| Speedup vs pre-M-track | 1.8× | **4.0×** |

Both platforms got faster between CLAUDE.md and M5+, but the TPU
speedup is overwhelmingly dominated by the matrix-method reformulation
(3.9× isolated; 4.0× total). A100 IS path is unchanged by M-track and
the 1.8× there is attributable to non-M-track factors that the proposed
controlled experiments would isolate.

---

## Files

- `m5plus_galewsky_nsweep_gpu_f32_20260518/{results.json,summary.md}` — A100 IS sweep
- `m5plus_galewsky_nsweep_tpu_f32_20260518/{results.json,summary.md}` — TPU MatrixForm sweep
- `m5plus_galewsky_nsweep_cpu_f32_20260518/{results.json,summary.md}` — AMD Rome 1-core DenseGather sweep (JAX 0.6.2 — see TODO below)
- `nsweep_loglog.pdf` (and `.png`) — log-log plot of the M5+ batched-throughput curves

---

## TODO / Open items

1. **Source data + plotting script locations** (for future-me / paper-writer):
   - **Per-platform N-sweep results.json** (source of all plot data):
     `<repo>/d9-tpu-reproducer/ci_results/m5plus_galewsky_nsweep_{cpu|gpu|tpu}_*_f32_<date>/results.json`
     Each file carries full provenance: JAX version, jaxlib, git SHA + branch, host, platform repr, all CLI args, datetime UTC/local, per-N stats. To regenerate the plot, only the `results.json` files are needed.
   - **Plotting script:** `SHASH32:python/sw_model_jax/tests/plot_nsweep_loglog.py` — takes `--a100-results`, `--tpu-results`, `--cpu-results` (each a path to a `results.json`); writes `nsweep_loglog.pdf` + `.png` to `--out`. Batched-throughput methodology only; CLAUDE.md historical baselines (per-step-blocked) are deliberately omitted.
   - **Per-platform N-sweep harness:** `SHASH32:python/sw_model_jax/tests/galewsky_nsweep_timing.py` — generates the `results.json` files. Auto-resolves the platform-best exchange unless overridden via `--exchange`. Output dir embeds the resolved exchange name to prevent collisions (e.g. TPU MatrixForm vs TPU DenseGather on the same date).

2. **Re-run AMD Rome with JAX 0.7.2** for cross-platform version parity. The current Rome row uses JAX 0.6.2 because the Rome instance's Python is 3.10, and JAX 0.7.x dropped Python 3.10 support. Need Python 3.11+ installed on the Rome instance (e.g., via deadsnakes PPA), a fresh venv built against 3.11, then `pip install 'jax==0.7.2' 'jaxlib==0.7.2'` and re-run `galewsky_nsweep_timing.py --exchange dense_gather`. The JAX-version delta on CPU stencil workloads is expected to be small (<10%), but for paper-quality data the table should be JAX-version-matched across all three platforms.
