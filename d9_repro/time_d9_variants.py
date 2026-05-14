"""
time_d9_variants.py — D9 Task 4: wall-clock timing across three variants.

Companion to dump_d9_hlo.py. JIT-compiles each variant from
d9_reproducer_core, performs warm-up (compile + first launch), runs
ITERS timed iterations via time.perf_counter() with block_until_ready()
on the outputs, reports median / min / max in milliseconds.

The same script runs on CPU, Colab A100, and Colab TPU v6e — each run
produces a parallel result directory. Combined with the D9.3 HLO
op-count tables, the timing tables form the empirical core of paper
Section 6 (TPU implementation strategies).

Usage:
    python time_d9_variants.py [--N 192] [--dtype f32]
                               [--iters 20] [--warmup 3]
                               [--out-dir <dir>]
                               [--variants index_scatter dense_gather vmap_ds]

Output layout (under <out-dir>, default ../ci_results):
    d9_task4_timing_<platform>_N<N>_<dtype>_<YYYYMMDD>/
        index_scatter.times       (one ms-per-iter per line, plain text)
        dense_gather.times
        vmap_dynamic_slice.times
        summary.md
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp

import d9_reproducer_core as core


VARIANT_LABEL = {
    "index_scatter": "IndexScatter",
    "dense_gather":  "DenseGather",
    "vmap_ds":       "vmap-of-DS",
}

VARIANT_FILESTEM = {
    "index_scatter": "index_scatter",
    "dense_gather":  "dense_gather",
    "vmap_ds":       "vmap_dynamic_slice",
}


def _build_inputs(N, dtype):
    u_pts, v_pts = core.build_synthetic_precompute(N)
    u_tiles, v_tiles = core.build_synthetic_state(N, core.HALO, dtype, 42)
    packed = {
        'u':        core._pack(u_pts, dtype),
        'v':        core._pack(v_pts, dtype),
        'u_strips': core._build_strips(u_pts, N),
        'v_strips': core._build_strips(v_pts, N),
    }
    edge_groups = {
        'u': core._build_dense_gather_edge_groups(u_pts, N, dtype),
        'v': core._build_dense_gather_edge_groups(v_pts, N, dtype),
    }
    return u_tiles, v_tiles, packed, edge_groups


def _jitted_for(variant, packed, edge_groups, dtype):
    if variant == "index_scatter":
        return jax.jit(
            lambda u, v: core.apply_index_scatter(u, v, packed, dtype)
        )
    if variant == "dense_gather":
        return jax.jit(
            lambda u, v: core.apply_dense_gather(
                u, v, packed, edge_groups, dtype,
            )
        )
    if variant == "vmap_ds":
        return jax.jit(
            lambda u, v: core.apply_vmap_dynamic_slice(u, v, packed, dtype)
        )
    raise ValueError(f"unknown variant {variant!r}")


def _time_variant(f, u_tiles, v_tiles, warmup, iters):
    """Returns list[float] of seconds per iteration (length=iters)."""
    # Warm-up: trigger compile + first launch. Subsequent calls hit the
    # cached compile.
    for _ in range(max(1, warmup)):
        ub, vb = f(u_tiles, v_tiles)
        ub.block_until_ready()
        vb.block_until_ready()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ub, vb = f(u_tiles, v_tiles)
        ub.block_until_ready()
        vb.block_until_ready()
        times.append(time.perf_counter() - t0)
    return times


def _make_summary_md(results, args, platform, date):
    cols = list(results.keys())
    col_labels = [VARIANT_LABEL[v] for v in cols]
    lines = []
    lines.append("# D9 Task 4 — Wall-Clock Timing")
    lines.append("")
    lines.append(f"**Platform:** {platform}  ")
    lines.append(f"**N:** {args.N}  ")
    lines.append(f"**dtype:** {args.dtype}  ")
    lines.append(f"**Warm-up iters:** {args.warmup}  ")
    lines.append(f"**Timed iters:** {args.iters}  ")
    lines.append(f"**Date:** {date}  ")
    lines.append(f"**JAX:** {jax.__version__}")
    lines.append("")
    lines.append("Each call is `f(u_tiles, v_tiles)` followed by "
                 "`block_until_ready()` on both outputs. Timing via "
                 "`time.perf_counter()`. Compile cost paid in warm-up "
                 "and excluded.")
    lines.append("")
    header = ["Statistic"] + col_labels
    sep = ["---"] + ["---:"] * len(cols)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")

    def _row(name, fn):
        cells = [name]
        for v in cols:
            ms = fn(np.asarray(results[v])) * 1e3
            cells.append(f"{ms:.4f}")
        return "| " + " | ".join(cells) + " |"

    lines.append(_row("median (ms)", np.median))
    lines.append(_row("min (ms)",    np.min))
    lines.append(_row("max (ms)",    np.max))
    lines.append(_row("mean (ms)",   np.mean))
    lines.append(_row("std (ms)",    np.std))
    lines.append("")
    lines.append("Per-iteration ms times are saved alongside this summary "
                 "as `<variant>.times` (one float per line).")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=192,
                        help="Tile resolution N (default 192 — paper baseline).")
    parser.add_argument("--dtype", choices=("f32", "f64"), default="f32",
                        help="Floating dtype (default f32).")
    parser.add_argument("--iters", type=int, default=20,
                        help="Number of timed iterations after warm-up.")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Number of warm-up iterations (untimed).")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE.parent / "ci_results",
                        help="Parent directory for the dated result dir.")
    parser.add_argument("--variants", nargs="+",
                        default=list(VARIANT_LABEL.keys()),
                        choices=list(VARIANT_LABEL.keys()),
                        help="Variants to time (default: all three).")
    args = parser.parse_args()

    dtype = jnp.float32 if args.dtype == "f32" else jnp.float64
    if args.dtype == "f64":
        jax.config.update("jax_enable_x64", True)

    platform = jax.devices()[0].platform
    date = datetime.datetime.now().strftime("%Y%m%d")
    out_dir = (args.out_dir
               / f"d9_task4_timing_{platform}_N{args.N}_{args.dtype}_{date}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== D9 Task 4 wall-clock timing ===")
    print(f"Platform: {platform}   N: {args.N}   dtype: {args.dtype}")
    print(f"Warm-up: {args.warmup}   Iters: {args.iters}")
    print(f"JAX: {jax.__version__}")
    print(f"Output: {out_dir}/")
    print()

    u_tiles, v_tiles, packed, edge_groups = _build_inputs(args.N, dtype)

    results = {}
    for variant in args.variants:
        print(f"  [{variant}] compiling + warm-up...", flush=True)
        f = _jitted_for(variant, packed, edge_groups, dtype)
        times = _time_variant(f, u_tiles, v_tiles, args.warmup, args.iters)
        results[variant] = times
        arr = np.asarray(times) * 1e3
        with (out_dir / f"{VARIANT_FILESTEM[variant]}.times").open("w") as fh:
            for t in arr:
                fh.write(f"{t:.6f}\n")
        print(f"  [{variant}] "
              f"median={np.median(arr):8.4f} ms  "
              f"min={np.min(arr):8.4f}  "
              f"max={np.max(arr):8.4f}  "
              f"std={np.std(arr):7.4f}")

    summary = _make_summary_md(results, args, platform, date)
    (out_dir / "summary.md").write_text(summary)
    print()
    print(summary)


if __name__ == "__main__":
    main()
