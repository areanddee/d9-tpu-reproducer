"""
dump_d9_hlo.py — D9 Task 3: HLO capture + op counts across three variants.

Imports the three exchange variants from d9_reproducer_core.py, lowers
each via jax.jit(f).lower(*args).as_text(), counts the representative
HLO ops, saves the full pre-optimization StableHLO per variant, and
emits a structured markdown summary table.

The same script runs on CPU, Colab A100, and Colab TPU v6e — each run
produces a parallel result directory. The three summary tables drop
into paper Section 6 (TPU implementation strategies) as the HLO
evidence for the structural-mismatch argument.

Usage:
    python dump_d9_hlo.py [--N 48] [--dtype f32]
                          [--out-dir <dir>]
                          [--variants index_scatter dense_gather vmap_ds]

Output layout (under <out-dir>, default ../ci_results):
    d9_task3_hlo_<platform>_N<N>_<dtype>_<YYYYMMDD>/
        index_scatter.stablehlo
        dense_gather.stablehlo
        vmap_dynamic_slice.stablehlo
        summary.md
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp

import d9_reproducer_core as core


# Variant id → human-readable label used in markdown table columns.
VARIANT_LABEL = {
    "index_scatter": "IndexScatter",
    "dense_gather":  "DenseGather",
    "vmap_ds":       "vmap-of-DS",
}

# Variant id → on-disk filename stem for the saved HLO file.
VARIANT_FILESTEM = {
    "index_scatter": "index_scatter",
    "dense_gather":  "dense_gather",
    "vmap_ds":       "vmap_dynamic_slice",
}

# Ops we count. The first four are the structural-mismatch evidence;
# dot_general is the M-track signal (matrix-method reformulation should
# show up here); reduce + concatenate help diagnose surprise lowerings.
OPS_TO_COUNT = (
    "stablehlo.gather",
    "stablehlo.scatter",
    "stablehlo.dynamic_slice",
    "stablehlo.dynamic_update_slice",
    "stablehlo.dot_general",
    "stablehlo.reduce",
    "stablehlo.concatenate",
)


def _build_inputs(N, dtype):
    """Build synthetic state, packed flat-arrays, and DenseGather edge groups."""
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


def _function_for(variant, packed, edge_groups, dtype):
    if variant == "index_scatter":
        return lambda u, v: core.apply_index_scatter(u, v, packed, dtype)
    if variant == "dense_gather":
        return lambda u, v: core.apply_dense_gather(
            u, v, packed, edge_groups, dtype,
        )
    if variant == "vmap_ds":
        return lambda u, v: core.apply_vmap_dynamic_slice(u, v, packed, dtype)
    raise ValueError(f"unknown variant {variant!r}")


def _lower_hlo(f, u_tiles, v_tiles):
    """Pre-optimization StableHLO text."""
    return jax.jit(f).lower(u_tiles, v_tiles).as_text()


def _count_ops(hlo_text):
    counts = {op: hlo_text.count(op) for op in OPS_TO_COUNT}
    counts["total_lines"] = len(hlo_text.splitlines())
    return counts


def _make_summary_md(results, args, platform, date):
    cols = list(results.keys())
    col_labels = [VARIANT_LABEL[v] for v in cols]
    lines = []
    lines.append("# D9 Task 3 — HLO Op Counts")
    lines.append("")
    lines.append(f"**Platform:** {platform}  ")
    lines.append(f"**N:** {args.N}  ")
    lines.append(f"**dtype:** {args.dtype}  ")
    lines.append(f"**Date:** {date}  ")
    lines.append(f"**JAX:** {jax.__version__}")
    lines.append("")
    lines.append("Pre-optimization StableHLO op counts via "
                 "`jax.jit(f).lower(*args).as_text()`. Each cell is a substring "
                 "count of `stablehlo.<op>` in the lowered IR.")
    lines.append("")
    header = ["Op"] + col_labels
    sep = ["---"] + ["---:"] * len(cols)
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for op in OPS_TO_COUNT:
        row = ["`" + op + "`"] + [str(results[v][op]) for v in cols]
        lines.append("| " + " | ".join(row) + " |")
    row = ["**total HLO lines**"] + [
        f"**{results[v]['total_lines']}**" for v in cols
    ]
    lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Full StableHLO per variant is saved alongside this "
                 "summary as `<variant>.stablehlo`.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=48,
                        help="Tile resolution N (default 48).")
    parser.add_argument("--dtype", choices=("f32", "f64"), default="f32",
                        help="Floating dtype (default f32 — paper baseline).")
    parser.add_argument("--out-dir", type=Path,
                        default=HERE.parent / "ci_results",
                        help="Parent directory for the dated result dir.")
    parser.add_argument("--variants", nargs="+",
                        default=list(VARIANT_LABEL.keys()),
                        choices=list(VARIANT_LABEL.keys()),
                        help="Variants to lower (default: all three).")
    args = parser.parse_args()

    dtype = jnp.float32 if args.dtype == "f32" else jnp.float64
    if args.dtype == "f64":
        jax.config.update("jax_enable_x64", True)

    platform = jax.devices()[0].platform
    date = datetime.datetime.now().strftime("%Y%m%d")
    out_dir = (args.out_dir
               / f"d9_task3_hlo_{platform}_N{args.N}_{args.dtype}_{date}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== D9 Task 3 HLO dump ===")
    print(f"Platform: {platform}   N: {args.N}   dtype: {args.dtype}")
    print(f"JAX: {jax.__version__}")
    print(f"Output:   {out_dir}/")
    print()

    u_tiles, v_tiles, packed, edge_groups = _build_inputs(args.N, dtype)

    results = {}
    for variant in args.variants:
        print(f"  [{variant}] lowering...", flush=True)
        f = _function_for(variant, packed, edge_groups, dtype)
        hlo = _lower_hlo(f, u_tiles, v_tiles)
        (out_dir / f"{VARIANT_FILESTEM[variant]}.stablehlo").write_text(hlo)
        counts = _count_ops(hlo)
        results[variant] = counts
        print(f"  [{variant}] "
              f"lines={counts['total_lines']:>6d}  "
              f"gather={counts['stablehlo.gather']:>4d}  "
              f"scatter={counts['stablehlo.scatter']:>4d}  "
              f"dyn_slice={counts['stablehlo.dynamic_slice']:>4d}  "
              f"dyn_upd={counts['stablehlo.dynamic_update_slice']:>4d}  "
              f"dot_general={counts['stablehlo.dot_general']:>3d}")

    summary = _make_summary_md(results, args, platform, date)
    (out_dir / "summary.md").write_text(summary)
    print()
    print(summary)


if __name__ == "__main__":
    main()
