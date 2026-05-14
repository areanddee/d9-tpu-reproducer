"""
_build_d9_notebook.py — assemble d9_tpu_reproducer.ipynb via nbformat.

Stage 2: notebook is content-complete. Markdown budget ~1000 words
across the 8-section structure. Code cells are sliced from
d9_reproducer_core.py at build time so the .py file remains the single
source of truth for variant implementations.

Run:
    cd python/ci_notebooks/d9_repro && python _build_d9_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent
CORE_PATH = HERE / "d9_reproducer_core.py"


def _slice_core(start: int, end: int) -> str:
    """Return lines [start, end] (1-indexed, inclusive) from the core file."""
    lines = CORE_PATH.read_text().splitlines(keepends=True)
    return "".join(lines[start - 1:end]).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Markdown cells (~995 words total)
# ---------------------------------------------------------------------------

FRONT_MATTER_MD = """\
# A TPU Performance Pathology in a JAX Halo Exchange

**Wraptor team — 2026-05-14**

Porting a 2D shallow-water atmospheric dycore to JAX, we hit a 200× gap
on one operator: a bicubic + metric-rotation halo exchange across
cubed-sphere panel boundaries. At N=192 f32 it runs in 0.05 ms on an
NVIDIA A100 and 10.6 ms on a TPU v6e — same source, same JAX 0.7.2,
nearly identical pre-optimization StableHLO. This notebook is the
self-contained reproducer: `pip install jax`, runs on CPU in under two
minutes. We compare three implementations and ask whether the gap is a
compiler issue, a missed JAX idiom, or a fundamental architectural limit.
"""


SECTION_1_MD = """\
## 1. Setting and finding

Wraptor is a JAX port of an SBP-SAT shallow-water solver on an
equiangular cubed sphere. The operator we examine is the **ECS Ch vector
halo exchange**: bicubic interpolation plus metric rotation, filling halo
cells of each panel from neighbor-panel interiors. Cubed-sphere panel
boundaries aren't aligned with field strides, so each output point reads
a 4×4 source block at indices the JIT cannot predict from a single
stride.

**N=192, f32, single device, full RHS step:**

| Backend | Step | Exchange share |
|---|---|---|
| A100 (Colab) | 1.06 ms | 5% (0.05 ms) |
| TPU v6e (Colab) | 12.74 ms | 83% (10.6 ms) |

The 12× ratio between TPU and A100 step time is entirely attributable
to this one operator. Every other RHS component is within 3× across
backends.
"""


SECTION_2_OPS_MD = """\
## 2. The operator and its access pattern

### 2.1 What it computes

Per halo point: interpolate (u, v) from the neighbor panel's interior
to the halo cell center via a 4×4 bicubic tensor-product stencil; apply
a 2×2 metric matrix to rotate the result into the target panel's
covariant basis.

### 2.2 Access pattern

Reads: per output point, one 4×4 source block at edge/depth/index-
dependent locations. Writes: 12 strips per tile (6 tiles × 4 edges × 3
depths), each ~N or N+1 wide — ≈7,000 output points at N=48, ≈28,000
at N=192. The pattern is **2D-indexed gather over irregular index
sequences**: corner-region indices perturb away from stride-1.
"""


SECTION_2_3_MD = """\
### 2.3 Synthetic state

The reproducer's synthetic indices preserve the access-pattern invariants
(per-component point ordering, halo width 8, single source panel per
edge group, bicubic stencil footprint, corner-region perturbation). Tile
shapes are asymmetric: `u` is `(6, N+16, N+17)`; `v` is `(6, N+17, N+16)`.
"""


SECTION_3_1_MD = """\
## 3. Three implementations

### 3.1 IndexScatter — fancy-index gather/scatter

The natural JAX expression: `tiles_u[sp, ui, uj]` with broadcast index
arrays, then `.at[...].set(...)` for the scatter. Lowers to ~8 fused
gathers + ~4 scatters on A100. The same source on TPU lowers to ~470
small gathers (no fusion). **Production path on GPU. 14× faster than
DenseGather on A100; 14× slower than DenseGather on TPU v6e.**
"""


SECTION_3_2_MD = """\
### 3.2 DenseGather — pre-built edge groups

Group the 7,000 points into 48 edge groups (6 tiles × 4 edges × 2
components). Compute one source-block bounding box per group at build
time. At exchange time: one `lax.dynamic_slice` of the bounding box,
gather per-point 4×4 stencils from inside it, reduce, then
`lax.dynamic_update_slice` the per-depth output strips back. Lowers to
~96 `dynamic_slice` + ~144 `dynamic_update_slice` ops. **Production
path on TPU. 1.45× faster than IndexScatter on TPU v6e.**
"""


SECTION_3_3_MD = """\
### 3.3 vmap-of-dynamic-slice — the JAX-idiomatic TPU approach

What the JAX docs suggest for TPU: write per-point bicubic with
`lax.dynamic_slice` extracting a 4×4 source block, `vmap` over M halo
points. Intended to emit contiguous slices. **In practice:** XLA rewrites
`vmap(lax.dynamic_slice)` back to `lax.gather`, producing ~1,152 gathers
at N=48 — 14× slower than IndexScatter on TPU v6e (158 ms vs 10.6 ms at
N=192). Falsified; kept here as evidence.
"""


SECTION_4_MD = """\
## 4. Cross-platform timing

Standalone exchange wall time, N=192 f32, median of 10 runs after warm-up:

| Variant | A100 | TPU v6e |
|---|---|---|
| IndexScatter | **0.05 ms** | 154 ms |
| DenseGather | 0.7 ms | **10.6 ms** |
| vmap-of-dynamic-slice | 2.4 ms | 158 ms |

Bold = production path. Each platform's optimum is the other's
near-worst. Run cell 20 to reproduce on the current backend.
"""


SECTION_5_MD = """\
## 5. HLO evidence

Pre-optimization StableHLO line counts (DenseGather, N=192 f32) differ
by 22% between TPU (11,801 lines) and A100 (9,633 lines). Post-
optimization HLO and execution kernels diverge dramatically. The
IndexScatter line `tiles_u[sp, ui, uj]` lowers to:

| Backend | Gather ops |
|---|---|
| A100 | 8 (fused) |
| TPU v6e | ~470 (decomposed) |

Cell 19 reproduces these counts on the current backend.
"""


SECTION_6_MD = """\
## 6. Approaches tried and falsified on TPU

- **`vmap(lax.dynamic_slice)`** — XLA rewrites to gather. 14× slower.
- **`lax.scan` per-point** — fully sequential. ~50× slower.
- **DenseStacked** (stack 24 edge-blocks into one tensor) — fewer HLO
  gathers (4) yet slower than DenseGather; access-pattern regularity
  matters more than op count.
- **Einsum-based reduction** — bit-drifts at ~1e-9 in f32; abandoned.

The GPU IndexScatter pattern, despite decomposing into ~470 gathers on
TPU, still beat every "TPU-native" alternative.
"""


SECTION_7_MD = """\
## 7. Three hypotheses for the gap

1. **Compiler.** XLA's TPU lowering of fancy-index gather is suboptimal;
   a fusion pass could close most of the gap.
2. **Missed idiom.** A JAX pattern exists that lowers efficiently on
   both backends, and we haven't found it.
3. **Architectural.** TPU's MXU is fundamentally mismatched to
   irregular 2D-indexed gather; the gap is structural and only sharding
   (multi-device) can amortize it.

We don't have a strong prior. The notebook is the question.
"""


SECTION_8_MD = """\
## 8. Reproducing

- **Cell 18** — bit-exact validation at N=24 (~30 s on CPU).
- **Cell 19** — StableHLO dump for the chosen variant.
- **Cell 20** — wall-time harness for the chosen variant.
- **Cell 21** — platform + recommended-variant summary.

For the full self-test across N ∈ {24, 48, 192}, edit cell 18 to set
`n_values = (24, 48, 192)` (~2 min on CPU; longer on TPU due to compile).
"""


# ---------------------------------------------------------------------------
# Custom code cells (cells 6, 19, 20, 21, and cell-18 driver tail)
# ---------------------------------------------------------------------------

CODE_BUILD_SYNTH = """\
# f64 must be enabled before any synthetic state is built at jnp.float64.
jax.config.update("jax_enable_x64", True)

u_pts, v_pts = build_synthetic_precompute(N=48)
u_tiles, v_tiles = build_synthetic_state(N=48, halo=HALO,
                                          dtype=jnp.float64, seed=42)
print(f"u_pts: {len(u_pts):>5d} points    v_pts: {len(v_pts):>5d} points")
print(f"u_tiles: {u_tiles.shape}   v_tiles: {v_tiles.shape}")
print(f"Sample u_pts[0]: "
      f"src_panel={u_pts[0]['src_panel']}, "
      f"tile={u_pts[0]['tile']}, "
      f"tgt=({u_pts[0]['tgt_i']},{u_pts[0]['tgt_j']}), "
      f"iu_base={u_pts[0]['iu_base']}, ju_base={u_pts[0]['ju_base']}")
"""


CODE_VALIDATE_DRIVER = """\

# Fast confirmation: bit-exact at N=24 on the current backend.
jax.config.update("jax_enable_x64", True)
_validate_at_N(N=24, seed=42, dtype=jnp.float64)
print(f"\\nOK \u2014 three variants agree bit-exactly on "
      f"platform={jax.devices()[0].platform}")
"""


CODE_HLO_DUMP = """\
# StableHLO dump for the chosen variant on the current backend.
# Toggle V to one of: 'index_scatter' | 'dense_gather' | 'vmap_ds'.

V = "index_scatter"
N = 48
DTYPE = jnp.float32

u_pts, v_pts = build_synthetic_precompute(N)
u_tiles, v_tiles = build_synthetic_state(N, HALO, DTYPE, 42)
packed = {
    'u':         _pack(u_pts, DTYPE),
    'v':         _pack(v_pts, DTYPE),
    'u_strips':  _build_strips(u_pts, N),
    'v_strips':  _build_strips(v_pts, N),
}

if V == "index_scatter":
    f = lambda u, v: apply_index_scatter(u, v, packed, DTYPE)
elif V == "dense_gather":
    edge_groups = {
        'u': _build_dense_gather_edge_groups(u_pts, N, DTYPE),
        'v': _build_dense_gather_edge_groups(v_pts, N, DTYPE),
    }
    f = lambda u, v: apply_dense_gather(u, v, packed, edge_groups, DTYPE)
elif V == "vmap_ds":
    f = lambda u, v: apply_vmap_dynamic_slice(u, v, packed, DTYPE)
else:
    raise ValueError(f"unknown variant {V!r}")

hlo = jax.jit(f).lower(u_tiles, v_tiles).as_text()
print(f"Variant '{V}' HLO @ N={N} on {jax.devices()[0].platform}: "
      f"{len(hlo.splitlines())} lines")
for op in ("stablehlo.gather", "stablehlo.scatter",
           "stablehlo.dynamic_slice", "stablehlo.dynamic_update_slice"):
    print(f"  {op:<35s} {hlo.count(op):>5d}")
"""


CODE_TIMING = """\
# Wall-time harness for the chosen variant on the current backend.
# Toggle V; run the cell. First invocation pays compile + warm-up cost.

import time
import numpy as _np

V = "index_scatter"
N = 48
DTYPE = jnp.float32
ITERS = 10

u_pts, v_pts = build_synthetic_precompute(N)
u_tiles, v_tiles = build_synthetic_state(N, HALO, DTYPE, 42)
packed = {
    'u':         _pack(u_pts, DTYPE),
    'v':         _pack(v_pts, DTYPE),
    'u_strips':  _build_strips(u_pts, N),
    'v_strips':  _build_strips(v_pts, N),
}

if V == "index_scatter":
    f = jax.jit(lambda u, v: apply_index_scatter(u, v, packed, DTYPE))
elif V == "dense_gather":
    edge_groups = {
        'u': _build_dense_gather_edge_groups(u_pts, N, DTYPE),
        'v': _build_dense_gather_edge_groups(v_pts, N, DTYPE),
    }
    f = jax.jit(
        lambda u, v: apply_dense_gather(u, v, packed, edge_groups, DTYPE)
    )
elif V == "vmap_ds":
    f = jax.jit(lambda u, v: apply_vmap_dynamic_slice(u, v, packed, DTYPE))
else:
    raise ValueError(f"unknown variant {V!r}")

# Warm-up (compile + first launch)
ub, vb = f(u_tiles, v_tiles)
ub.block_until_ready(); vb.block_until_ready()

ts = []
for _ in range(ITERS):
    t0 = time.perf_counter()
    ub, vb = f(u_tiles, v_tiles)
    ub.block_until_ready(); vb.block_until_ready()
    ts.append(time.perf_counter() - t0)

print(f"'{V}' @ N={N} {_np.dtype(DTYPE).name} "
      f"on {jax.devices()[0].platform}: "
      f"median={_np.median(ts)*1e3:8.3f} ms  "
      f"min={_np.min(ts)*1e3:8.3f}  "
      f"max={_np.max(ts)*1e3:8.3f}")
"""


CODE_PLATFORM = """\
platform = jax.devices()[0].platform
devices = jax.devices()
print(f"Platform:     {platform}")
print(f"Devices:      {[str(d) for d in devices]}")
print(f"JAX version:  {jax.__version__}")
print()
print("Recommended production variant by platform (from Wraptor):")
print("  cpu : apply_dense_gather    (most regular memory access)")
print("  gpu : apply_index_scatter   (XLA fuses fancy-index gather)")
print("  tpu : apply_dense_gather    (pre-built bounding-box slices)")
"""


# ---------------------------------------------------------------------------
# Notebook assembly
# ---------------------------------------------------------------------------

def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        # Cell 1 — front matter / abstract
        nbf.v4.new_markdown_cell(FRONT_MATTER_MD),
        # Cell 2 — Section 1 (setting + finding)
        nbf.v4.new_markdown_cell(SECTION_1_MD),
        # Cell 3 — Section 2.1 + 2.2 (operator math + access pattern)
        nbf.v4.new_markdown_cell(SECTION_2_OPS_MD),
        # Cell 4 — imports, constants, synth precompute, pack/strips,
        #          synth state (lines 41-458 of d9_reproducer_core.py)
        nbf.v4.new_code_cell(_slice_core(41, 458)),
        # Cell 5 — Section 2.3 framing
        nbf.v4.new_markdown_cell(SECTION_2_3_MD),
        # Cell 6 — build synth at N=48, print shapes / sample point
        nbf.v4.new_code_cell(CODE_BUILD_SYNTH),
        # Cell 7 — Section 3.1 (IndexScatter framing)
        nbf.v4.new_markdown_cell(SECTION_3_1_MD),
        # Cell 8 — IndexScatter implementation (lines 461-535)
        nbf.v4.new_code_cell(_slice_core(461, 535)),
        # Cell 9 — Section 3.2 (DenseGather framing)
        nbf.v4.new_markdown_cell(SECTION_3_2_MD),
        # Cell 10 — DenseGather implementation (lines 538-822)
        nbf.v4.new_code_cell(_slice_core(538, 822)),
        # Cell 11 — Section 3.3 (vmap framing)
        nbf.v4.new_markdown_cell(SECTION_3_3_MD),
        # Cell 12 — vmap-of-dynamic-slice implementation (lines 825-949)
        nbf.v4.new_code_cell(_slice_core(825, 949)),
        # Cell 13 — Section 4 (cross-platform timing tables)
        nbf.v4.new_markdown_cell(SECTION_4_MD),
        # Cell 14 — Section 5 (HLO evidence)
        nbf.v4.new_markdown_cell(SECTION_5_MD),
        # Cell 15 — Section 6 (approaches tried)
        nbf.v4.new_markdown_cell(SECTION_6_MD),
        # Cell 16 — Section 7 (three hypotheses)
        nbf.v4.new_markdown_cell(SECTION_7_MD),
        # Cell 17 — Section 8 framing (reproducing the measurements)
        nbf.v4.new_markdown_cell(SECTION_8_MD),
        # Cell 18 — _validate_at_N + fast bit-exact at N=24
        nbf.v4.new_code_cell(_slice_core(956, 1006) + CODE_VALIDATE_DRIVER),
        # Cell 19 — HLO dump for chosen variant
        nbf.v4.new_code_cell(CODE_HLO_DUMP),
        # Cell 20 — wall-time harness
        nbf.v4.new_code_cell(CODE_TIMING),
        # Cell 21 — platform detect + auto-dispatch summary
        nbf.v4.new_code_cell(CODE_PLATFORM),
    ]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }
    return nb


def main() -> None:
    out = HERE / "d9_tpu_reproducer.ipynb"
    nb = build()
    nbf.write(nb, str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
