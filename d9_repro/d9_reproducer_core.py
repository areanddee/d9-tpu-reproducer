"""
d9_reproducer_core.py — D9 ECS Ch vector halo exchange reproducer.

Self-contained reproducer of the three Wraptor exchange variants
(IndexScatter, DenseGather, vmap-of-dynamic-slice) on a synthetic
state. No Wraptor imports. Runs with `pip install jax` plus standard
numpy.

The script:
  1. Builds a synthetic halo-point precompute (Section 3 of the
     D9 Task 1 brief).
  2. Builds a synthetic state (six tiles of u, v with halo padding).
  3. Applies all three exchange variants to the same state.
  4. Asserts bit-exact agreement in f64 across variants.
  5. Prints a one-line success message.

The synthesizer preserves the access-pattern invariants I1-I5 from
the brief: per-component point ordering, halo width 8, single source
panel per edge group, contiguous source-block bounding boxes, and
the corner-irregularity structure that defeats trivial XLA fusion.

Bit-exact f64 agreement is the gate criterion. f32 drift between
variants is expected (~1e-9 per CLAUDE.md / dense_gather.py:268) and
characterized in Task 4, not gated here.

Usage:
    python d9_reproducer_core.py [--N 48] [--seed 42]

Expected output (one line, all success):
    OK D9 reproducer core: N=48, dtype=float64, platform=<cpu|gpu|tpu>
       — all three variants agree bit-exactly.

D9 Task 1 deliverable. Pasted variants come from:
  - IndexScatter: ecs_halo_ch_vec_jax.py::_bicubic_gather_and_eval_gpu
  - DenseGather:  exchanges/dense_gather.py::_apply_edge_group + the
                  loop over edge groups in exchange_ecs_vector
  - vmap_dynamic_slice: ecs_halo_ch_vec_jax.py::_bicubic_one_point /
                  _gather_strip / _apply_strips_tpu / _scatter_strips
"""

# ---------------------------------------------------------------------------
# Imports — jax, jax.numpy, numpy ONLY. No Wraptor imports allowed.
# ---------------------------------------------------------------------------
from __future__ import annotations

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Constants. HALO is baked into Wraptor's cubed-sphere convention; all three
# variants assume halo=8 (with +6 source-index offset that pack() applies).
# ---------------------------------------------------------------------------
HALO = 8                # cells of halo padding on each side of each tile
HALO_DEPTH = 3          # depth of bicubic halo data needed (3 depth strips)
STENCIL_WIDTH = 4       # bicubic stencil is 4-wide
N_PANELS = 6            # cubed sphere has 6 panels


# ---------------------------------------------------------------------------
# Section 1: Synthetic precompute builder.
#
# Produces u_pts and v_pts lists of dicts matching the structure of the real
# _precompute output (in python/sw_model/sw_nonlinear/ecs_halo_ch_vec.py).
#
# Per-point dict has 15 fields (verified by inspection):
#   target side: tile, tgt_i, tgt_j, component
#   source side: src_panel, iu_base, ju_base, iv_base, jv_base
#   weights:     wxu[4], wyu[4], wxv[4], wyv[4], wu, wv
#
# Source indices are produced in raw `iu_base` form (pre-+6 offset). The
# pasted _pack function (Section 2) adds +6 to convert to absolute haloed-
# array indices.
#
# Synthesis honors invariants I1-I5 from the D9 Task 1 brief. See I-by-I
# notes in the function body.
# ---------------------------------------------------------------------------

# SYNTHETIC_ROUTING assigns one source panel per (target_tile, edge).
# Per Brief §3.4: values need not match real cubed-sphere topology — just
# consistent. We use a fixed permutation that ensures src_panel != tile
# for every edge (so the synthetic exercises the cross-panel pattern that
# defeats trivial XLA fusion) and gives each tile four distinct neighbors.
SYNTHETIC_ROUTING = {
    0: {'south': 1, 'north': 2, 'west': 3, 'east': 4},
    1: {'south': 2, 'north': 0, 'west': 4, 'east': 5},
    2: {'south': 0, 'north': 1, 'west': 5, 'east': 3},
    3: {'south': 4, 'north': 5, 'west': 0, 'east': 1},
    4: {'south': 5, 'north': 3, 'west': 1, 'east': 2},
    5: {'south': 3, 'north': 4, 'west': 2, 'east': 0},
}

# Bicubic Lagrange weights at the midpoint x=0.5 of the central span.
# Nodes [-1, 0, 1, 2]; sums to 1 (partition of unity). Same values for
# every halo point per Brief §3.3 — bit-exact validation is a property
# of arithmetic structure, not weight diversity.
_BICUBIC_W_AT_HALF = np.array([-0.0625, 0.5625, 0.5625, -0.0625])

# Metric rotation scalars (per Brief §3.3): fixed values exercising both
# u and v contributions per output point. Differ from each other so the
# wu*u_interp + wv*v_interp output is sensitive to both terms.
_WU = 0.7
_WV = 0.3


def _target_indices(tile, edge, depth, k, L, component, N, halo):
    """Compute (tgt_i, tgt_j) for one halo point.

    Mirrors `_build_strips`'s fixed_idx formulas exactly (verified by
    inspection of ecs_halo_ch_vec_jax.py:99-180). The N vs N+1 asymmetry
    reflects C/V staggering:
      - u: cell-centered in i (N along south/north), vertex-centered in
        j (N+1 along west/east).
      - v: vertex-centered in i (N+1 along south/north),
        cell-centered in j (N along west/east).
    """
    if edge == 'south':
        # tgt_j fixed below interior (halo region with smaller index)
        tgt_j = halo - depth                          # = 8 - depth
        tgt_i = halo + k                              # varies along strip
    elif edge == 'north':
        if component == 'u':
            # u north: fixed_j = N + 1 + j_off + 7 = N + 8 + depth
            tgt_j = N + halo + depth
        else:  # v
            # v north: fixed_j = N + j_off + 7 = N + 7 + depth
            tgt_j = N + halo - 1 + depth
        tgt_i = halo + k
    elif edge == 'west':
        tgt_i = halo - depth                          # = 8 - depth
        tgt_j = halo + k
    elif edge == 'east':
        if component == 'u':
            # u east: fixed_i = N + i_off + 7 = N + 7 + depth
            tgt_i = N + halo - 1 + depth
        else:  # v
            # v east: fixed_i = N + 1 + i_off + 7 = N + 8 + depth
            tgt_i = N + halo + depth
        tgt_j = halo + k
    else:
        raise ValueError(f"unknown edge {edge!r}")
    return tgt_i, tgt_j


def _make_strip(tile, edge, depth, L, component, src_panel, N, halo):
    """Build L per-point dicts for one strip. Honors I1, I3, I4, I5.

    Source indices are synthesized to mimic real cubed-sphere access:
      - Along-strip dimension is monotone stride-1 (interior) with
        small ±1 perturbations at the first/last 3 points (corner
        irregularity per I5).
      - Across-depth dimension uses a fixed source-tile region near the
        edge facing the target tile; depth offsets are 0/±1/±2 within
        the 4-cell bicubic stencil's range.
      - src_panel is the same for all L points of this strip and for
        all 3 depths of this (tile, edge) pair (I3).

    iu_base / ju_base / iv_base / jv_base are produced in raw form
    (without +6 offset); _pack adds the offset at packing time so
    absolute source-tile indices land in valid range.
    """
    pts = []
    # Source-side fixed-depth coordinate: place the source-block near
    # the source tile's boundary facing the target. For our purposes,
    # iu_base/ju_base just need to be valid stencil positions with
    # 4-cell room in both directions.
    #
    # Strategy: vary the along-strip source index linearly with k,
    # and put the fixed (depth) source index near the source-tile
    # edge with depth-dependent small offsets.
    src_depth_base = N - 4 - depth     # near source's far boundary, leaves
                                       # 4-cell stencil room
    # Corner perturbation: at the first and last 3 positions of the
    # strip, perturb the along-strip source index by +/-1. This is what
    # makes the gather pattern irregular and defeats XLA's stride-1
    # fusion on TPU. I5.
    def _corner_perturb(k):
        if k < 3:
            return [-1, +1, 0][k]
        if k >= L - 3:
            return [0, +1, -1][L - 1 - k]
        return 0

    for k in range(L):
        tgt_i, tgt_j = _target_indices(
            tile, edge, depth, k, L, component, N, halo,
        )
        perturb = _corner_perturb(k)
        # Along-strip source index = k + corner perturb (within [0, L-1+1])
        along = k + perturb
        # Clamp into valid source-tile range. For source-tile shape
        # (N+2h, N+1+2h) and stencil 4-wide, valid iu_base range is
        # ~[0, N+2h-4-6] = [0, N+6] after +6 offset. We keep along in
        # [0, N-1] which after +6 is [6, N+5] — well within bounds.
        along = max(0, min(L - 1, along))

        if edge in ('south', 'north'):
            # tgt_i varies, so along-strip is the i-axis on source.
            # Source tile's i is the "along" coordinate; j is the depth.
            iu_base = along
            ju_base = src_depth_base
            iv_base = along
            jv_base = src_depth_base
        else:  # west / east
            iu_base = src_depth_base
            ju_base = along
            iv_base = src_depth_base
            jv_base = along

        pts.append(dict(
            src_panel=src_panel,
            iu_base=iu_base, ju_base=ju_base,
            iv_base=iv_base, jv_base=jv_base,
            wxu=_BICUBIC_W_AT_HALF, wyu=_BICUBIC_W_AT_HALF,
            wxv=_BICUBIC_W_AT_HALF, wyv=_BICUBIC_W_AT_HALF,
            wu=_WU, wv=_WV,
            tile=tile, tgt_i=tgt_i, tgt_j=tgt_j,
            component=component,
        ))
    return pts


def build_synthetic_precompute(N, halo=HALO):
    """Return (u_pts, v_pts), per-halo-point dict lists.

    Per Brief §3 + finding F4: produce u_pts and v_pts separately (i.e.
    post-component-filter order), so the dicts feed _pack(u_pts) and
    _pack(v_pts) directly with no interleave-then-split step.

    Per-strip ordering within each list mirrors _build_strips's order
    exactly (Invariant I1):

      u_pts: per tile, j-edges (south,north × 3 depths) of length N,
             then i-edges (west,east × 3 depths) of length N+1.
      v_pts: per tile, i-edges (west,east × 3 depths) of length N,
             then j-edges (south,north × 3 depths) of length N+1.

    Point counts at N=48:
      u_pts: 6 panels × (6 j-strips × N + 6 i-strips × (N+1))
           = 6 × (6·48 + 6·49) = 6 × 582 = 3,492 points
      v_pts: same total = 3,492 points
    """
    u_pts = []
    v_pts = []

    for tile in range(N_PANELS):
        nbrs = SYNTHETIC_ROUTING[tile]

        # u component: j-edges first (length N), then i-edges (length N+1).
        for j_off in (1, 2, 3):
            u_pts.extend(_make_strip(
                tile, 'south', j_off, N, 'u',
                src_panel=nbrs['south'], N=N, halo=halo,
            ))
            u_pts.extend(_make_strip(
                tile, 'north', j_off, N, 'u',
                src_panel=nbrs['north'], N=N, halo=halo,
            ))
        for i_off in (1, 2, 3):
            u_pts.extend(_make_strip(
                tile, 'west', i_off, N + 1, 'u',
                src_panel=nbrs['west'], N=N, halo=halo,
            ))
            u_pts.extend(_make_strip(
                tile, 'east', i_off, N + 1, 'u',
                src_panel=nbrs['east'], N=N, halo=halo,
            ))

        # v component: i-edges first (length N), then j-edges (length N+1).
        for i_off in (1, 2, 3):
            v_pts.extend(_make_strip(
                tile, 'west', i_off, N, 'v',
                src_panel=nbrs['west'], N=N, halo=halo,
            ))
            v_pts.extend(_make_strip(
                tile, 'east', i_off, N, 'v',
                src_panel=nbrs['east'], N=N, halo=halo,
            ))
        for j_off in (1, 2, 3):
            v_pts.extend(_make_strip(
                tile, 'south', j_off, N + 1, 'v',
                src_panel=nbrs['south'], N=N, halo=halo,
            ))
            v_pts.extend(_make_strip(
                tile, 'north', j_off, N + 1, 'v',
                src_panel=nbrs['north'], N=N, halo=halo,
            ))

    return u_pts, v_pts


# ---------------------------------------------------------------------------
# Section 2: _pack and _build_strips — pasted directly from
# python/sw_model_jax/operators/ecs_halo_ch_vec_jax.py (lines 70-183).
#
# These produce the JAX-consumable flat arrays + strip metadata that all
# three variants consume.
# ---------------------------------------------------------------------------

def _pack(pts_list, dtype):
    """Pack per-point dict list into flat arrays for JAX runtime.

    Lifted from ecs_halo_ch_vec_jax._get_ecs_ch_vec_data's nested _pack
    helper. The +6 offset on source indices converts from the raw
    _precompute base (which uses Fortran 1-based interior indexing
    shifted by +7 internally, then -1 for 0-indexing → +6 net) to
    absolute haloed-array indices.

    Returned dict has fields the variants index against:
      src_panel, u_src_i, u_src_j, v_src_i, v_src_j  (int32)
      wxu, wyu, wxv, wyv                              (dtype, shape (M,4))
      wu, wv                                          (dtype, shape (M,))
      tgt_tile, tgt_i, tgt_j                          (int32)
      off                                             (int32 [0,1,2,3])
    """
    M = len(pts_list)
    if M == 0:
        return {}
    fdtype = np.float64 if dtype == jnp.float64 else np.float32
    src_panel = np.array([p['src_panel'] for p in pts_list], dtype=np.int32)
    u_src_i = np.array([p['iu_base'] + 6 for p in pts_list], dtype=np.int32)
    u_src_j = np.array([p['ju_base'] + 6 for p in pts_list], dtype=np.int32)
    v_src_i = np.array([p['iv_base'] + 6 for p in pts_list], dtype=np.int32)
    v_src_j = np.array([p['jv_base'] + 6 for p in pts_list], dtype=np.int32)
    wxu = np.array([p['wxu'] for p in pts_list]).astype(fdtype)
    wyu = np.array([p['wyu'] for p in pts_list]).astype(fdtype)
    wxv = np.array([p['wxv'] for p in pts_list]).astype(fdtype)
    wyv = np.array([p['wyv'] for p in pts_list]).astype(fdtype)
    wu = np.array([p['wu'] for p in pts_list]).astype(fdtype)
    wv = np.array([p['wv'] for p in pts_list]).astype(fdtype)
    tgt_tile = np.array([p['tile'] for p in pts_list], dtype=np.int32)
    tgt_i = np.array([p['tgt_i'] for p in pts_list], dtype=np.int32)
    tgt_j = np.array([p['tgt_j'] for p in pts_list], dtype=np.int32)
    off = np.arange(4, dtype=np.int32)
    return dict(
        src_panel=src_panel, u_src_i=u_src_i, u_src_j=u_src_j,
        v_src_i=v_src_i, v_src_j=v_src_j,
        wxu=wxu, wyu=wyu, wxv=wxv, wyv=wyv,
        wu=wu, wv=wv,
        tgt_tile=tgt_tile, tgt_i=tgt_i, tgt_j=tgt_j,
        off=off,
    )


def _build_strips(pts_list, N):
    """Build strip metadata for the vmap-of-dynamic-slice variant.

    Lifted from ecs_halo_ch_vec_jax._get_ecs_ch_vec_data's nested
    _build_strips helper. Each strip is a contiguous row/column segment
    of the per-component point list (start:end indices), with metadata
    for the scatter target (tile, axis 'i' or 'j', fixed_idx, vary_start,
    vary_len).

    The iteration order MUST match _precompute's per-tile order
    (Invariant I1):
      u component: per tile, j_off 1..3 × {south, north} (length N),
                   then i_off 1..3 × {west, east} (length N+1).
      v component: per tile, i_off 1..3 × {west, east} (length N),
                   then j_off 1..3 × {south, north} (length N+1).
    """
    if len(pts_list) == 0:
        return []

    comp = pts_list[0]['component']
    strips = []
    pos = 0

    for tile in range(N_PANELS):
        if comp == 'u':
            # j-edge strips: south/north × 3 depths, N points each
            for j_off in range(1, 4):
                fixed_j = 1 - j_off + 7  # = 8 - j_off (south)
                strips.append(dict(
                    start=pos, end=pos + N, tile=tile,
                    axis='j', fixed_idx=fixed_j,
                    vary_start=8, vary_len=N))
                pos += N
                fixed_j = N + 1 + j_off + 7  # = N + 8 + j_off (north)
                strips.append(dict(
                    start=pos, end=pos + N, tile=tile,
                    axis='j', fixed_idx=fixed_j,
                    vary_start=8, vary_len=N))
                pos += N
            # i-edge strips: west/east × 3 depths, N+1 points each
            for i_off in range(1, 4):
                fixed_i = 1 - i_off + 7  # = 8 - i_off (west)
                strips.append(dict(
                    start=pos, end=pos + N + 1, tile=tile,
                    axis='i', fixed_idx=fixed_i,
                    vary_start=8, vary_len=N + 1))
                pos += N + 1
                fixed_i = N + i_off + 7  # = N + 7 + i_off (east)
                strips.append(dict(
                    start=pos, end=pos + N + 1, tile=tile,
                    axis='i', fixed_idx=fixed_i,
                    vary_start=8, vary_len=N + 1))
                pos += N + 1
        else:  # v-component
            # i-edge strips: west/east × 3 depths, N points each
            for i_off in range(1, 4):
                fixed_i = 1 - i_off + 7
                strips.append(dict(
                    start=pos, end=pos + N, tile=tile,
                    axis='i', fixed_idx=fixed_i,
                    vary_start=8, vary_len=N))
                pos += N
                fixed_i = N + 1 + i_off + 7
                strips.append(dict(
                    start=pos, end=pos + N, tile=tile,
                    axis='i', fixed_idx=fixed_i,
                    vary_start=8, vary_len=N))
                pos += N
            # j-edge strips: south/north × 3 depths, N+1 points each
            for j_off in range(1, 4):
                fixed_j = 1 - j_off + 7
                strips.append(dict(
                    start=pos, end=pos + N + 1, tile=tile,
                    axis='j', fixed_idx=fixed_j,
                    vary_start=8, vary_len=N + 1))
                pos += N + 1
                fixed_j = N + j_off + 7
                strips.append(dict(
                    start=pos, end=pos + N + 1, tile=tile,
                    axis='j', fixed_idx=fixed_j,
                    vary_start=8, vary_len=N + 1))
                pos += N + 1

    assert pos == len(pts_list), \
        f"Strip count mismatch: pos={pos} vs len(pts_list)={len(pts_list)}"
    return strips


# ---------------------------------------------------------------------------
# Section 3: Synthetic state.
# ---------------------------------------------------------------------------

def build_synthetic_state(N, halo, dtype, seed):
    """Six tiles of u (cell-centered i, vertex j) and v (vertex i,
    cell-centered j), random uniform [-1, 1].

    u_tiles shape: (6, N+2h, N+1+2h)
    v_tiles shape: (6, N+1+2h, N+2h)

    These shapes are asymmetric per CLAUDE.md array-dimension table —
    NOT square. The asymmetry is what gives the strip-length invariant
    in I1 its N-vs-N+1 distinction. See Q1 in the Task 1 clarifications.
    """
    key = jax.random.PRNGKey(seed)
    ku, kv = jax.random.split(key)
    u_shape = (N_PANELS, N + 2*halo,     N + 1 + 2*halo)
    v_shape = (N_PANELS, N + 1 + 2*halo, N + 2*halo)
    u = jax.random.uniform(ku, u_shape, dtype=dtype, minval=-1.0, maxval=1.0)
    v = jax.random.uniform(kv, v_shape, dtype=dtype, minval=-1.0, maxval=1.0)
    return u, v


# ---------------------------------------------------------------------------
# Section 4: Variant 1 — IndexScatter.
#
# Source: ecs_halo_ch_vec_jax._bicubic_gather_and_eval_gpu + the
# scatter in the else-branch of ecs_ch_vec_halo_jax.
#
# Lowers to ~8 gather + ~4 scatter HLO ops per call (CLAUDE.md baseline).
# Production path on GPU. Reference for bit-exact validation.
# ---------------------------------------------------------------------------

def _bicubic_gather_and_eval_gpu(tiles_u, tiles_v, d):
    """Vectorized index-array gather for M halo points.

    Pasted verbatim from ecs_halo_ch_vec_jax.py:198. Per output
    point m: gather a 4x4 stencil from tiles_u at the source-tile
    index (sp, ui+a, uj+b) for a,b in {0..3}; multiply by the
    bicubic tensor-product weight w2d_u[m,a,b] = wxu[m,a]*wyu[m,b];
    sum over (a, b) to get u_interp[m]. Same for v. Combine via
    metric rotation: wu * u_interp + wv * v_interp.

    XLA lowers this to ~8 fused gather kernels on GPU; on TPU the
    same source decomposes into ~hundreds of small gather/scatter
    ops because TPU's lowering of `tiles_u[sp, ui, uj]` with fancy-
    index broadcasting doesn't share GPU's fusion path.
    """
    off = d['off']
    sp = d['src_panel'][:, None, None]
    ui = d['u_src_i'][:, None, None] + off[None, :, None]
    uj = d['u_src_j'][:, None, None] + off[None, None, :]
    u_blocks = tiles_u[sp, ui, uj]
    w2d_u = d['wxu'][:, :, None] * d['wyu'][:, None, :]
    u_interp = (w2d_u * u_blocks).sum(axis=(1, 2))

    vi = d['v_src_i'][:, None, None] + off[None, :, None]
    vj = d['v_src_j'][:, None, None] + off[None, None, :]
    v_blocks = tiles_v[sp, vi, vj]
    w2d_v = d['wxv'][:, :, None] * d['wyv'][:, None, :]
    v_interp = (w2d_v * v_blocks).sum(axis=(1, 2))

    return d['wu'] * u_interp + d['wv'] * v_interp


def apply_index_scatter(u_tiles, v_tiles, packed, dtype):
    """Apply ECS Ch vec halo exchange via index-array gather/scatter.

    Args:
        u_tiles: (6, N+2h, N+1+2h) source u field, haloed.
        v_tiles: (6, N+1+2h, N+2h) source v field, haloed.
        packed:  dict with keys 'u' and 'v' (from _pack), plus
                 'u_strips' and 'v_strips' (from _build_strips, not
                 used by this variant — consumed by Variant 3).

    Returns:
        (u_out, v_out): same shapes as inputs, with halo regions
        filled by the ECS bicubic + metric rotation interpolation.

    Lowers to ~8 gather + ~4 scatter HLO ops at N=48 f32 (CLAUDE.md
    baseline). Production path on GPU (the `_USE_CONTIGUOUS=False`
    default in Wraptor's ecs_ch_vec_halo_jax).
    """
    u_out = u_tiles
    v_out = v_tiles
    d_u = packed['u']
    d_v = packed['v']

    if d_u:
        u_vals = _bicubic_gather_and_eval_gpu(u_tiles, v_tiles, d_u)
        u_out = u_out.at[d_u['tgt_tile'], d_u['tgt_i'],
                         d_u['tgt_j']].set(u_vals)
    if d_v:
        v_vals = _bicubic_gather_and_eval_gpu(u_tiles, v_tiles, d_v)
        v_out = v_out.at[d_v['tgt_tile'], d_v['tgt_i'],
                         d_v['tgt_j']].set(v_vals)

    return u_out, v_out


# ---------------------------------------------------------------------------
# Section 5: Variant 2 — DenseGather.
#
# Source: exchanges/dense_gather.py::_apply_edge_group plus the loop over
# edge groups in DenseGatherExchange.exchange_ecs_vector.
#
# Pre-builds source-block bounding boxes per edge group at __init__ time;
# at exchange time, dynamic-slices the source blocks once per group, applies
# the bicubic + metric stencil, dynamic-update-slices the result. Lowers
# to ~96 dynamic_slice + ~144 dynamic_update_slice ops (verified D9 Task 0).
# Production path on TPU.
# ---------------------------------------------------------------------------

_OFF4 = np.arange(4, dtype=np.int32)


def _group_strip_indices_for_comp(comp):
    """Return list of (tile, edge_label, [strip_d1_idx, ..., strip_d3_idx]).

    24 groups per component × 2 components = 48 edge groups total.
    Pasted from dense_gather.py:63. Numbers are strip indices into the
    flat strip list returned by _build_strips — they index into the
    12-strip-per-tile layout described in I1.
    """
    groups = []
    for tile in range(N_PANELS):
        base = tile * 12
        if comp == 'u':
            # u order per tile in _build_strips:
            #   s_d1, n_d1, s_d2, n_d2, s_d3, n_d3,
            #   w_d1, e_d1, w_d2, e_d2, w_d3, e_d3
            groups.append((tile, 'south', [base+0, base+2, base+4]))
            groups.append((tile, 'north', [base+1, base+3, base+5]))
            groups.append((tile, 'west',  [base+6, base+8, base+10]))
            groups.append((tile, 'east',  [base+7, base+9, base+11]))
        else:  # v
            # v order per tile in _build_strips:
            #   w_d1, e_d1, w_d2, e_d2, w_d3, e_d3,
            #   s_d1, n_d1, s_d2, n_d2, s_d3, n_d3
            groups.append((tile, 'west',  [base+0, base+2, base+4]))
            groups.append((tile, 'east',  [base+1, base+3, base+5]))
            groups.append((tile, 'south', [base+6, base+8, base+10]))
            groups.append((tile, 'north', [base+7, base+9, base+11]))
    return groups


def _build_edge_group(pts_list, strips, strip_indices, fdtype):
    """Collect per-point data across the 3 depth-strips of one edge.

    Pasted from dense_gather.py:88. Asserts I3 (single source panel
    across all strips in the group); computes the source-block
    bounding boxes via min/max + 4 (I4 — no hand-picked dimensions).

    Each output dict has the fields _apply_edge_group consumes:
      src_panel, u_block_i0/j0/h/w, v_block_*, u_local_*, v_local_*,
      wxu, wyu, wxv, wyv, wu, wv, depths.
    """
    all_src_panel = []
    all_u_src_i = []
    all_u_src_j = []
    all_v_src_i = []
    all_v_src_j = []
    all_wxu = []
    all_wyu = []
    all_wxv = []
    all_wyv = []
    all_wu = []
    all_wv = []

    depth_info = []
    cursor = 0
    for sidx in strip_indices:
        s = strips[sidx]
        seg = pts_list[s['start']:s['end']]
        for p in seg:
            all_src_panel.append(p['src_panel'])
            # +6 offset matches _pack's convention (Wraptor halo=8 base).
            all_u_src_i.append(p['iu_base'] + 6)
            all_u_src_j.append(p['ju_base'] + 6)
            all_v_src_i.append(p['iv_base'] + 6)
            all_v_src_j.append(p['jv_base'] + 6)
            all_wxu.append(p['wxu'])
            all_wyu.append(p['wyu'])
            all_wxv.append(p['wxv'])
            all_wyv.append(p['wyv'])
            all_wu.append(p['wu'])
            all_wv.append(p['wv'])
        depth_info.append(dict(
            pt_start=cursor, pt_end=cursor + len(seg),
            tile=s['tile'], axis=s['axis'],
            fixed_idx=s['fixed_idx'],
            vary_start=s['vary_start'], vary_len=s['vary_len'],
        ))
        cursor += len(seg)

    # I3 — single source panel across all 3 depth-strips of the group.
    src_panel = int(all_src_panel[0])
    for sp in all_src_panel:
        if sp != src_panel:
            raise AssertionError(
                f"Edge group crosses panels: {set(all_src_panel)}. "
                f"Synthesis violates I3.")

    u_src_i = np.array(all_u_src_i, dtype=np.int32)
    u_src_j = np.array(all_u_src_j, dtype=np.int32)
    v_src_i = np.array(all_v_src_i, dtype=np.int32)
    v_src_j = np.array(all_v_src_j, dtype=np.int32)

    # I4 — source-block bounding boxes computed (not prescribed) via
    # min/max + 4. The +4 covers the 4×4 bicubic stencil footprint.
    u_i_lo = int(u_src_i.min())
    u_i_hi = int(u_src_i.max()) + 4
    u_j_lo = int(u_src_j.min())
    u_j_hi = int(u_src_j.max()) + 4
    v_i_lo = int(v_src_i.min())
    v_i_hi = int(v_src_i.max()) + 4
    v_j_lo = int(v_src_j.min())
    v_j_hi = int(v_src_j.max()) + 4

    return dict(
        src_panel=src_panel,
        u_block_i0=u_i_lo, u_block_j0=u_j_lo,
        u_block_h=u_i_hi - u_i_lo, u_block_w=u_j_hi - u_j_lo,
        v_block_i0=v_i_lo, v_block_j0=v_j_lo,
        v_block_h=v_i_hi - v_i_lo, v_block_w=v_j_hi - v_j_lo,
        u_local_i=(u_src_i - u_i_lo).astype(np.int32),
        u_local_j=(u_src_j - u_j_lo).astype(np.int32),
        v_local_i=(v_src_i - v_i_lo).astype(np.int32),
        v_local_j=(v_src_j - v_j_lo).astype(np.int32),
        wxu=np.array(all_wxu, dtype=fdtype),
        wyu=np.array(all_wyu, dtype=fdtype),
        wxv=np.array(all_wxv, dtype=fdtype),
        wyv=np.array(all_wyv, dtype=fdtype),
        wu=np.array(all_wu, dtype=fdtype),
        wv=np.array(all_wv, dtype=fdtype),
        depths=depth_info,
    )


def _build_dense_gather_edge_groups(pts_list, N, dtype):
    """Build the 24 edge-group structures for one component (u or v).

    Mirrors dense_gather._get_dense_gather_data. Calls _build_strips
    on the per-component point list (which asserts I1 ordering), then
    groups strips into edges via _group_strip_indices_for_comp, then
    runs _build_edge_group per edge.

    Brief v1.1 §3.2 I4 sanity check: after construction, verify every
    group has well-formed bounding boxes (H>=4 and W>=4 to fit the
    bicubic stencil; bounds within source-tile dims). Fails loudly if
    the synthesizer produced ill-formed indices.
    """
    if not pts_list:
        return []

    fdtype = np.float64 if dtype == jnp.float64 else np.float32
    comp = pts_list[0]['component']
    strips = _build_strips(pts_list, N)
    groups_meta = _group_strip_indices_for_comp(comp)
    edge_groups = [
        _build_edge_group(pts_list, strips, si, fdtype)
        for _, _, si in groups_meta
    ]

    # I4 sanity check (per Brief v1.1).
    # Source tile shapes (after halo padding):
    u_tile_h = N + 2 * HALO
    u_tile_w = N + 1 + 2 * HALO
    v_tile_h = N + 1 + 2 * HALO
    v_tile_w = N + 2 * HALO
    for idx, g in enumerate(edge_groups):
        assert g['u_block_h'] >= 4 and g['u_block_w'] >= 4, (
            f"{comp}-edge-group {idx}: u_block too small "
            f"(h={g['u_block_h']}, w={g['u_block_w']})")
        assert g['v_block_h'] >= 4 and g['v_block_w'] >= 4, (
            f"{comp}-edge-group {idx}: v_block too small "
            f"(h={g['v_block_h']}, w={g['v_block_w']})")
        assert g['u_block_i0'] >= 0 and (
            g['u_block_i0'] + g['u_block_h']) <= u_tile_h, (
            f"{comp}-edge-group {idx}: u_block i out of bounds "
            f"({g['u_block_i0']}..{g['u_block_i0']+g['u_block_h']}, "
            f"u_tile_h={u_tile_h})")
        assert g['u_block_j0'] >= 0 and (
            g['u_block_j0'] + g['u_block_w']) <= u_tile_w, (
            f"{comp}-edge-group {idx}: u_block j out of bounds")
        assert g['v_block_i0'] >= 0 and (
            g['v_block_i0'] + g['v_block_h']) <= v_tile_h, (
            f"{comp}-edge-group {idx}: v_block i out of bounds")
        assert g['v_block_j0'] >= 0 and (
            g['v_block_j0'] + g['v_block_w']) <= v_tile_w, (
            f"{comp}-edge-group {idx}: v_block j out of bounds")
    return edge_groups


def _apply_edge_group(out_tile, u_tiles, v_tiles, group, off, dtype):
    """Compute and scatter one edge group's contribution.

    Pasted verbatim from dense_gather.py:224. Uses explicit
    lax.dynamic_slice / lax.dynamic_update_slice everywhere rather
    than `tiles[sp]` + `.at[t, ...].set(...)` — the latter lowers to
    scatter/gather on some platforms (observed on CPU), which defeats
    the point.

    The reduction structure (wxu[:,:,None]*wyu[:,None,:]) intentionally
    matches _bicubic_gather_and_eval_gpu's axis order so DenseGather
    output is bit-identical to IndexScatter (an earlier einsum-based
    implementation reordered axes and drifted at ~1e-9 in f32 — see
    dense_gather.py:268 comment).
    """
    sp = group['src_panel']
    u_block = jax.lax.dynamic_slice(
        u_tiles,
        (sp, group['u_block_i0'], group['u_block_j0']),
        (1, group['u_block_h'], group['u_block_w']))[0]
    v_block = jax.lax.dynamic_slice(
        v_tiles,
        (sp, group['v_block_i0'], group['v_block_j0']),
        (1, group['v_block_h'], group['v_block_w']))[0]

    u_li = jnp.asarray(group['u_local_i'])
    u_lj = jnp.asarray(group['u_local_j'])
    v_li = jnp.asarray(group['v_local_i'])
    v_lj = jnp.asarray(group['v_local_j'])

    u_stencils = u_block[
        u_li[:, None, None] + off[None, :, None],
        u_lj[:, None, None] + off[None, None, :]]
    v_stencils = v_block[
        v_li[:, None, None] + off[None, :, None],
        v_lj[:, None, None] + off[None, None, :]]

    wxu = jnp.asarray(group['wxu'], dtype=dtype)
    wyu = jnp.asarray(group['wyu'], dtype=dtype)
    wxv = jnp.asarray(group['wxv'], dtype=dtype)
    wyv = jnp.asarray(group['wyv'], dtype=dtype)
    wu = jnp.asarray(group['wu'], dtype=dtype)
    wv = jnp.asarray(group['wv'], dtype=dtype)

    wu_2d = wxu[:, :, None] * wyu[:, None, :]
    wv_2d = wxv[:, :, None] * wyv[:, None, :]

    u_interp = (wu_2d * u_stencils).sum(axis=(1, 2))
    v_interp = (wv_2d * v_stencils).sum(axis=(1, 2))
    values = wu * u_interp + wv * v_interp

    t = group['depths'][0]['tile']
    for d in group['depths']:
        seg = values[d['pt_start']:d['pt_end']]
        vs = d['vary_start']
        vl = d['vary_len']
        fi = d['fixed_idx']
        if d['axis'] == 'j':
            update = seg.reshape(1, vl, 1)
            out_tile = jax.lax.dynamic_update_slice(
                out_tile, update, (t, vs, fi))
        else:
            update = seg.reshape(1, 1, vl)
            out_tile = jax.lax.dynamic_update_slice(
                out_tile, update, (t, fi, vs))

    return out_tile


def apply_dense_gather(u_tiles, v_tiles, packed, edge_groups, dtype):
    """Apply ECS Ch vec halo exchange via pre-built edge groups.

    Pasted from dense_gather.DenseGatherExchange.exchange_ecs_vector:314.
    Loops over the 24 u-edge groups (writing into u_out) and 24 v-edge
    groups (writing into v_out). Per Task 0 measurement at N=192 f32
    TPU v6e: this is the production path (1.45× faster than IndexScatter
    on TPU); 14× SLOWER than IndexScatter on A100 GPU. Platform-specific
    optimum — the cross-platform inversion is the D9 narrative.
    """
    off = jnp.asarray(_OFF4)
    u_out = u_tiles
    for group in edge_groups['u']:
        u_out = _apply_edge_group(
            u_out, u_tiles, v_tiles, group, off, dtype,
        )
    v_out = v_tiles
    for group in edge_groups['v']:
        v_out = _apply_edge_group(
            v_out, u_tiles, v_tiles, group, off, dtype,
        )
    return u_out, v_out


# ---------------------------------------------------------------------------
# Section 6: Variant 3 — vmap-of-dynamic-slice (falsified-on-TPU).
#
# Source: ecs_halo_ch_vec_jax.py::_bicubic_one_point + _gather_strip +
# _apply_strips_tpu + _scatter_strips, plus the if-use_tpu_path branch of
# ecs_ch_vec_halo_jax.
#
# Intended to emit contiguous dynamic_slice reads. In practice XLA's vmap
# machinery rewrites vmap(lax.dynamic_slice) back to lax.gather, producing
# ~1152 gather ops at N=48 (verified D9 Task 0). The variant is included
# in the reproducer as the pedagogical centerpiece of Section 6 of the
# notebook: "this is what the JAX-idiomatic approach suggests for TPU; it
# does not work; see HLO comparison."
# ---------------------------------------------------------------------------

def _bicubic_one_point(tiles_u, tiles_v,
                       src_panel, u_src_i, u_src_j, v_src_i, v_src_j,
                       wxu, wyu, wxv, wyv, wu, wv):
    """Bicubic + metric rotation for ONE halo point via dynamic_slice.

    Pasted from ecs_halo_ch_vec_jax.py:221. The intended pattern:
    dynamic_slice extracts a (4,4) source block at runtime; weights
    multiply; sum over the 16 cells; metric combine. When this is
    wrapped in vmap over M points (next function), XLA's vmap
    machinery rewrites lax.dynamic_slice back to lax.gather, defeating
    the goal of "contiguous slices on TPU." See Variant 3 docstring.
    """
    u_tile = tiles_u[src_panel]
    u_block = jax.lax.dynamic_slice(u_tile, (u_src_i, u_src_j), (4, 4))
    w2d_u = wxu[:, None] * wyu[None, :]
    u_interp = jnp.sum(w2d_u * u_block)

    v_tile = tiles_v[src_panel]
    v_block = jax.lax.dynamic_slice(v_tile, (v_src_i, v_src_j), (4, 4))
    w2d_v = wxv[:, None] * wyv[None, :]
    v_interp = jnp.sum(w2d_v * v_block)

    return wu * u_interp + wv * v_interp


def _gather_strip(tiles_u, tiles_v, d, start, end):
    """vmap _bicubic_one_point over one contiguous strip of halo points.

    Pasted from ecs_halo_ch_vec_jax.py:238. This is the call that XLA's
    vmap machinery rewrites — vmap(lax.dynamic_slice) becomes lax.gather,
    producing ~1152 gather ops at N=48 instead of the intended contiguous
    dynamic_slice reads.
    """
    return jax.vmap(
        lambda sp, ui, uj, vi, vj, wxu, wyu, wxv, wyv, wu, wv:
            _bicubic_one_point(tiles_u, tiles_v,
                               sp, ui, uj, vi, vj,
                               wxu, wyu, wxv, wyv, wu, wv)
    )(d['src_panel'][start:end], d['u_src_i'][start:end],
      d['u_src_j'][start:end], d['v_src_i'][start:end],
      d['v_src_j'][start:end], d['wxu'][start:end],
      d['wyu'][start:end], d['wxv'][start:end],
      d['wyv'][start:end], d['wu'][start:end], d['wv'][start:end])


def _scatter_strips(out, vals_flat, strips):
    """Write pre-gathered values using scatter-free row/column concat.

    Pasted from ecs_halo_ch_vec_jax.py:267. For each strip, read the
    full row/column, splice the new strip values in via jnp.concatenate,
    write the row/column back as one dynamic_update_slice. Avoids the
    XLA scatter decomposition that .at[].set() can trigger on TPU.
    """
    for s in strips:
        strip_vals = vals_flat[s['start']:s['end']]
        t = s['tile']
        vs = s['vary_start']
        vl = s['vary_len']
        fi = s['fixed_idx']
        if s['axis'] == 'j':
            col = out[t, :, fi]
            new_col = jnp.concatenate([col[:vs], strip_vals, col[vs+vl:]])
            out = out.at[t, :, fi].set(new_col)
        else:
            row = out[t, fi, :]
            new_row = jnp.concatenate([row[:vs], strip_vals, row[vs+vl:]])
            out = out.at[t, fi, :].set(new_row)
    return out


def apply_vmap_dynamic_slice(u_tiles, v_tiles, packed, dtype):
    """Apply ECS Ch vec halo exchange via vmap-of-dynamic-slice.

    THIS IS WHAT THE JAX-IDIOMATIC APPROACH SUGGESTS FOR TPU. IT DOES
    NOT WORK. Despite the explicit use of lax.dynamic_slice (intended
    to emit contiguous source-block reads), XLA's vmap machinery
    rewrites vmap(lax.dynamic_slice) back to lax.gather, producing
    ~1152 gather ops at N=48 vs IndexScatter's 8 — making the variant
    ~14× SLOWER than IndexScatter on TPU v6e (Task 0 measurement at
    N=192 f32: 158 ms vs 10.6 ms standalone).

    Kept in the reproducer as the pedagogical centerpiece of the
    notebook's HLO-evidence section: the Google engineers see the
    intended pattern, see the actual HLO it produces, see the
    factor-of-14 slowdown. Not a viable production path.

    Pasted from ecs_halo_ch_vec_jax.py:335-348 (the `if use_tpu_path:`
    branch of ecs_ch_vec_halo_jax).
    """
    u_out = u_tiles
    v_out = v_tiles
    d_u = packed['u']
    d_v = packed['v']
    u_strips = packed['u_strips']
    v_strips = packed['v_strips']

    if d_u:
        u_vals = jnp.concatenate([
            _gather_strip(u_tiles, v_tiles, d_u, s['start'], s['end'])
            for s in u_strips
        ])
        u_out = _scatter_strips(u_out, u_vals, u_strips)
    if d_v:
        v_vals = jnp.concatenate([
            _gather_strip(u_tiles, v_tiles, d_v, s['start'], s['end'])
            for s in v_strips
        ])
        v_out = _scatter_strips(v_out, v_vals, v_strips)

    return u_out, v_out


# ---------------------------------------------------------------------------
# Section 7: Main validation gate.
# ---------------------------------------------------------------------------

def _validate_at_N(N, seed, dtype):
    """Run the three variants at one N and assert bit-exact agreement.

    Returns nothing on success; raises AssertionError with a descriptive
    message on disagreement (which-variant, max |diff|, which field).
    """
    print(f"\n=== N={N}, dtype={dtype} ===")

    print(f"Building synthetic precompute...")
    u_pts, v_pts = build_synthetic_precompute(N)

    print(f"Building synthetic state...")
    u_tiles, v_tiles = build_synthetic_state(N, HALO, dtype, seed)

    packed = {
        'u':         _pack(u_pts, dtype),
        'v':         _pack(v_pts, dtype),
        'u_strips':  _build_strips(u_pts, N),
        'v_strips':  _build_strips(v_pts, N),
    }

    print(f"Applying IndexScatter...")
    u_is, v_is = apply_index_scatter(u_tiles, v_tiles, packed, dtype)

    print(f"Applying DenseGather...")
    edge_groups = {
        'u': _build_dense_gather_edge_groups(u_pts, N, dtype),
        'v': _build_dense_gather_edge_groups(v_pts, N, dtype),
    }
    u_dg, v_dg = apply_dense_gather(u_tiles, v_tiles, packed, edge_groups, dtype)

    print(f"Applying vmap-of-dynamic-slice...")
    u_vm, v_vm = apply_vmap_dynamic_slice(u_tiles, v_tiles, packed, dtype)

    print(f"Validating bit-exact agreement (f64) at N={N}...")
    for name, (u_a, v_a), (u_b, v_b) in [
        ("IndexScatter vs DenseGather",
         (u_is, v_is), (u_dg, v_dg)),
        ("IndexScatter vs vmap_dynamic_slice",
         (u_is, v_is), (u_vm, v_vm)),
    ]:
        if not jnp.array_equal(u_a, u_b):
            max_du = float(jnp.max(jnp.abs(u_a - u_b)))
            raise AssertionError(
                f"N={N} {name}: u disagrees, max|du| = {max_du:.3e}"
            )
        if not jnp.array_equal(v_a, v_b):
            max_dv = float(jnp.max(jnp.abs(v_a - v_b)))
            raise AssertionError(
                f"N={N} {name}: v disagrees, max|dv| = {max_dv:.3e}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--N", type=int, default=None,
        help="If given, run only this N (development). Default: run gate "
             "values 24, 48, 192.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64

    # Gate (per Brief v1.1 §5, §8 step 7): bit-exact at N ∈ {24, 48, 192}.
    #   N=24  — fast wiring/iteration check
    #   N=48  — bounding-box scaling
    #   N=192 — production resolution, cross-validates D9 Task 0 measurements
    n_values = (args.N,) if args.N is not None else (24, 48, 192)
    for N in n_values:
        _validate_at_N(N, args.seed, dtype)

    platform = jax.devices()[0].platform
    print(f"\nOK D9 reproducer core: platform={platform}\n"
          f"   all three variants agree bit-exactly at N ∈ {n_values}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
