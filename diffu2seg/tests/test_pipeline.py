"""Pipeline wiring, driven by a synthetic affinity so no SD2 is needed.

WHAT THIS CAN AND CANNOT SHOW. A hand-built graph proves the plumbing: that
prompts are seeded in the right cells, that propagation reaches the right
tokens, that a mask becomes a box with the right coordinates, that padding is
excluded. It says NOTHING about whether SD2's real attention separates CE-130
objects -- that is what tools/check_attention_separates.py is for, on the GPU,
on real images.

Keeping the two apart matters: a green suite here must never be read as "the
method works".

Run:  python -m pytest tests/test_pipeline.py -q
      python tests/test_pipeline.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig  # noqa: E402
from d2s.pipeline import segment_image  # noqa: E402
from d2s.plaplacian import plaplacian_propagate  # noqa: E402
from d2s.prompts import build_prompt_grid, f0_onehot  # noqa: E402

R = 16                      # small grid: a 128 canvas, fast and readable
CANVAS = 128


def _cfg(**kw):
    base = dict(canvas=CANVAS, prompt_stride_cells=2, mask_quantile=0.90,
                max_iter=60, tau_prop=1e-8, lam=1e-2, p=1.6)
    base.update(kw)
    return Diffu2SegConfig(**base)


def _blocky_affinity(blocks, r=R, within=1.0, across=1e-4):
    """Graph where tokens inside a block attend to each other and little else.

    Stands in for "SD2 knows these patches belong to the same object".
    """
    n = r * r
    A = np.full((n, n), across, dtype=np.float64)
    owner = -np.ones(n, dtype=int)
    for b_idx, (r0, r1, c0, c1) in enumerate(blocks):
        idx = [i * r + j for i in range(r0, r1) for j in range(c0, c1)]
        owner[idx] = b_idx
        for i in idx:
            for j in idx:
                A[i, j] = within
    A /= A.sum(axis=1, keepdims=True)
    return torch.tensor(A, dtype=torch.float32), owner


def test_two_objects_give_two_boxes():
    """Two disjoint blocks of identical appearance -> two separate boxes.

    This is the CE-130 situation in miniature: ~21 objects of ONE class per
    image, which attention cannot tell apart and connectivity can.
    """
    A, _ = _blocky_affinity([(2, 6, 2, 6), (10, 14, 10, 14)])
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)

    out = segment_image(img, valid_h=1.0, cfg=_cfg(), A=A)

    assert out["n_boxes"] == 2, f"expected 2 boxes, got {out['n_boxes']}"
    cx = sorted(b[0] for b in out["boxes"])
    assert cx[0] < 0.5 < cx[1]


def test_box_coordinates_match_the_block():
    """Block rows/cols 2..5 -> canvas [16, 48) -> cxcywh (0.25, 0.25, .25, .25)."""
    A, _ = _blocky_affinity([(2, 6, 2, 6)])
    out = segment_image(np.zeros((CANVAS, CANVAS, 3), np.uint8),
                        valid_h=1.0, cfg=_cfg(), A=A)

    assert out["n_boxes"] == 1
    cell = CANVAS / R                       # 8 px
    lo, hi = 2 * cell, 6 * cell             # outer edge of the last cell
    expect = [((lo + hi) / 2) / CANVAS, ((lo + hi) / 2) / CANVAS,
              (hi - lo) / CANVAS, (hi - lo) / CANVAS]
    assert np.allclose(out["boxes"][0], expect, atol=1e-9)


def test_padding_yields_no_prompts_no_boxes():
    """valid_h small enough that every seed row is padding -> empty, not garbage."""
    A, _ = _blocky_affinity([(12, 15, 12, 15)])
    out = segment_image(np.zeros((CANVAS, CANVAS, 3), np.uint8),
                        valid_h=0.02, cfg=_cfg(), A=A)
    assert out["n_boxes"] == 0


def test_object_in_padding_is_filtered_out():
    """A block below valid_h must not become a box even if seeds reach it."""
    A, _ = _blocky_affinity([(2, 6, 2, 6), (13, 16, 2, 6)])
    out = segment_image(np.zeros((CANVAS, CANVAS, 3), np.uint8),
                        valid_h=12.0 / R, cfg=_cfg(), A=A)

    assert out["n_boxes"] == 1, "the padding block should have been removed"
    assert out["boxes"][0][1] < 12.0 / R


def test_diagnostics_are_reported():
    """The gate reads these; absent or wrong, results cannot be interpreted."""
    A, _ = _blocky_affinity([(2, 6, 2, 6)])
    out = segment_image(np.zeros((CANVAS, CANVAS, 3), np.uint8),
                        valid_h=1.0, cfg=_cfg(), A=A)

    for key in ("n_prompts", "n_masks", "n_boxes", "n_iter", "converged",
                "f_max", "filter_info"):
        assert key in out, f"missing diagnostic: {key}"
    assert out["n_prompts"] > 0
    assert 0.0 < out["f_max"] <= 1.0
    assert set(out["filter_info"]) == {
        "n_in", "n_kept", "n_degenerate", "n_too_large", "n_in_padding"}


def test_hitting_the_iteration_cap_is_reported_not_hidden():
    """max_iter=1 cannot converge; `converged` must say so.

    Numbers from a run that hit the cap describe the cap, not the mechanism --
    the gate rejects such a run rather than reading its oracle_recall.
    """
    A, _ = _blocky_affinity([(2, 6, 2, 6)])
    out = segment_image(np.zeros((CANVAS, CANVAS, 3), np.uint8), valid_h=1.0,
                        cfg=_cfg(max_iter=1, tau_prop=1e-30), A=A)
    assert out["n_iter"] == 1
    assert out["converged"] is False


def test_p2_runs_end_to_end_through_the_same_path():
    """p=2 is gate 1's control arm, so it must traverse the identical code path.

    Only that it RUNS and stays finite is asserted here. How many boxes it
    produces is the gate's question, not this suite's -- see the next test.
    """
    A, _ = _blocky_affinity([(2, 6, 2, 6), (10, 14, 10, 14)])
    out = segment_image(np.zeros((CANVAS, CANVAS, 3), np.uint8), valid_h=1.0,
                        cfg=_cfg(p=2.0), A=A)
    assert out["n_boxes"] > 0
    assert np.isfinite(out["boxes"]).all()
    assert out["n_prompts"] == 64


def test_p_below_two_suppresses_background_bleed():
    """The edge-preserving claim, as a measurement rather than an expectation.

    Measured on this graph (weak cross-links at 1e-4, 8 of 64 prompts inside a
    block):

        p     max f on background / max f inside a block
        1.6              0.0010          -> below mask_rel_floor, dropped
        2.0              0.0803          -> above it, survives as 56 1x1 boxes

    Linear diffusion (p=2) leaks across the weak links; the sub-quadratic
    penalty does not. Roughly an 80x difference in background response.

    WHAT THIS IS NOT. A hand-built graph with a clean 1e-4 boundary is the
    easiest possible case for edge preservation. It shows the implementation
    reproduces the mechanism; it says NOTHING about whether SD2's real attention
    on CE-130 has boundaries this clean, where the median object is 4.65 cells
    across. That is exactly what tools/check_plaplacian_vs_p2.py measures, and
    this test must never be cited as an answer to it.
    """
    blocks = [(2, 6, 2, 6), (10, 14, 10, 14)]
    A, _ = _blocky_affinity(blocks)
    cells = build_prompt_grid(R, 2, valid_h=1.0)
    f0 = f0_onehot(cells, R)

    in_block = [i for i, (r, c) in enumerate(cells)
                if any(r0 <= r < r1 and c0 <= c < c1 for r0, r1, c0, c1 in blocks)]
    background = [i for i in range(len(cells)) if i not in in_block]
    assert len(in_block) == 8 and len(background) == 56

    ratio = {}
    for p in (1.6, 2.0):
        f, _, _ = plaplacian_propagate(A, f0, p=p, lam=1e-2, tau_prop=1e-8,
                                       max_iter=60)
        f = f.numpy()
        ratio[p] = f[background].max() / max(f[in_block].max(), 1e-12)

    assert ratio[1.6] < ratio[2.0], "p<2 must bleed less than linear diffusion"
    assert ratio[1.6] < 0.05, "p=1.6 background should fall under mask_rel_floor"
    assert ratio[2.0] > 0.05, "p=2.0 background should survive the floor here"


def test_affinity_size_must_match_grid():
    A, _ = _blocky_affinity([(2, 6, 2, 6)], r=8)
    try:
        segment_image(np.zeros((CANVAS, CANVAS, 3), np.uint8),
                      valid_h=1.0, cfg=_cfg(), A=A)
    except AssertionError:
        return
    raise AssertionError("a mismatched affinity size must be rejected")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
