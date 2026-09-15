"""Prompt grid: one-hot seeds, padding removal, and coverage.

Run:  python -m pytest tests/test_prompts.py -q
      python tests/test_prompts.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d2s.prompts import build_prompt_grid, cells_to_canvas_xy, f0_onehot  # noqa: E402

R = 64                 # latent grid for a 512 canvas
CANVAS = 512
CELL_PX = CANVAS / R   # 8 px


def test_f0_is_exactly_onehot():
    """Each row sums to exactly 1.0 -- lam*f0 must mean the same for every prompt.

    Guards against drifting back to M2N2's bilinear-and-normalise-by-max seed,
    whose rows sum to 1.0-4.0 and would make the anchor strength per-prompt.
    """
    cells = build_prompt_grid(R, 3)
    f0 = f0_onehot(cells, R)

    assert f0.shape == (len(cells), R * R)
    assert torch.all(f0.sum(dim=1) == 1.0)
    assert torch.all((f0 == 0.0) | (f0 == 1.0)), "seeds must be hard one-hot"
    assert int((f0 > 0).sum().item()) == len(cells)


def test_f0_lands_on_the_requested_cell():
    cells = np.array([[0, 0], [5, 9], [R - 1, R - 1]])
    f0 = f0_onehot(cells, R)
    for k, (r, c) in enumerate(cells):
        assert f0[k, r * R + c].item() == 1.0


def test_padding_cells_are_dropped():
    """valid_h = 0.71 (aspect 1.41, the CE-130 median) -> nothing below row 45."""
    valid_h = 384.0 / 541.0                     # ~0.71, a real CE-130 shape
    cells = build_prompt_grid(R, 3, valid_h=valid_h)
    assert len(cells) > 0
    assert cells[:, 0].max() < valid_h * R, "a prompt was seeded in the padding"


def test_square_image_keeps_full_grid():
    """valid_h = 1.0 (a 384x384 image) -> no rows removed."""
    full = build_prompt_grid(R, 3, valid_h=1.0)
    cropped = build_prompt_grid(R, 3, valid_h=384.0 / 541.0)
    assert len(full) > len(cropped)
    n_cols = len(np.unique(full[:, 1]))
    assert len(full) == len(np.unique(full[:, 0])) * n_cols


def test_extreme_aspect_still_returns_prompts():
    """W/H = 4.99 is the measured CE-130 maximum: valid_h ~ 0.20, still usable."""
    cells = build_prompt_grid(R, 3, valid_h=384.0 / 1918.0)
    assert len(cells) > 0


def _coverage(box_cells, stride):
    """Fraction of synthetic boxes of side `box_cells` containing >=1 prompt."""
    cells = build_prompt_grid(R, stride, valid_h=1.0)
    xy = cells_to_canvas_xy(cells, R, CANVAS)
    side = box_cells * CELL_PX

    rng = np.random.default_rng(0)
    hits = 0
    n = 400
    for _ in range(n):
        cx, cy = rng.uniform(side / 2, CANVAS - side / 2, size=2)
        inside = ((np.abs(xy[:, 0] - cx) <= side / 2) &
                  (np.abs(xy[:, 1] - cy) <= side / 2))
        hits += int(inside.any())
    return hits / n


def test_grid_coverage_matches_measured_numbers():
    """Median CE-130 box short side is 4.65 cells; stride 3 measured 96.6 %."""
    assert _coverage(4.65, stride=3) >= 0.90


def test_stride6_misses_more_than_stride3():
    """Why we deviate from the paper.

    At the median SMALLEST box per image (2.24 cells), the paper's stride of 6
    steps over objects that stride 3 catches.
    """
    small = 2.24
    assert _coverage(small, stride=6) < _coverage(small, stride=3)


def test_cells_to_canvas_xy_is_cell_centre():
    xy = cells_to_canvas_xy(np.array([[0, 0], [1, 2]]), R, CANVAS)
    assert np.allclose(xy[0], [4.0, 4.0])          # centre of the first cell
    assert np.allclose(xy[1], [2 * 8 + 4, 1 * 8 + 4])


def test_stride_must_fit_the_grid():
    try:
        build_prompt_grid(R, R + 1)
    except AssertionError:
        return
    raise AssertionError("an out-of-range stride must be rejected")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
