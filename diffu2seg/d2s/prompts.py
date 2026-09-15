"""The grid of one-hot point prompts that seeds propagation.

This replaces the human in M2N2. M2N2 is INTERACTIVE: a person clicks, and the
four scoring functions that pick its cut threshold (s_prior, s_edge, s_pos,
s_neg) read those clicks back. Take the human away and two of the four become
meaningless, which is why Diffuse2Seg drops that whole mechanism and lets
clusters compete by argmax instead. What survives is the seeding idea, now on a
regular grid -- exactly what SAM's automatic mask generator does.

                    WHY THE STRIDE IS 3 AND NOT THE PAPER'S 6

Measured on 200 CE-130 val images (canvas 512, r=64, so one cell is 8 px):

    stride  prompts  in padding  boxes with >=1 prompt
      2       1024      20 %            99.0 %
      3        441      22 %            96.6 %     <- chosen
      4        256      19 %            91.5 %

The median box short side is 4.65 cells and the median SMALLEST box per image
is 2.24 cells, so the paper's stride of 6 steps clean over small objects. A
prompt that never lands inside an object cannot produce a mask for it, and no
downstream stage recovers that.

                          WHY PADDING MUST BE DROPPED

Every CE-130 image is exactly 384 px tall and 384-1918 wide, so the
aspect-preserving resize always leaves padding at the BOTTOM -- about 29 % of
the canvas at the median aspect ratio of 1.41. A prompt seeded on that flat
CLIP-mean grey propagates freely across it and yields one huge mask covering the
padding. `valid_h` from scale_to_canvas is the boundary, and it is the only
thing standing between the pipeline and a pile of garbage boxes.
"""

import numpy as np
import torch

__all__ = ["build_prompt_grid", "f0_onehot", "cells_to_canvas_xy"]


def build_prompt_grid(grid_r, stride_cells, valid_h=1.0, min_valid_frac=1.0,
                     valid_w=1.0):
    """Regular grid of seed cells, with padding cells dropped.

    Args:
        grid_r:         latent grid side (64 for a 512 canvas).
        stride_cells:   spacing in cells.
        valid_h:        fraction of canvas height that is real image.
        min_valid_frac: how much of a cell must be inside the real image.
                        1.0 = the whole cell.
        valid_w:        fraction of canvas WIDTH that is real image. Defaults to
                        1.0, which is exactly right for CE-130: every image
                        there is 384 tall and at least that wide, so W >= H and
                        padding only ever lands at the bottom. COCO has portrait
                        images too (427x640 is common), where padding lands on
                        the RIGHT instead -- without this, seeds would be
                        planted on flat grey and propagate into one huge mask.

    Returns:
        (K, 2) int array of (row, col) cell indices.

    The grid is offset by stride//2 so seeds sit near cell centres rather than
    hugging the top-left edge.
    """
    assert 1 <= stride_cells < grid_r, f"stride {stride_cells} out of range for r={grid_r}"
    off = stride_cells // 2
    rows = np.arange(off, grid_r, stride_cells)
    cols = np.arange(off, grid_r, stride_cells)

    # A cell spans rows [r, r+1) in cell units; require enough of it inside.
    keep_r = (rows + min_valid_frac) <= valid_h * grid_r + 1e-9
    keep_c = (cols + min_valid_frac) <= valid_w * grid_r + 1e-9
    rows, cols = rows[keep_r], cols[keep_c]

    if len(rows) == 0:                       # degenerate aspect ratio
        rows = np.array([0])
    if len(cols) == 0:
        cols = np.array([0])

    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int64)


def f0_onehot(cells, grid_r, device="cpu", dtype=torch.float32):
    """(K, N) one-hot seeds -- exactly one 1.0 per row.

    DELIBERATELY NOT M2N2's create_single_point_heatmap. That function splits a
    point bilinearly over four cells and then normalises by MAX, so a row sums
    to anywhere between 1.0 and 4.0 depending on where the click fell inside a
    cell. Here f0 is multiplied by `lam` as the anchor term, so a varying row
    sum would silently make the anchor strength per-prompt -- lam would no
    longer mean one thing. Our prompts sit on cell indices by construction, so
    interpolation would buy nothing anyway.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    K, N = len(cells), grid_r * grid_r
    flat = cells[:, 0] * grid_r + cells[:, 1]
    assert flat.min() >= 0 and flat.max() < N, "prompt cell outside the grid"

    f0 = torch.zeros(K, N, device=device, dtype=dtype)
    f0[torch.arange(K), torch.from_numpy(flat)] = 1.0
    return f0


def cells_to_canvas_xy(cells, grid_r, canvas):
    """Cell (row, col) -> canvas pixel (x, y) at the cell centre. For plots."""
    cell_px = canvas / float(grid_r)
    cells = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
    return np.stack([(cells[:, 1] + 0.5) * cell_px,
                     (cells[:, 0] + 0.5) * cell_px], axis=1)
