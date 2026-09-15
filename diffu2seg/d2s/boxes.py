"""Binary masks -> boxes in the project's canonical coordinate system.

Output is cxcywh in [0,1] on the 512 canvas -- the same system every CE-130
number in docs/03-ket-qua.md already lives in.

                        THE OFF-BY-ONE THAT MATTERS

A latent cell is not a point, it is an 8x8 canvas-pixel tile. Cell (i, j) covers
canvas x in [j*8, (j+1)*8) and y in [i*8, (i+1)*8), so a mask's box runs to the
OUTER edge of the last cell:

    x1 = c_min * 8            x2 = (c_max + 1) * 8

Dropping the +1 shortens every box by exactly one cell = 8 px. On CE-130 the
median object short side is 4.65 cells, so that is a ~20 % error on a typical
box -- easily enough to push a true detection under the IoU 0.5 threshold, and
it would look like a mechanism failure rather than an arithmetic one.

                    NO SCORES ANYWHERE IN THIS FILE

Training-free means there is nothing to rank by. Dedup therefore proceeds by
AREA, largest first, and the returned order carries no confidence information.
utils/metrics.py deliberately omits score_AUC for the same reason.
"""

import numpy as np

from utils.box_ops_np import box_iou, cxcywh_to_xyxy

__all__ = ["masks_to_boxes", "filter_boxes", "dedup_boxes"]

VAE_STRIDE = 8


def masks_to_boxes(masks, grid_r, canvas=512):
    """(M, r, r) bool -> (M, 4) cxcywh in [0,1]. Empty masks give a zero row."""
    masks = np.asarray(masks, dtype=bool).reshape(-1, grid_r, grid_r)
    cell_px = canvas / float(grid_r)          # 8.0 for r=64 on a 512 canvas

    out = np.zeros((len(masks), 4), dtype=np.float64)
    for m_idx, m in enumerate(masks):
        if not m.any():
            continue
        rows = np.where(m.any(axis=1))[0]
        cols = np.where(m.any(axis=0))[0]

        x1 = cols[0] * cell_px
        x2 = (cols[-1] + 1) * cell_px         # +1: outer edge, see docstring
        y1 = rows[0] * cell_px
        y2 = (rows[-1] + 1) * cell_px

        out[m_idx] = [((x1 + x2) / 2.0) / canvas,
                      ((y1 + y2) / 2.0) / canvas,
                      (x2 - x1) / canvas,
                      (y2 - y1) / canvas]
    return out


def filter_boxes(boxes, grid_r, canvas=512, min_box_cells=0.5,
                 max_area_frac=0.25, valid_h=1.0, valid_w=1.0):
    """Drop degenerate, oversized, and padding-region boxes.

    Returns (kept_boxes, info) where `info` counts each rejection reason.

    `max_area_frac` is a TRIPWIRE, not an accuracy filter. The median CE-130
    object covers 0.33 % of its image, so a box over 25 % of the canvas nearly
    always means a prompt escaped into the padding and propagated across it.
    The count is reported: a large one means the padding logic is broken, not
    that the images contain huge objects.

    Padding rejection uses the FRACTION OF BOX AREA inside the real region, not
    the box centre. A real object sitting near the bottom of the image can have
    its centre below valid_h while most of it is still inside -- testing the
    centre alone would throw those away.

    `valid_w` defaults to 1.0, which is exact for CE-130 (W >= H always, so
    padding is only ever at the bottom). COCO portrait images pad on the right,
    and there the width term is the one that matters.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes, {"n_in": 0, "n_kept": 0, "n_degenerate": 0,
                       "n_too_large": 0, "n_in_padding": 0}

    min_side = (min_box_cells * (canvas / float(grid_r))) / canvas
    xyxy = cxcywh_to_xyxy(boxes)

    degenerate = (boxes[:, 2] < min_side) | (boxes[:, 3] < min_side)
    too_large = (boxes[:, 2] * boxes[:, 3]) > max_area_frac

    x1, y1, x2, y2 = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3]
    height = np.clip(y2 - y1, 1e-12, None)
    width = np.clip(x2 - x1, 1e-12, None)
    inside_h = np.clip(np.minimum(y2, valid_h) - y1, 0.0, None) / height
    inside_w = np.clip(np.minimum(x2, valid_w) - x1, 0.0, None) / width
    # Area fraction inside the real region, as the product of the two 1-D
    # overlaps -- the box is a rectangle, so this is exact, not an approximation.
    in_padding = (inside_h * inside_w) < 0.5

    keep = ~(degenerate | too_large | in_padding)
    info = {
        "n_in": int(len(boxes)),
        "n_kept": int(keep.sum()),
        "n_degenerate": int(degenerate.sum()),
        "n_too_large": int(too_large.sum()),
        "n_in_padding": int(in_padding.sum()),
    }
    return boxes[keep], info


def dedup_boxes(boxes, iou_thr=0.70):
    """Score-free NMS: ~441 prompts land on ~21 objects, so duplicates dominate.

    Ordered by AREA (largest first) because there is no score to order by. The
    threshold is tighter than CE-LocModel's 0.5 on purpose -- CE-130 objects of
    one class genuinely crowd together, and a loose threshold merges neighbours
    into one box.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) <= 1:
        return boxes, np.arange(len(boxes))

    order = np.argsort(-(boxes[:, 2] * boxes[:, 3]))
    xyxy = cxcywh_to_xyxy(boxes)

    keep = []
    for idx in order:
        if not keep:
            keep.append(idx)
            continue
        iou = box_iou(xyxy[idx][None, :], xyxy[keep])[0][0]
        if iou.max() <= iou_thr:
            keep.append(idx)

    keep = np.array(keep, dtype=np.int64)
    return boxes[keep], keep
