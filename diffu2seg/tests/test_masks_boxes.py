"""Masks and boxes: components, the outer-edge rule, padding, round-trip.

Run:  python -m pytest tests/test_masks_boxes.py -q
      python tests/test_masks_boxes.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d2s.boxes import dedup_boxes, filter_boxes, masks_to_boxes  # noqa: E402
from d2s.masks import (extract_masks, largest_component_containing_seed,  # noqa: E402
                       threshold_maps)
from utils.box_ops_np import (box_iou, cxcywh_to_xyxy, scale_to_canvas,  # noqa: E402
                              xywh_to_xyxy)

R = 64
CANVAS = 512
CELL = CANVAS / R          # 8.0 px


# ----------------------------------------------------------------- masks

def test_connected_components_keeps_only_seed_component():
    """Two sheep of the same class: attention links them, geometry separates them."""
    m = np.zeros((R, R), dtype=bool)
    m[10:14, 10:14] = True          # object A, holds the seed
    m[40:44, 40:44] = True          # object B, identical-looking

    comp = largest_component_containing_seed(m, (11, 11))
    assert comp[10:14, 10:14].all()
    assert not comp[40:44, 40:44].any()
    assert comp.sum() == 16


def test_component_is_empty_when_seed_below_threshold():
    """A prompt whose own cell did not survive must yield nothing.

    NEGATIVE CONTROL against "just take the biggest component instead", which
    would invent a mask for a prompt that said nothing.
    """
    m = np.zeros((R, R), dtype=bool)
    m[40:44, 40:44] = True
    assert not largest_component_containing_seed(m, (11, 11)).any()


def test_threshold_is_per_map():
    """Each map is cut at ITS OWN scale, so absolute magnitude cannot leak across.

    Two comparably strong maps of different absolute size must both yield their
    own 100 cells. rel_floor is disabled here to isolate the quantile.
    """
    f = np.zeros((2, R * R))
    f[0, :100] = 1.0
    f[1, :100] = 0.5
    binm = threshold_maps(f, R, quantile=0.90, rel_floor=0.0)
    assert binm[0].sum() == binm[1].sum() == 100


def test_rel_floor_kills_maps_with_no_signal():
    """The fix for the 2-objects-became-58-boxes bug.

    A quantile is RELATIVE, so on its own it hands back ~10 % of the cells even
    from a map that propagated nowhere. Measured on a synthetic graph: a prompt
    seeded on background reached max 0.0000 while its own q90 was 1.5e-05, so
    `f > q90` still returned its own cell and that became a tidy 1x1 box.

    rel_floor adds the absolute question -- is anything here at least 5 % of the
    strongest response IN THIS IMAGE -- and a dead map fails it.

    NEGATIVE CONTROL: with rel_floor=0 the dead map DOES produce cells, which is
    what makes this test proof that the floor is doing the work.
    """
    f = np.zeros((2, R * R))
    f[0, :100] = 1.0                       # a real object
    f[1, :100] = 1e-6                      # background: 1e-6 of the strongest

    with_floor = threshold_maps(f, R, quantile=0.90, rel_floor=0.05)
    assert with_floor[0].sum() == 100, "the real map must survive"
    assert with_floor[1].sum() == 0, "the dead map must be suppressed"

    without = threshold_maps(f, R, quantile=0.90, rel_floor=0.0)
    assert without[1].sum() > 0, "rel_floor is no longer being tested"


def test_extract_masks_drops_dead_prompts():
    f = np.zeros((2, R * R))
    f[0].reshape(R, R)[10:14, 10:14] = 1.0
    # prompt 1 sits at (11,11) but its map is flat -> nothing above quantile
    cells = np.array([[11, 11], [11, 11]])
    masks, kept = extract_masks(f, cells, R, quantile=0.99)
    assert len(masks) == 1 and len(kept) == 1


# ----------------------------------------------------------------- boxes

def test_mask_to_box_is_outer_edge_exact():
    """Cells (2..5, 3..7) -> x [24, 64), y [16, 48) in canvas px.

    Catches the missing +1: without it x2 would be 56 and y2 40, shrinking every
    box by one 8 px cell (~20 % of a median CE-130 object).
    """
    m = np.zeros((1, R, R), dtype=bool)
    m[0, 2:6, 3:8] = True
    box = masks_to_boxes(m, R, CANVAS)[0]

    x1, x2, y1, y2 = 3 * CELL, 8 * CELL, 2 * CELL, 6 * CELL
    expected = np.array([((x1 + x2) / 2) / CANVAS, ((y1 + y2) / 2) / CANVAS,
                         (x2 - x1) / CANVAS, (y2 - y1) / CANVAS])
    assert np.allclose(box, expected)
    assert np.allclose(box[2] * CANVAS, 40.0)      # 5 cells wide
    assert np.allclose(box[3] * CANVAS, 32.0)      # 4 cells tall


def test_single_cell_mask_gives_one_cell_box():
    m = np.zeros((1, R, R), dtype=bool)
    m[0, 7, 9] = True
    box = masks_to_boxes(m, R, CANVAS)[0]
    assert np.allclose(box[2], CELL / CANVAS)
    assert np.allclose(box[3], CELL / CANVAS)


def test_empty_mask_gives_zero_row():
    box = masks_to_boxes(np.zeros((1, R, R), dtype=bool), R, CANVAS)[0]
    assert np.allclose(box, 0.0)


def test_filter_rejects_oversized_and_degenerate():
    boxes = np.array([
        [0.5, 0.4, 0.10, 0.10],     # keep
        [0.5, 0.4, 0.90, 0.90],     # too large -> padding escape tripwire
        [0.5, 0.4, 0.001, 0.10],    # degenerate
    ])
    kept, info = filter_boxes(boxes, R, CANVAS, valid_h=1.0)
    assert len(kept) == 1
    assert info["n_too_large"] == 1 and info["n_degenerate"] == 1


def test_padding_filter_uses_area_not_centre():
    """A real object low in the frame must survive; a padding box must not.

    valid_h = 0.94 (a 408x384 image). The first box has its CENTRE below
    valid_h but most of its area inside -- testing the centre alone would
    discard a genuine detection.
    """
    valid_h = 0.9412
    mostly_inside = [0.5, valid_h - 0.005, 0.06, 0.05]
    in_padding = [0.5, valid_h + 0.03, 0.06, 0.05]

    kept, info = filter_boxes(np.array([mostly_inside, in_padding]), R, CANVAS,
                              valid_h=valid_h)
    assert info["n_in_padding"] == 1
    assert len(kept) == 1
    assert kept[0][1] < valid_h


def test_right_hand_padding_is_filtered_for_portrait_images():
    """Same rule, applied to the WIDTH -- the COCO portrait case.

    valid_w = 0.9152 (a 586x640 image). CE-130 can never produce this, so
    without a test the width term would be dead code that looks correct.
    """
    valid_w = 0.9152
    mostly_inside = [valid_w - 0.005, 0.5, 0.06, 0.05]
    in_padding = [valid_w + 0.03, 0.5, 0.06, 0.05]

    kept, info = filter_boxes(np.array([mostly_inside, in_padding]), R, CANVAS,
                              valid_w=valid_w)
    assert info["n_in_padding"] == 1
    assert len(kept) == 1
    assert kept[0][0] < valid_w


def test_valid_w_default_leaves_ce130_filtering_unchanged():
    """[NEGATIVE CONTROL] valid_w=1.0 must reproduce the CE-130 behaviour byte
    for byte. If it does not, adding COCO support changed CE-130 results."""
    valid_h = 0.9412
    boxes = np.array([[0.5, valid_h - 0.005, 0.06, 0.05],
                      [0.5, valid_h + 0.03, 0.06, 0.05],
                      [0.2, 0.3, 0.10, 0.10]])
    a, ia = filter_boxes(boxes, R, CANVAS, valid_h=valid_h)
    b, ib = filter_boxes(boxes, R, CANVAS, valid_h=valid_h, valid_w=1.0)
    assert np.array_equal(a, b) and ia == ib


def test_dedup_removes_duplicates_keeps_neighbours():
    """441 prompts on ~21 objects: duplicates must go, true neighbours must not."""
    dup = [[0.30, 0.30, 0.10, 0.10]] * 5
    neighbour = [[0.42, 0.30, 0.10, 0.10]]         # adjacent object, low IoU
    kept, _ = dedup_boxes(np.array(dup + neighbour), iou_thr=0.70)
    assert len(kept) == 2


def test_dedup_orders_by_area_not_input_order():
    boxes = np.array([[0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.20, 0.20]])
    kept, idx = dedup_boxes(boxes, iou_thr=0.70)
    assert idx[0] == 1, "largest box must be considered first"


# ------------------------------------------------- coordinates / round-trip

def test_xywh_to_xyxy_from_coco():
    """COCO (x,y) is TOP-LEFT, not a centre -- the third CE-130 box convention."""
    assert np.allclose(xywh_to_xyxy([[74.0, 167.0, 25.0, 20.0]])[0],
                       [74.0, 167.0, 99.0, 187.0])


def _roundtrip_iou(W, H, gt_px):
    """GT pixels -> canvas -> cell mask -> box, returning IoU against the GT."""
    gt_norm, _, valid_h = scale_to_canvas(gt_px, W, H, CANVAS)
    assert abs(valid_h - int(H * min(CANVAS / W, CANVAS / H)) / CANVAS) < 1e-12

    xyxy = cxcywh_to_xyxy(gt_norm)[0] * CANVAS
    c0, c1 = int(xyxy[0] // CELL), int(np.ceil(xyxy[2] / CELL))
    r0, r1 = int(xyxy[1] // CELL), int(np.ceil(xyxy[3] / CELL))

    m = np.zeros((1, R, R), dtype=bool)
    m[0, r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = True
    back = masks_to_boxes(m, R, CANVAS)
    return box_iou(cxcywh_to_xyxy(back), cxcywh_to_xyxy(gt_norm))[0][0, 0]


def test_pad_coordinate_roundtrip():
    """Round-trip on the shapes that carry 96 % of val.

    Aspect ratios up to ~1.8 cover 96.5 % of val images (measured: 29.2 % below
    1.2, 26.4 % in 1.2-1.5, 40.9 % in 1.5-2.0). There the median 39x32 px object
    still spans 4-6 cells and survives the 8 px grid.
    """
    for W, H in [(384, 384), (541, 384), (700, 384)]:
        gt_px = np.array([[100.0, 120.0, 139.0, 152.0]])     # 39x32, the median
        iou = _roundtrip_iou(W, H, gt_px)
        assert iou > 0.6, f"round-trip IoU {iou:.3f} too low for {W}x{H}"


def test_extreme_aspect_loses_resolution_by_design():
    """THE COST OF r=64, as a number rather than a caveat.

    A 1918x384 image (aspect 4.99, the measured CE-130 maximum) is scaled by
    0.267 to fit a square canvas, so the median 39x32 px object shrinks to
    1.30 x 1.07 cells and quantisation eats most of it: round-trip IoU ~0.35
    even with a PERFECT mask.

    This is a property of the resolution choice, not a bug, and it bounds what
    any p can achieve on those images. It is rare -- only 3.5 % of val images
    have aspect > 2.0, and 0.2 % exceed 3.0 -- but run_stage1.py reports the
    per-image ceiling so a low oracle_recall is never misread as a failure of
    the mechanism. If gate 1 comes back GREY, r=96 is the next variable.
    """
    iou = _roundtrip_iou(1918, 384, np.array([[100.0, 120.0, 139.0, 152.0]]))
    assert iou < 0.5, "extreme aspect should visibly lose resolution"
    assert iou > 0.2, "...but not vanish entirely"


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
