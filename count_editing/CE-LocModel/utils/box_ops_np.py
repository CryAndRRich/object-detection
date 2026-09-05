"""Bounding-box math — pure numpy, NO torch dependency.

This file is the SOURCE OF TRUTH for every coordinate transform. The torch port
(`box_ops.py`) must be mechanical, with `allclose(torch_fn, np_fn)` tests.

THE ONE CANONICAL SYSTEM: cxcywh in [0, 1] on a square canvas (default 512).
- `w > 0` always holds, and reads directly as "what fraction of the image wide".
- `snr_scale` exists ONLY inside the diffusion part; it never leaks outside.

The 6-step chain (see docs/thiet-ke-ce-loc-vong-2.md §4.7):

    all_bboxes: xyxy absolute pixels on the original image (W x H)
      (1) xyxy -> cxcywh
      (2) x scale = min(target/W, target/H)      # pad at the BOTTOM, no shift
      (3) / target -> [0, 1]
    ===> cxcywh [0,1]  (CANONICAL)
      (4) (x*2 - 1) * snr_scale                  # diffusion only
    ===> x_start, used by q_sample / DDIM
      (5) clamp(+-snr) -> /snr -> (x+1)/2        # back to [0,1]
      (6) cxcywh -> xyxy                         # only to compute GIoU

ROUND-1 BUG (do not repeat): decoding an extent is `(norm+1)/2`, NOT `(norm+1)/4`.
Dividing twice leaves boxes at half width; IoU between right and wrong is 0.25.
"""

import numpy as np

__all__ = [
    "compute_scale",
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "scale_to_canvas",
    "canvas_to_pixel",
    "encode_diffusion",
    "decode_diffusion",
    "box_iou",
    "generalized_box_iou",
    "pairwise_l1",
    "flip_horizontal",
    "filter_degenerate",
]


# --------------------------------------------------------------------------
# Steps (1) and (6): change format, NOT scale
# --------------------------------------------------------------------------

def xyxy_to_cxcywh(boxes):
    """[..., 4] (x1,y1,x2,y2) -> (cx,cy,w,h). Units unchanged."""
    b = np.asarray(boxes, dtype=np.float64)
    x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1)


def cxcywh_to_xyxy(boxes):
    """[..., 4] (cx,cy,w,h) -> (x1,y1,x2,y2). Units unchanged."""
    b = np.asarray(boxes, dtype=np.float64)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=-1)


# --------------------------------------------------------------------------
# Steps (2)+(3): original-image pixels -> canonical [0,1]
# --------------------------------------------------------------------------

def compute_scale(W, H, target=512):
    """Aspect-preserving resize factor. Every CE-130 image is exactly 384px tall,
    so W >= H always, meaning new_w == target and padding is ALWAYS at the bottom
    (see data/README.md §8)."""
    return min(target / float(W), target / float(H))


def scale_to_canvas(boxes_xyxy_px, W, H, target=512):
    """xyxy pixels on the original (W,H) -> cxcywh [0,1] on a target x target canvas.

    Returns (boxes_cxcywh_norm, scale, valid_h) where `valid_h = new_h / target`
    is the boundary of the real image region (everything below it is padding).
    """
    s = compute_scale(W, H, target)
    b = np.asarray(boxes_xyxy_px, dtype=np.float64).reshape(-1, 4) * s
    n = xyxy_to_cxcywh(b) / float(target)
    valid_h = int(H * s) / float(target)
    return n, s, valid_h


def canvas_to_pixel(boxes_cxcywh_norm, W, H, target=512):
    """Inverse of `scale_to_canvas`: cxcywh [0,1] -> xyxy pixels on the original."""
    s = compute_scale(W, H, target)
    b = np.asarray(boxes_cxcywh_norm, dtype=np.float64).reshape(-1, 4) * float(target)
    return cxcywh_to_xyxy(b) / s


# --------------------------------------------------------------------------
# Steps (4) and (5): canonical <-> diffusion space
# --------------------------------------------------------------------------

def encode_diffusion(boxes_norm, snr_scale=2.0):
    """cxcywh [0,1] -> x_start for diffusion: (x*2 - 1) * snr_scale.

    Applied to ALL 4 dimensions, exactly like DiffusionDet (`detector.py:396`).
    Small objects therefore give NEGATIVE w/h (w=0.0686 -> -1.73); that is
    NORMAL, not a bug.
    """
    b = np.asarray(boxes_norm, dtype=np.float64)
    return (b * 2.0 - 1.0) * float(snr_scale)


def decode_diffusion(x, snr_scale=2.0):
    """x_start -> cxcywh [0,1]. Clamp BEFORE dividing (same as DiffusionDet).

    ROUND-1 BUG: dividing by 2 one extra time here halved every box.
    Correct is `(x/snr + 1) / 2`, not `(x/snr + 1) / 4`.
    """
    s = float(snr_scale)
    v = np.clip(np.asarray(x, dtype=np.float64), -s, s)
    return (v / s + 1.0) / 2.0


# --------------------------------------------------------------------------
# IoU / GIoU — take xyxy, shared by matcher, loss and tests alike
# --------------------------------------------------------------------------

def _area_xyxy(b):
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def box_iou(boxes1, boxes2):
    """Pairwise IoU. boxes1 [N,4], boxes2 [M,4] xyxy -> (iou [N,M], union [N,M])."""
    b1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    a1, a2 = _area_xyxy(b1)[:, None], _area_xyxy(b2)[None, :]

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = a1 + a2 - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """Pairwise GIoU, xyxy. Range [-1, 1].

    GIoU has a gradient even when two boxes are DISJOINT (unlike plain IoU = 0),
    which matters on CE-130 because objects are tiny (median 0.41 % of image
    area), so most prediction-GT pairs are disjoint early in training.

    Requires valid boxes (x2>=x1, y2>=y1). Inverted boxes give nonsense — round 1
    once produced GIoU 5e5 from a bad decode. Run `filter_degenerate` first.
    """
    b1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    assert (b1[:, 2:] >= b1[:, :2]).all(), "boxes1 has an inverted box (x2<x1 or y2<y1)"
    assert (b2[:, 2:] >= b2[:, :2]).all(), "boxes2 has an inverted box (x2<x1 or y2<y1)"

    iou, union = box_iou(b1, b2)

    lt = np.minimum(b1[:, None, :2], b2[None, :, :2])
    rb = np.maximum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    enclosing = wh[..., 0] * wh[..., 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(enclosing > 0, iou - (enclosing - union) / enclosing, iou)


def pairwise_l1(boxes1, boxes2):
    """Pairwise L1 distance over the 4 coordinates. [N,4] x [M,4] -> [N,M]."""
    b1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    return np.abs(b1[:, None, :] - b2[None, :, :]).sum(-1)


# --------------------------------------------------------------------------
# Augmentation + data hygiene
# --------------------------------------------------------------------------

def flip_horizontal(boxes_cxcywh_norm):
    """Horizontal flip in canonical [0,1]: cx -> 1 - cx. w/h/cy unchanged.

    Horizontal ONLY. No rotation: rotating 5-30 degrees inflates axis-aligned
    boxes by 1.20x-1.98x, far too much for CE-130 objects (median 0.069 x 0.061),
    and rotating 90 degrees breaks the "padding is always at the bottom"
    assumption (only 9.6 % of images are square).
    """
    b = np.asarray(boxes_cxcywh_norm, dtype=np.float64).copy()
    b[..., 0] = 1.0 - b[..., 0]
    return b


def filter_degenerate(boxes_xyxy, min_size=0.0):
    """Drop boxes with w or h <= min_size. Returns (clean_boxes, keep_mask).

    Measured 14 of 37,110 CE-130 train boxes are degenerate. Must be filtered
    BEFORE the matcher AND before computing GIoU.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    keep = (b[:, 2] - b[:, 0] > min_size) & (b[:, 3] - b[:, 1] > min_size)
    return b[keep], keep
