"""Bounding-box math — pure numpy, NO torch dependency.

COPIED (not imported) from count_editing/CE-LocModel/utils/box_ops_np.py, because
sub-projects under object-detection/ deliberately do not import from each other
(see object-detection/README.md: "Bon project khong phu thuoc lan nhau").

Only the five functions Diffu2Seg actually needs are kept. Deliberately absent:
  - encode_diffusion / decode_diffusion : training-free, no diffusion on boxes
  - generalized_box_iou                 : no loss, so no GIoU
  - flip_horizontal                     : no augmentation at inference

THE ONE CANONICAL SYSTEM (identical to CE-LocModel, so numbers stay comparable):
cxcywh in [0, 1] on a square canvas (default 512).

CE-130 specifics that make the padding rule unambiguous:
  - every image is exactly 384px tall, width 384..1918 (measured on disk)
  - therefore W >= H always -> new_w == target -> padding is ALWAYS at the BOTTOM
  - padding covers ~29 % of the canvas at the median aspect ratio (1.41)

`scale_to_canvas` returns `valid_h`, the boundary of the real image region. That
number is what keeps prompts (and later, mask components) out of the padding.
"""

import numpy as np

__all__ = [
    "compute_scale",
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "xywh_to_xyxy",
    "scale_to_canvas",
    "box_iou",
]


def compute_scale(W, H, target=512):
    """Aspect-preserving resize factor."""
    return min(target / float(W), target / float(H))


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


def xywh_to_xyxy(boxes):
    """[..., 4] COCO (x,y,w,h) -> (x1,y1,x2,y2).

    NOT a duplicate of cxcywh_to_xyxy: COCO's (x,y) is the TOP-LEFT corner, not
    the centre. CE-130 ships both conventions and mixing them fails silently --
    the box stays inside the image and no assert fires:
      - data/ce130_coco/*.json   : COCO xywh   (top-left)  <- this function
      - all_phase2_V2/*/annotation.json `all_bboxes` : xyxy
      - samples/*/annotation/*.json `target_bbox`    : cxcywh
    """
    b = np.asarray(boxes, dtype=np.float64)
    x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([x, y, x + w, y + h], axis=-1)


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


def _area_xyxy(b):
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


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
