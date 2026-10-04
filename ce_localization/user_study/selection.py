"""Chọn K box hiện cho người chấm từ 30 mẫu của model — CHUNG mọi model, KHÔNG nhìn lỗ GT lẫn box vật (đo chính model:
lọc box đè vật sẽ che đúng điểm yếu GAMMA3 / GAMMA4 nhắm sửa).

  1. kẹp box vào ảnh; bỏ box cạnh < MIN_SIDE px sau khi kẹp (suy biến / nằm ngoài ảnh)
  2. phiếu bầu của mỗi box = số mẫu (kể cả nó) có IoU >= VOTE_IOU với nó — độ đồng thuận của chính model
  3. NMS tham lam theo phiếu (hoà: thứ tự mẫu): nhận box có IoU <= NMS_LEVELS[0] với mọi box đã nhận; chưa đủ K thì nới dần
     ngưỡng (mức 1,0 = nhận cả box trùng) ⇒ luôn đủ K nếu còn >= K box hợp lệ. Thứ tự nhận = hạng (box 1 = phiếu cao nhất).
  `n_distinct` = số box nhận được ở ngưỡng đầu khi không giới hạn K — độ đa dạng của 30 mẫu.
"""

import numpy as np

from ce_localization.utils.box_ops_np import box_iou

__all__ = ["K", "VOTE_IOU", "NMS_LEVELS", "MIN_SIDE", "clip_to_image", "select_boxes", "ioa"]

K = 4
VOTE_IOU = 0.5
NMS_LEVELS = (0.3, 0.5, 0.7, 1.0)
MIN_SIDE = 2.0


def clip_to_image(boxes, wh):
    """[N,4] xyxy -> kẹp vào [0, W] x [0, H]; box ngược (x2 < x1) thành bề rộng 0."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    W, H = wh
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
    b[:, 2] = np.maximum(b[:, 2], b[:, 0])
    b[:, 3] = np.maximum(b[:, 3], b[:, 1])
    return b


def _greedy(order, iou, thr, picked, limit):
    for o in order:
        if len(picked) >= limit:
            break
        if o not in picked and all(iou[o, p] <= thr for p in picked):
            picked.append(o)
    return picked


def select_boxes(boxes, wh, k=K, vote_iou=VOTE_IOU, nms_levels=NMS_LEVELS, min_side=MIN_SIDE):
    """boxes [N,4] xyxy (pixel ảnh), wh = (W, H) -> dict: `idx` (chỉ số mẫu gốc), `boxes` (đã kẹp), `vote`, `nms` (ngưỡng lúc
    nhận) — theo hạng; `n_valid`, `n_distinct`."""
    b = clip_to_image(boxes, wh)
    valid = np.flatnonzero((b[:, 2] - b[:, 0] >= min_side) & (b[:, 3] - b[:, 1] >= min_side))
    out = {"idx": [], "boxes": [], "vote": [], "nms": [], "n_valid": int(len(valid)), "n_distinct": 0}
    if not len(valid):
        return out
    bv = b[valid]
    iou = box_iou(bv, bv)[0]
    vote = (iou >= vote_iou).sum(1)
    order = np.argsort(-vote, kind="stable")
    out["n_distinct"] = len(_greedy(order, iou, nms_levels[0], [], len(order)))
    picked, level = [], []
    for thr in nms_levels:
        n0 = len(picked)
        _greedy(order, iou, thr, picked, k)
        level += [thr] * (len(picked) - n0)
        if len(picked) >= k:
            break
    out.update(idx=[int(valid[p]) for p in picked], boxes=[bv[p].round(2).tolist() for p in picked],
               vote=[int(vote[p]) for p in picked], nms=level)
    return out


def ioa(boxes, objects):
    """[K,4], [M,4] -> [K,M] giao / diện tích box (box diện tích 0 -> 0); như `on_object` của eval."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    o = np.asarray(objects, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(b[:, None, :2], o[None, :, :2])
    rb = np.minimum(b[:, None, 2:], o[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(area > 0, inter / area, 0.0)
