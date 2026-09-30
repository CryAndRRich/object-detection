"""Metric mMR / Recall / AP50 theo giao thức CrowdHuman — numpy thuần.

Tách khỏi ``crowdhuman_eval.py`` (bản gắn với detectron2) để phần toán có thể test được
độc lập, không cần cài detectron2. Xem self-test ở ``tests/test_mmr.py``.

mMR (log-average miss rate), theo Dollár et al. "Pedestrian Detection: An Evaluation of
the State of the Art" và toolkit CrowdHuman:

    mMR = exp( mean_i( log( MR(fppi_i) ) ) )

với 9 mốc FPPI cách đều theo log trong [1e-2, 1e0], MR = 1 - recall,
FPPI = số false positive / số ảnh. Tại mỗi mốc lấy MR ở điểm có FPPI <= mốc đó gần nhất.
Nhỏ hơn là tốt hơn.
"""

import numpy as np


def iou_matrix(dets, gts):
    """IoU giữa (N,4) và (M,4), định dạng xyxy."""
    if len(dets) == 0 or len(gts) == 0:
        return np.zeros((len(dets), len(gts)), dtype=np.float64)
    lt = np.maximum(dets[:, None, :2], gts[None, :, :2])
    rb = np.minimum(dets[:, None, 2:], gts[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_d = ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))[:, None]
    area_g = ((gts[:, 2] - gts[:, 0]) * (gts[:, 3] - gts[:, 1]))[None, :]
    return inter / np.maximum(area_d + area_g - inter, 1e-9)


def ioa_matrix(dets, ignores):
    """IoA = intersection / diện tích detection.

    Dùng cho vùng ignore: detection nằm gần hết trong vùng ignore thì bỏ qua, không tính
    false positive. Toolkit CrowdHuman cũng ghép ignore theo IoA chứ không phải IoU, vì
    vùng ignore thường lớn hơn detection nhiều nên IoU sẽ luôn nhỏ.
    """
    if len(dets) == 0 or len(ignores) == 0:
        return np.zeros((len(dets), len(ignores)), dtype=np.float64)
    lt = np.maximum(dets[:, None, :2], ignores[None, :, :2])
    rb = np.minimum(dets[:, None, 2:], ignores[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_d = ((dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1]))[:, None]
    return inter / np.maximum(area_d, 1e-9)


def compute_mmr_and_recall(per_image, iou_thr=0.5, ioa_thr=0.5):
    """Tính mMR, Recall, AP50.

    Args:
        per_image: dict ``{image_id: (dets, gts, ignores)}`` với
            ``dets`` (N,5) = xyxy + score, ``gts`` (M,4) xyxy person thật,
            ``ignores`` (K,4) xyxy vùng ignore. Ảnh không có detection vẫn phải có mặt
            trong dict, vì nó tính vào mẫu số của FPPI và recall.
    Returns:
        dict với ``mMR``, ``Recall``, ``AP50`` (đơn vị %).
    """
    n_images = len(per_image)
    n_gt = sum(len(g) for _, g, _ in per_image.values())
    if n_images == 0 or n_gt == 0:
        return {"mMR": float("nan"), "Recall": float("nan"), "AP50": float("nan")}

    rows = []   # (score, is_tp, is_fp) — gộp detection của mọi ảnh
    for dets, gts, igs in per_image.values():
        if len(dets) == 0:
            continue
        dets = dets[np.argsort(-dets[:, 4])]
        matched = np.zeros(len(gts), dtype=bool)
        ious = iou_matrix(dets[:, :4], gts)
        ioas = ioa_matrix(dets[:, :4], igs)
        for di in range(len(dets)):
            # ghép greedy theo score giảm dần với GT chưa bị chiếm, IoU cao nhất
            cand = np.where(~matched)[0]
            best, best_iou = -1, iou_thr
            for gi in cand:
                if ious[di, gi] >= best_iou:
                    best, best_iou = gi, ious[di, gi]
            if best >= 0:
                matched[best] = True
                rows.append((dets[di, 4], 1, 0))
            elif len(igs) and ioas[di].max() > ioa_thr:
                rows.append((dets[di, 4], 0, 0))   # trong vùng ignore -> không tính
            else:
                rows.append((dets[di, 4], 0, 1))   # false positive

    if not rows:
        return {"mMR": 100.0, "Recall": 0.0, "AP50": 0.0}

    rows.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in rows])
    fp = np.cumsum([r[2] for r in rows])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)
    fppi = fp / n_images

    ref = np.logspace(-2.0, 0.0, 9)
    mrs = []
    for r in ref:
        idx = np.where(fppi <= r)[0]
        mr = 1.0 - (recall[idx[-1]] if len(idx) else 0.0)
        mrs.append(max(mr, 1e-12))   # tránh log(0) khi MR = 0
    mmr = float(np.exp(np.mean(np.log(mrs))))

    # AP50 kiểu VOC all-point interpolation
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([0.0], precision))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))

    return {
        "mMR": mmr * 100.0,
        "Recall": float(recall[-1]) * 100.0,
        "AP50": ap * 100.0,
    }
