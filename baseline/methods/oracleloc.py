"""OracleLoc (bài Add One, Take One, mục 4.1 "Localization ability comparison" + phụ lục A.4.2) — MỐC ORACLE, không model, không weight.

Bài: "privileged access to the mean feature vector µ computed from the ground-truth annotated objects … exhaustive search over image
locations to identify the box center whose feature vector best matches µ, while the box width and height are directly determined by µ".
Đặc trưng = đúng 4-D của C-NLL (`ce_localization/engine/add_eval.cnll_features`: [w / nw, h / nh, IoU lớn nhất với vật, TB khoảng cách tới tâm
3 vật gần nhất / √(nw·nh)]) trên CANVAS 512 như lúc chấm; μ (+ Σ) fit trên vật cùng lớp (CoCount: quan hệ tính cả vật lớp kia) bằng
chính `cnll_fit` của code chấm.

- Cỡ box = μ (w, h TB của mọi vật cùng lớp trong ảnh); quét tâm trên lưới bước `predict.stride` px canvas, box nằm trọn trong ảnh.
- "best matches µ" bài không nói theo khoảng cách nào (`predict.criterion`, chạy cả hai rồi chốt — người dùng 2026-11-07):
  `euclid` = ‖z − μ‖₂ trên đặc trưng đã chuẩn hoá như code chấm (bám câu chữ: chỉ dùng μ); `cnll` = ½ (z − μ)ᵀ Σ⁻¹ (z − μ) = tối ưu
  thẳng metric chấm (cận dưới thật của C-NLL với cỡ cố định).
- K box = K vị trí điểm thấp nhất sau NMS IoU `predict.nms_iou` (box[0] = vị trí tốt nhất); thiếu chỗ không trùng thì lặp lại theo thứ tự.
- Ảnh < `CNLL_MIN_OBJ` vật cùng lớp (không chấm C-NLL): box cỡ TB vật (hoặc 1/4 cạnh ảnh nếu không có vật) ở giữa ảnh, lặp K lần.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ce_localization.utils.log import Progress
from ce_localization.data.turns import letterbox_size
from ce_localization.engine.add_eval import CNLL_MIN_OBJ, cnll_fit, cnll_features, other_class, select_nms

__all__ = ["CRITERIA", "grid_boxes", "scores", "search", "predict"]

CRITERIA = ("euclid", "cnll")
CHUNK = 4_000_000                 # số cặp (vị trí × vật tham chiếu) mỗi lượt tính đặc trưng (IoU + kNN ~ vài chục MB)


def grid_boxes(wh, bw, bh, stride):
    """Mọi box cỡ (bw, bh) nằm trọn trong vùng ảnh (nw, nh), góc trên trái trên lưới `stride` px (+ sát mép phải / dưới). -> [N, 4]."""
    nw, nh = wh
    bw, bh = min(bw, nw), min(bh, nh)
    xs = np.unique(np.append(np.arange(0.0, nw - bw + 1e-9, stride), nw - bw))
    ys = np.unique(np.append(np.arange(0.0, nh - bh + 1e-9, stride), nh - bh))
    x, y = np.meshgrid(xs, ys)
    x, y = x.ravel(), y.ravel()
    return np.stack([x, y, x + bw, y + bh], 1)


def scores(feats, mu, inv, criterion):
    d = feats - mu
    if criterion == "euclid":
        return np.linalg.norm(d, axis=1)
    if criterion == "cnll":
        return 0.5 * np.einsum("ni,ij,nj->n", d, inv, d)
    raise ValueError(f"criterion {criterion!r} không thuộc {CRITERIA}")


def search(objects, wh, k, stride, criteria=CRITERIA, others=None, nms_iou=0.5):
    """Hình học canvas (vật cùng lớp `objects` [M, 4], vùng ảnh `wh`, vật lớp kia `others`) -> {tiêu chí: box [k, 4] canvas}."""
    objects = np.asarray(objects, dtype=np.float64).reshape(-1, 4)
    nw, nh = wh
    if len(objects) < CNLL_MIN_OBJ:
        bw, bh = ((objects[:, 2] - objects[:, 0]).mean(), (objects[:, 3] - objects[:, 1]).mean()) if len(objects) else (nw / 4, nh / 4)
        bw, bh = min(bw, nw), min(bh, nh)
        b = np.array([[(nw - bw) / 2, (nh - bh) / 2, (nw + bw) / 2, (nh + bh) / 2]])
        return {c: np.repeat(b, k, 0) for c in criteria}
    ref = objects if others is None or not len(others) else np.concatenate([objects, others])
    mu, inv = cnll_fit(objects, ref, wh)
    cand = grid_boxes(wh, mu[0] * nw, mu[1] * nh, stride)
    step = max(256, CHUNK // len(ref))
    feats = np.concatenate([cnll_features(cand[i:i + step], ref, wh) for i in range(0, len(cand), step)])
    out = {}
    for c in criteria:
        s = scores(feats, mu, inv, c)
        keep = select_nms(cand, s, k=k, iou=nms_iou)
        if len(keep) < k:                                            # ảnh nhỏ / box to: không đủ chỗ không trùng
            rest = [int(i) for i in np.argsort(s, kind="stable") if int(i) not in set(keep)]
            keep = (keep + rest)[:k] if len(keep) + len(rest) >= k else (keep * k)[:k]
        out[c] = cand[keep]
    return out


def _job(args):
    return search(*args)


def predict(samples, geo_fn, k, p, log=print, log_every=500):
    """-> {tiêu chí: list {"image_id", "boxes" [K, 4] pixel ẢNH GỐC (box[0] = vị trí tốt nhất)}}. `geo_fn(sample)` = hình học canvas như
    lúc chấm (`baseline.data.geometry`). `p` = khối `predict` của config (stride, criteria, nms_iou, workers = số tiến trình CPU, mặc
    định min(16, số CPU); kết quả không phụ thuộc số tiến trình)."""
    crit = tuple(p.get("criteria", CRITERIA))
    for c in crit:
        if c not in CRITERIA:
            raise ValueError(f"criterion {c!r} không thuộc {CRITERIA}")
    jobs = []
    for s in samples:
        g = geo_fn(s)
        others = other_class(g["objects"], g["objects_all"]) if "objects_all" in g else None
        jobs.append((g["objects"], tuple(float(v) for v in g["wh"]), k, p["stride"], crit, others, p.get("nms_iou", 0.5)))
    workers = int(p.get("workers") or min(16, os.cpu_count() or 1))
    log(f"    [oracleloc] {len(jobs)} ảnh | lưới {p['stride']} px | {crit} | {workers} tiến trình")
    out, prog = {c: [] for c in crit}, Progress("oracleloc", len(samples), log_every, log)
    ex = ProcessPoolExecutor(workers) if workers > 1 else None
    it = ex.map(_job, jobs, chunksize=4) if ex else map(_job, jobs)
    for n, (s, res) in enumerate(zip(samples, it), 1):
        scale = letterbox_size(s["wh"], 512)[0]
        for c in crit:
            out[c].append({"image_id": s["image_id"], "boxes": (res[c] / scale).round(4).tolist()})
        prog.update(n)
    if ex:
        ex.shutdown()
    return out
