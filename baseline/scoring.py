"""Bộ chấm CHUNG cho mọi baseline — cùng thước đo với ALPHA / BETA và bảng docs/SCORE.md.

Mỗi baseline chỉ dump dự đoán (box xyxy pixel ẢNH GỐC + score, mỗi ảnh CE-130 một mục, khoá = iid);
chấm bằng đúng `ce_localization.engine.evaluate.score` (AP của `metrics_np`, `oracle_recall`,
`score_AUC`, trần khi score = IoU, recall theo độ dày / cỡ, chỉ số điểm), không dùng COCOEvaluator.

Quy đổi sang bản ghi y khuôn `predict()` của ce_localization:
  - GT lấy từ `scan_ce130` (không từ json COCO), kẹp vào ảnh như `scale_boxes`;
  - box / GT chia (W, H) của ảnh gốc rồi sang cxcywh. ALPHA chia theo vùng ảnh thật trên canvas
    letterbox — phép co giãn theo trục nên IoU, tâm-trong-box không đổi;
  - `gt_size_px` = cạnh √(w·h) quy về canvas 512 = 512 / max(W, H) · √(w·h) (đúng quy đổi của ALPHA).
Ngân sách box: mặc định 200 box điểm cao nhất mỗi ảnh (N = 200 của SCORE.md); `None` = mọi box dump.
"""

import datetime
import json
import os

import numpy as np
from PIL import Image

from ce_localization.data.dataset import scale_boxes, scan_ce130
from ce_localization.engine.evaluate import DENSITY_BINS, SIZE_BINS, score
from ce_localization.utils.box_ops_np import xyxy_to_cxcywh

__all__ = ["BUDGET", "ORDERS", "iid_of_path", "load_gt", "top_budget", "records_from_pred", "score_pred",
           "write_dump", "read_dump", "score_row", "print_summary", "budget_tag"]

BUDGET = 200
ORDERS = ("topk_first", "nms_first")


def iid_of_path(path):
    """`.../<split>/<iid>_b<k>/ground_truth.jpg` -> iid, cùng quy tắc với `scan_ce130`."""
    return os.path.basename(os.path.dirname(path)).split("_b")[0]


def load_gt(root, split, limit=None):
    """-> {iid: {"wh": (W, H), "gt": xyxy pixel ảnh gốc đã kẹp, "text", "img_path"}}.
    Chỉ đọc header ảnh để lấy (W, H)."""
    items = scan_ce130(root, split)
    if limit:
        items = items[:limit]
    table = {}
    for it in items:
        with Image.open(it["img_path"]) as im:
            w, h = im.size
        table[it["image_id"]] = {"wh": (w, h), "gt": scale_boxes(it["boxes_xyxy_px"], 1.0, w, h),
                                 "text": it["text"], "img_path": it["img_path"]}
    return table


def top_budget(boxes, scores, budget):
    """Giữ `budget` box điểm cao nhất (thứ tự ổn định). budget None -> giữ hết."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError(f"số box {len(boxes)} != số score {len(scores)}")
    if budget is None or len(scores) <= budget:
        return boxes, scores
    keep = np.argsort(-scores, kind="stable")[:budget]
    return boxes[keep], scores[keep]


def records_from_pred(pred, gt_table, budget=BUDGET):
    """pred {iid: {"boxes_xyxy", "scores"}} -> bản ghi cho `score()`. Thiếu / thừa ảnh -> lỗi
    (thiếu ảnh mà im lặng thì recall sai mà không ai biết)."""
    missing, extra = sorted(set(gt_table) - set(pred)), sorted(set(pred) - set(gt_table))
    if missing or extra:
        raise KeyError(f"dump lệch GT: thiếu {len(missing)} ảnh {missing[:5]}, thừa {len(extra)} ảnh {extra[:5]}")
    records = []
    for iid in sorted(gt_table):
        g = gt_table[iid]
        w, h = g["wh"]
        whwh = np.array([w, h, w, h], dtype=np.float64)
        boxes, scores = top_budget(pred[iid]["boxes_xyxy"], pred[iid]["scores"], budget)
        gt = np.asarray(g["gt"], dtype=np.float64).reshape(-1, 4)
        gt_size = 512.0 / max(w, h) * np.sqrt(np.clip(gt[:, 2] - gt[:, 0], 0, None)
                                               * np.clip(gt[:, 3] - gt[:, 1], 0, None))
        records.append({"image_id": iid, "boxes": xyxy_to_cxcywh(boxes / whwh), "scores": scores,
                        "keep": np.zeros(0, dtype=int), "gt": xyxy_to_cxcywh(gt / whwh), "gt_size_px": gt_size})
    return records


def score_pred(pred, gt_table, budget=BUDGET, top_k=100, nms_thr=0.5, oracle=True):
    """-> {"topk_first": res, "nms_first": res} (res = dict của `score()`, ghi JSON được)."""
    records = records_from_pred(pred, gt_table, budget)
    out = {}
    for order in ORDERS:
        res = score(records, [], top_k, nms_thr, oracle=oracle, order=order)
        res["budget"] = budget
        res["boxes_per_image"] = float(np.mean([len(r["scores"]) for r in records])) if records else 0.0
        out[order] = res
    return out


def budget_tag(budget):
    return "Ball" if budget is None else f"B{budget}"


def write_dump(path, meta, pred, ndigits=2):
    """Dump dự đoán: box làm tròn 0,01 px, score 6 chữ số — đủ cho IoU / xếp hạng, file nhỏ."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    slim = {iid: {"boxes_xyxy": np.round(np.asarray(p["boxes_xyxy"], dtype=np.float64).reshape(-1, 4),
                                         ndigits).tolist(),
                  "scores": np.round(np.asarray(p["scores"], dtype=np.float64), 6).tolist()}
            for iid, p in pred.items()}
    with open(path, "w") as f:
        json.dump({"meta": meta, "pred": slim}, f)


def read_dump(path):
    with open(path) as f:
        d = json.load(f)
    return d["meta"], d["pred"]


def _num(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}".replace(".", ",")


def score_row(meta, res):
    """Một hàng markdown cho docs/SCORE.md (đúng thứ tự cột của bảng đó). `res` = kết quả `score_pred`."""
    m, t = res["nms_first"], res["topk_first"]
    dens = [m["density_recall"][name]["oracle_recall"] for _, _, name in DENSITY_BINS]
    size = [m["size_recall"][name]["oracle_recall"] for _, _, name in SIZE_BINS]
    steps = meta.get("steps", "—")
    if meta.get("budget", BUDGET) is None and isinstance(steps, int) and steps > 1:
        steps = f"{steps} (gộp {m['boxes_per_image']:.0f} box)"
    cells = [meta.get("run", "?"), f"`{meta.get('config', '?')}`", "—", str(steps),
             _num(m["AP50"]), _num(m["AP75"]), _num(m["AP_coco"]), _num(m["recall"]), _num(m["oracle_recall"]),
             _num(m["mean_bestIoU"]), _num(m["score_AUC"]), _num(m["oracle_score"]["AP50"]) if "oracle_score" in m else "—",
             *[_num(v) for v in dens], *[_num(v) for v in size], _num(m["kept_per_image"], 1), _num(t["AP50"]),
             str(meta.get("iter", "—")), str(meta.get("batch", "—")), meta.get("where", "—"),
             meta.get("train_time", "—"), meta.get("date", datetime.date.today().isoformat())]
    return "| " + " | ".join(cells) + " |"


def print_summary(tag, res, log=print):
    for order in ORDERS:
        r = res[order]
        o = r.get("oracle_score", {})
        log(f"  [{tag} | {order}] AP50 {r['AP50']:.4f} | AP75 {r['AP75']:.4f} | AP_coco {r['AP_coco']:.4f} | "
            f"recall {r['recall']:.4f} | oracle_recall {r['oracle_recall']:.4f} | score_AUC {r['score_AUC']:.4f} | "
            f"trần AP50 {o.get('AP50', float('nan')):.4f} | box giữ/ảnh {r['kept_per_image']:.1f} | "
            f"AP_pt {r['AP_pt']:.4f}")
    r = res["nms_first"]
    log("    oracle_recall theo độ dày: " + " | ".join(
        f"{name} {v['oracle_recall']:.3f}" for name, v in r["density_recall"].items())
        + " ; theo cỡ: " + " | ".join(f"{name} {v['oracle_recall']:.3f}" for name, v in r["size_recall"].items()))
