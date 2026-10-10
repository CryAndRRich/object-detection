#!/usr/bin/env python3
"""Cửa G1 của EXPERIMENT BETA: TRẦN AP của box dựng từ TÂM + một nguồn cỡ — KHÔNG train, KHÔNG model.

Câu hỏi: giám sát chỉ bằng tâm thì cỡ box phải đến từ đâu đó; mỗi nguồn cỡ cho AP tối đa bao nhiêu?
Mỗi bộ box = (nguồn tâm) × (nguồn cỡ), chấm với box GT bằng đúng `score_records` của eval.py, score =
IoU thật (xếp hạng hoàn hảo, như cột "trần AP50" của docs/SCORE.md), mỗi tâm MỘT box, không NMS.
Box GT dùng để dựng cỡ ở các bộ `img_*` / `obj_*` / `gt_wh` — đó là TRẦN, không phải nhãn train.

Nguồn tâm:
  gt    : tâm box GT (density hoàn hảo)
  peaks : điểm density thật (`data/density_points.json`, = nhãn BETA0); cỡ lấy từ GT thì ghép
          đỉnh <-> GT một-một (`match_points_to_boxes`), đỉnh không ghép được dùng cỡ chung của ảnh
Nguồn cỡ (s = sqrt(w·h)):
  knn_sq     : vuông, cỡ kNN (công thức + tham số của `data.pseudo_size`, = box giả BETA0)
  knn_rel_sq : vuông, TỈ LỆ cỡ giữa các vật theo kNN, thang chung của ảnh lấy ĐÚNG từ GT
  img_sq     : vuông, MỘT cỡ cho cả ảnh = trung bình nhân s của GT trong ảnh
  img_wh     : MỘT (w, h) cho cả ảnh = trung bình nhân w, h của GT trong ảnh (tỉ lệ cạnh chung)
  obj_sq     : vuông, cỡ đúng từng vật (s của GT)
  gt_wh      : w, h đúng từng vật (với tâm gt = chính box GT, kiểm tra = 1; với peaks = chỉ còn lỗi tâm)

Đọc: img_sq cao -> chỉ cần học MỘT cỡ mỗi ảnh; obj_sq << gt_wh -> phải học tỉ lệ cạnh;
img_wh ≈ gt_wh -> tỉ lệ cạnh chung của ảnh là đủ; mọi bộ đều thấp -> "chỉ tâm" không đạt AP.

Box GT kẹp vào ảnh như eval (`scale_boxes`); box dựng không kẹp (như box giả BETA0).
~3.600 ảnh, chỉ đọc annotation + header ảnh + tính AP 10 ngưỡng cho 12 bộ × 3 split ⇒ ước 10–25 phút
(đĩa dùng chung), chạy NỀN:
  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  LOG=/mnt/disk1/aiotlab/haitn/log/detection/beta0/beta_g1_box_ceiling_$(date +%m%d_%H%M).log
  nohup python tools/point_box_ceiling.py --workers 8 \\
      --report /mnt/disk1/aiotlab/haitn/output/detection/beta0/beta_g1_box_ceiling.json > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.dataset import scale_boxes, scan_ce130  # noqa: E402
from ce_localization.data.points import (PointTable, match_points_to_boxes, pseudo_boxes,  # noqa: E402
                                         pseudo_sizes)
from ce_localization.engine.evaluate import DENSITY_BINS, score_records  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou, xyxy_to_cxcywh  # noqa: E402
from ce_localization.utils.box_ops_np import cxcywh_to_xyxy as c2x  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402

SPLITS = ("train", "val", "test")
CENTERS = ("gt", "peaks")
SIZES = ("knn_sq", "knn_rel_sq", "img_sq", "img_wh", "obj_sq", "gt_wh")
REPORT_KEYS = ("AP50", "AP75", "AP_coco", "recall", "precision", "oracle_recall", "mean_bestIoU", "median_bestIoU")


def _gmean(x):
    return float(np.exp(np.mean(np.log(np.maximum(x, 1e-6))))) if len(x) else float("nan")


def box_sets(gt, peaks, H, ps):
    """Một ảnh -> {(tâm, cỡ): box xyxy [K,4]}. gt [G,4] xyxy (đã kẹp), peaks [P,2], H = chiều cao ảnh,
    ps = data.pseudo_size (knn, beta, min_frac, max_frac). Hàm thuần."""
    gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
    gw, gh = gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]
    gs = np.sqrt(gw * gh)
    img_s, img_w, img_h = _gmean(gs), _gmean(gw), _gmean(gh)
    knn = lambda p: pseudo_sizes(p, ps["knn"], ps["beta"], ps["min_frac"] * H, ps["max_frac"] * H)  # noqa: E731

    out = {}
    for src in CENTERS:
        if src == "gt":
            c = (gt[:, :2] + gt[:, 2:]) / 2
            w_obj, h_obj = gw, gh
        else:
            c = np.asarray(peaks, dtype=np.float64).reshape(-1, 2)
            # đỉnh ghép được -> cỡ của GT đó; không ghép được -> cỡ chung của ảnh (không có GT thì không biết)
            w_obj, h_obj = np.full(len(c), img_w), np.full(len(c), img_h)
            for i, j in match_points_to_boxes(c, gt)["pairs"]:
                w_obj[i], h_obj[i] = gw[j], gh[j]
        n = len(c)
        if n == 0 or not len(gt):
            for sz in SIZES:
                out[(src, sz)] = np.zeros((0, 4))
            continue
        k = knn(c)
        wh = {
            "knn_sq": (k, k),
            "knn_rel_sq": (k * img_s / _gmean(k),) * 2,
            "img_sq": (np.full(n, img_s),) * 2,
            "img_wh": (np.full(n, img_w), np.full(n, img_h)),
            "obj_sq": (np.sqrt(w_obj * h_obj),) * 2,
            "gt_wh": (w_obj, h_obj),
        }
        for sz, (w, h) in wh.items():
            if sz == "knn_sq":
                out[(src, sz)] = pseudo_boxes(c, w)                     # đúng box giả BETA0
            else:
                out[(src, sz)] = np.concatenate([c - np.stack([w, h], 1) / 2, c + np.stack([w, h], 1) / 2], 1)
    return out


def _job(args):
    iid, img_path, gt_px, peaks, ps = args
    with Image.open(img_path) as im:
        W, H = im.size                                                    # chỉ đọc header
    gt = scale_boxes(gt_px, 1.0, W, H)
    sets = box_sets(gt, peaks, H, ps)
    g = xyxy_to_cxcywh(gt)
    recs = {}
    for key, b in sets.items():
        sc = box_iou(b, gt)[0].max(1) if len(b) and len(gt) else np.zeros(len(b))
        recs[key] = {"boxes": xyxy_to_cxcywh(b), "scores": sc, "keep": np.argsort(-sc, kind="stable"), "gt": g}
    return iid, recs


def _by_density(recs, thr):
    """oracle_recall tại IoU thr theo số GT của ảnh (mỗi tâm một box ⇒ = recall khi score = IoU)."""
    acc = {name: [0, 0] for _, _, name in DENSITY_BINS}
    for r in recs:
        n = len(r["gt"])
        name = next(nm for lo, hi, nm in DENSITY_BINS if lo <= n <= hi)
        acc[name][1] += n
        if n and len(r["boxes"]):
            acc[name][0] += int((box_iou(c2x(r["boxes"]), c2x(r["gt"]))[0].max(0) >= thr).sum())
    return {name: h / max(g, 1) for name, (h, g) in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--config", default="config/beta/beta0.yaml", help="lấy data.points + data.pseudo_size")
    ap.add_argument("--points", default=None, help="ghi đè data.points")
    ap.add_argument("--splits", nargs="+", default=list(SPLITS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="chỉ N ảnh đầu mỗi split (chạy thử)")
    ap.add_argument("--report", default=None, help="JSON số đo (vào /mnt/disk1/aiotlab/haitn/output/detection/beta0/)")
    a = ap.parse_args()

    t0 = time.time()
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    ps = cfg["data"]["pseudo_size"]
    pts_path = a.points or cfg["data"]["points"]
    pt = PointTable(pts_path)
    log(f"[G1] config {a.config} | pseudo_size {ps} | điểm {pts_path} (tách đỉnh {pt.params})")

    report = {"pseudo_size": ps, "points": pts_path, "point_params": pt.params, "splits": {}}
    for split in a.splits:
        t = time.time()
        items = scan_ce130(a.ce130, split)
        if a.limit:
            items = items[: a.limit]
        miss = [it["image_id"] for it in items if it["image_id"] not in pt]
        if miss:
            sys.exit(f"{len(miss)} ảnh {split} không có trong {pts_path} (vd. {miss[:3]})")
        log(f"[scan] {split}: {len(items)} ảnh, {sum(len(x['boxes_xyxy_px']) for x in items)} box GT | "
            f"{fmt_time(time.time() - t)}")
        jobs = [(it["image_id"], it["img_path"], it["boxes_xyxy_px"], pt[it["image_id"]], ps) for it in items]
        recs, t1 = {key: [] for key in [(c, s) for c in CENTERS for s in SIZES]}, time.time()
        pool = Pool(a.workers) if a.workers > 0 else None
        for i, (_, r) in enumerate(pool.imap(_job, jobs, chunksize=8) if pool else map(_job, jobs), 1):
            for key, v in r.items():
                recs[key].append(v)
            if i % 400 == 0 or i == len(jobs):
                el = time.time() - t1
                log(f"  dựng box {i}/{len(jobs)} | {fmt_time(el)} | còn ~{fmt_time(el / i * (len(jobs) - i))}")
        if pool:
            pool.close()
            pool.join()

        log(f"\n[{split}] trần AP (score = IoU thật, mỗi tâm 1 box, không NMS):")
        log(f"  {'tâm':5s} {'cỡ':10s} | {'AP50':>6s} {'AP75':>6s} {'AP_coco':>7s} | {'recall':>6s} {'prec':>6s} | "
            f"{'bestIoU tb/tv':>13s} | recall@0.5 ≤30 / 31-100 / >100 vật | recall@0.75 ≤30 / 31-100 / >100")
        res, t2 = {}, time.time()
        for j, (key, rs) in enumerate(recs.items(), 1):
            s = score_records(rs)
            d50, d75 = _by_density(rs, 0.5), _by_density(rs, 0.75)
            res[f"{key[0]}/{key[1]}"] = {**{k: s[k] for k in REPORT_KEYS}, "by_density@0.5": d50,
                                         "by_density@0.75": d75}
            log(f"  {key[0]:5s} {key[1]:10s} | {s['AP50']:6.3f} {s['AP75']:6.3f} {s['AP_coco']:7.3f} | "
                f"{s['recall']:6.3f} {s['precision']:6.3f} | {s['mean_bestIoU']:6.3f} {s['median_bestIoU']:6.3f} | "
                + " / ".join(f"{v:.3f}" for v in d50.values()) + " | "
                + " / ".join(f"{v:.3f}" for v in d75.values())
                + f"   ({fmt_time(time.time() - t2)}, còn ~{fmt_time((time.time() - t2) / j * (len(recs) - j))})")
        report["splits"][split] = res
        log(f"[{split}] xong {fmt_time(time.time() - t)} (tổng {fmt_time(time.time() - t0)})\n")

    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        log(f"-> {a.report}")
    log(f"[G1] xong {fmt_time(time.time() - t0)}")


if __name__ == "__main__":
    main()
