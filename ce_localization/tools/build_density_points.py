#!/usr/bin/env python3
"""Cửa G0 của EXPERIMENT BETA (docs/EXPERIMENT_BETA.md mục 1, 7): tách đỉnh density `full` thành điểm
vật cho mọi ảnh 3 split -> `data/density_points.json`, và đo chất lượng nhãn so với box GT.

Density `full` = bản density diện tích blob lớn nhất của mỗi ảnh (chỉ mục `data/density_index.json`
của ALPHA3, `tools/build_density_index.py`). Đỉnh: `data/points.find_peaks(levels, tau, radius)`.

1. Quét lưới (tau, radius) trên MỌI ảnh; chọn cặp có F1 (ghép MỘT-MỘT đỉnh <-> box GT chứa nó) cao
   nhất trên TRAIN (val / test chỉ để báo). `--tau --radius` để ép một cặp.
2. Với cặp đã chọn, báo theo split và theo số vật trong ảnh (<=30 / 31-100 / >100):
     precision nhãn = đỉnh được ghép / số đỉnh ; recall nhãn = box GT được ghép / số GT
     recall_any     = box GT chứa >= 1 đỉnh ; precision_any = đỉnh nằm trong >= 1 box GT
     box GT chứa >= 2 đỉnh (đỉnh thừa) ; đỉnh nằm trong >= 2 box GT
3. Cỡ giả (TRAIN): tỉ số kNN(đỉnh) / sqrt(w·h) của box GT được ghép -> đề xuất `beta` = 1 / trung vị;
   `min_frac` / `max_frac` = p2 / p98 của sqrt(w·h) / H trên box GT train. In sẵn dòng yaml cho
   `data.pseudo_size` của config/beta/beta0.yaml (đây là 3 số vô hướng lấy từ box GT, ghi rõ khi báo cáo).
4. `--config-in config/beta/beta0.yaml --config-out <file>`: ghi một BẢN config đã điền `data.pseudo_size`
   (đề xuất ở bước 3, hoặc `--pseudo-size knn beta min_frac max_frac`) và `data.points` = `--out`
   (đường dẫn tuyệt đối) — dùng khi `pseudo_size` trong git chưa khớp (Kaggle ô 6b). Trên server
   config/beta/beta0.yaml đã điền số G0 2026-09-30 -> train thẳng bằng file đó.

~3.600 PNG + quét annotation 3 split (4–6 phút trên đĩa dùng chung) ⇒ ước 5–15 phút, chạy NỀN:
  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  LOG=/mnt/disk1/aiotlab/haitn/log/detection/beta0/beta_g0_points_$(date +%m%d_%H%M).log
  nohup python tools/build_density_points.py --out ../data/density_points.json \\
      --report /mnt/disk1/aiotlab/haitn/output/detection/beta0/beta_g0_points.json --workers 8 > $LOG 2>&1 &
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.density import DensityIndex, load_density_levels  # noqa: E402
from ce_localization.engine.evaluate import DENSITY_BINS  # noqa: E402
from ce_localization.data.points import find_peaks, knn_distance, match_points_to_boxes  # noqa: E402
from ce_localization.data.dataset import scan_ce130  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402

SPLITS = ("train", "val", "test")
COUNT_KEYS = ("n_peaks", "n_gt", "tp", "gt_any", "peak_any", "gt_multi", "peak_shared")


def label_stats(points, gt):
    """Đếm cho MỘT ảnh: ghép một-một + các tỉ lệ 'any'. -> (dict đếm, cặp ghép)."""
    m = match_points_to_boxes(points, gt)
    ins = m["inside"]
    per_gt = ins.sum(0) if ins.size else np.zeros(len(gt), dtype=int)
    per_pt = ins.sum(1) if ins.size else np.zeros(len(points), dtype=int)
    return {"n_peaks": len(points), "n_gt": len(gt), "tp": len(m["pairs"]),
            "gt_any": int((per_gt >= 1).sum()), "peak_any": int((per_pt >= 1).sum()),
            "gt_multi": int((per_gt >= 2).sum()), "peak_shared": int((per_pt >= 2).sum())}, m["pairs"]


def _job(job):
    iid, path, gt, taus, radii = job
    out = {}
    lv = load_density_levels(path) if path else None
    for r in radii:
        for tau in taus:
            pts = find_peaks(lv, tau, r) if lv is not None else np.zeros((0, 2))
            out[(tau, r)] = (pts.astype(np.float32), label_stats(pts, gt)[0])
    return iid, out


def summarise(counts):
    c = {k: sum(x[k] for x in counts) for k in COUNT_KEYS}
    p, r = c["tp"] / max(c["n_peaks"], 1), c["tp"] / max(c["n_gt"], 1)
    per_img = [x["n_peaks"] / x["n_gt"] for x in counts if x["n_gt"]]
    return {"n_img": len(counts), **c, "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "recall_any": c["gt_any"] / max(c["n_gt"], 1), "precision_any": c["peak_any"] / max(c["n_peaks"], 1),
            "gt_multi_frac": c["gt_multi"] / max(c["n_gt"], 1),
            "peak_shared_frac": c["peak_shared"] / max(c["n_peaks"], 1),
            "peaks_per_gt_p10_p50_p90": [float(np.percentile(per_img, q)) for q in (10, 50, 90)] if per_img else []}


def write_config(a, suggested, log):
    """Bản config đã điền `data.pseudo_size` (đề xuất G0 hoặc `--pseudo-size`) + `data.points`."""
    if a.pseudo_size is not None:
        k, b, lo, hi = a.pseudo_size
        ps, src = {"knn": int(k), "beta": b, "min_frac": lo, "max_frac": hi}, "--pseudo-size"
    elif suggested:
        ps, src = suggested, "đề xuất G0"
    else:
        sys.exit("không có đề xuất pseudo_size (train không có cặp đỉnh–GT nào) — truyền --pseudo-size")
    with open(a.config_in) as f:
        cfg = yaml.safe_load(f)
    cfg["data"]["pseudo_size"] = {"knn": int(ps["knn"]), "beta": round(float(ps["beta"]), 4),
                                  "min_frac": round(float(ps["min_frac"]), 5), "max_frac": round(float(ps["max_frac"]), 5)}
    cfg["data"]["points"] = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(os.path.abspath(a.config_out)), exist_ok=True)
    with open(a.config_out, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    log(f"-> {a.config_out} (pseudo_size {cfg['data']['pseudo_size']} từ {src}; points {cfg['data']['points']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--density-index", default="../data/density_index.json")
    ap.add_argument("--out", default="../data/density_points.json")
    ap.add_argument("--report", default=None, help="JSON số đo (vào /mnt/disk1/aiotlab/haitn/output/detection/beta0/)")
    ap.add_argument("--taus", type=int, nargs="+", default=[8, 32, 64, 96, 128, 160])
    ap.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    ap.add_argument("--tau", type=int, default=None, help="ép tau (cùng --radius), bỏ qua chọn theo F1")
    ap.add_argument("--radius", type=int, default=None)
    ap.add_argument("--knn", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="chỉ N ảnh đầu mỗi split (chạy thử)")
    ap.add_argument("--config-in", default=None, help="config gốc (vd. config/beta/beta0.yaml) để ghi bản đã điền")
    ap.add_argument("--config-out", default=None, help="bản config đã điền data.pseudo_size + data.points")
    ap.add_argument("--pseudo-size", type=float, nargs=4, default=None, metavar=("KNN", "BETA", "MIN_FRAC", "MAX_FRAC"),
                    help="ép pseudo_size thay cho đề xuất của G0")
    a = ap.parse_args()
    if (a.config_in is None) != (a.config_out is None):
        sys.exit("--config-in và --config-out phải đi cùng nhau")
    if (a.tau is None) != (a.radius is None):
        sys.exit("--tau và --radius phải đi cùng nhau")
    taus, radii = ([a.tau], [a.radius]) if a.tau is not None else (a.taus, a.radii)

    t0 = time.time()
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    di = DensityIndex(a.density_index, a.samples)
    items = {}
    for split in SPLITS:
        t = time.time()
        its = scan_ce130(a.ce130, split)
        if a.limit:
            its = its[: a.limit]
        items[split] = its
        log(f"[scan] {split}: {len(its)} ảnh, {sum(len(x['boxes_xyxy_px']) for x in its)} box GT | "
            f"{fmt_time(time.time() - t)} (tổng {fmt_time(time.time() - t0)})")

    jobs, split_of, gt_of, h_of = [], {}, {}, {}
    for split, its in items.items():
        for it in its:
            iid = it["image_id"]
            if iid not in di:
                sys.exit(f"ảnh {iid} ({split}) không có trong chỉ mục density {a.density_index}")
            rel, _ = di.pick(iid, "full")
            area = di.variants[iid][0][1]
            gt = np.asarray(it["boxes_xyxy_px"], dtype=np.float64).reshape(-1, 4)
            jobs.append((iid, di.path(rel) if area > 0 else None, gt, taus, radii))
            split_of[iid], gt_of[iid] = split, gt
            h_of[iid] = di.variants[iid][0][2][1]                       # [W, H] ảnh gốc
    log(f"[peaks] {len(jobs)} ảnh × {len(taus) * len(radii)} cặp (tau, radius), {a.workers} worker ...")
    res, done, t1 = {}, 0, time.time()
    pool = Pool(a.workers) if a.workers > 0 else None
    for iid, out in (pool.imap_unordered(_job, jobs, chunksize=8) if pool else map(_job, jobs)):
        res[iid] = out
        done += 1
        if done % 400 == 0 or done == len(jobs):
            el = time.time() - t1
            log(f"  {done}/{len(jobs)} | {fmt_time(el)} | còn ~{fmt_time(el / done * (len(jobs) - done))}")
    if pool:
        pool.close()
        pool.join()

    # ---------------------------------------------------------------- chọn (tau, radius) theo F1 train
    grid = {}
    for combo in res[next(iter(res))]:
        grid[combo] = {s: summarise([res[i][combo][1] for i in res if split_of[i] == s]) for s in SPLITS}
    log("\n[quét] F1 ghép một-một trên TRAIN (precision / recall nhãn):")
    for (tau, r), v in sorted(grid.items(), key=lambda kv: -kv[1]["train"]["f1"]):
        tr = v["train"]
        log(f"  tau {tau:3d} radius {r} | F1 {tr['f1']:.4f} | P {tr['precision']:.4f} R {tr['recall']:.4f}")
    best = max(grid, key=lambda c: grid[c]["train"]["f1"])
    tau, radius = best
    log(f"\n[chọn] tau {tau} radius {radius}" + (" (ép bằng --tau --radius)" if a.tau is not None else ""))

    # ---------------------------------------------------------------- báo cặp đã chọn
    report = {"params": {"tau": tau, "radius": radius, "variant": "full"}, "grid": {}, "chosen": {}}
    for (t_, r_), v in grid.items():
        report["grid"][f"tau{t_}_r{r_}"] = {s: {k: v[s][k] for k in ("precision", "recall", "f1")} for s in SPLITS}
    for s in SPLITS:
        ids = [i for i in res if split_of[i] == s]
        rep = {"all": summarise([res[i][best][1] for i in ids])}
        for lo, hi, name in DENSITY_BINS:
            sub = [res[i][best][1] for i in ids if lo <= len(gt_of[i]) <= hi]
            rep[name] = summarise(sub)
        report["chosen"][s] = rep
        log(f"\n[nhãn] {s}:")
        for name, v in rep.items():
            log(f"  {name:10s} {v['n_img']:4d} ảnh | GT {v['n_gt']:6d} | đỉnh {v['n_peaks']:6d} | "
                f"P {v['precision']:.4f} R {v['recall']:.4f} F1 {v['f1']:.4f} | recall_any {v['recall_any']:.4f} "
                f"precision_any {v['precision_any']:.4f} | GT >=2 đỉnh {v['gt_multi_frac']:.4f} | "
                f"đỉnh trong >=2 GT {v['peak_shared_frac']:.4f} | đỉnh/GT p10/50/90 {v['peaks_per_gt_p10_p50_p90']}")
        n_empty = sum(1 for i in ids if len(res[i][best][0]) == 0)
        log(f"  ảnh không có đỉnh nào: {n_empty} ({n_empty / max(len(ids), 1) * 100:.1f} %)")

    # ---------------------------------------------------------------- cỡ giả (train)
    ratio, gt_frac = [], []
    for i in res:
        if split_of[i] != "train":
            continue
        gt, pts = gt_of[i], res[i][best][0].astype(np.float64)
        gt_frac += (np.sqrt((gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])) / h_of[i]).tolist()
        _, pairs = label_stats(pts, gt)
        d = knn_distance(pts, a.knn)
        for pi, gj in pairs:
            s_true = np.sqrt((gt[gj, 2] - gt[gj, 0]) * (gt[gj, 3] - gt[gj, 1]))
            if not np.isnan(d[pi]) and s_true > 0:
                ratio.append(d[pi] / s_true)
    ratio, gt_frac = np.asarray(ratio), np.asarray(gt_frac)
    if len(ratio):
        beta = float(1 / np.median(ratio))
        err = np.abs(np.log(beta * ratio))
        pq = {q: float(np.percentile(ratio, q)) for q in (5, 25, 50, 75, 95)}
        mn, mx = float(np.percentile(gt_frac, 2)), float(np.percentile(gt_frac, 98))
        log(f"\n[cỡ giả] train, {len(ratio)} cặp: kNN{a.knn}/sqrt(wh) p5/25/50/75/95 "
            f"{' / '.join(f'{v:.2f}' for v in pq.values())} | beta = 1/trung vị = {beta:.3f} | "
            f"|log(ŝ/sqrt(wh))| trung vị {np.median(err):.3f}, trong ×2: {np.mean(err <= np.log(2)) * 100:.1f} %")
        log(f"[cỡ giả] sqrt(wh)/H của box GT train: p2 {mn:.4f} p50 {np.median(gt_frac):.4f} p98 {mx:.4f}")
        report["pseudo_size"] = {"knn": a.knn, "beta": beta, "min_frac": mn, "max_frac": mx,
                                 "ratio_percentiles": pq, "median_abs_log_err": float(np.median(err))}
        log(f"yaml: pseudo_size: {{knn: {a.knn}, beta: {beta:.3f}, min_frac: {mn:.4f}, max_frac: {mx:.4f}}}")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pts_out = {i: np.round(res[i][best][0].astype(np.float64), 2).tolist() for i in sorted(res)}
    with open(a.out, "w") as f:
        json.dump({"params": report["params"], "points": pts_out}, f)
    log(f"\n-> {a.out} ({len(pts_out)} ảnh, {os.path.getsize(a.out) / 2 ** 20:.1f} MB)")
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
        log(f"-> {a.report}")
    if a.config_out:
        write_config(a, report.get("pseudo_size"), log)
    log(f"[G0] xong {fmt_time(time.time() - t0)}")


if __name__ == "__main__":
    main()
