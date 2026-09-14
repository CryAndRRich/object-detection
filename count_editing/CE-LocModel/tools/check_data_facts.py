#!/usr/bin/env python3
"""ĐO LẠI TỪ DỮ LIỆU THẬT các con số đang được trích đi trích lại trong docs.

VÌ SAO TỒN TẠI
--------------
Thiết kế của cửa chặn keypoint đang dựa vào một loạt con số lấy từ docs mà KHÔNG ai đọc lại
từ dữ liệu trong phiên này:

  - "CE-130 có 20-30 vật/ảnh (trung vị), mean 37,6-48,5"   -> quyết định dải quét K
  - "box rộng 2,4-4,4 ô lưới"                              -> quyết định --min-dist-cells=2.0
                                                              và ngưỡng "trong 1 ô"
  - "vật chiếm ~0,4 % diện tích ảnh"
  - "test có lô annotation rác, val thì không"             -> quyết định đo trên val

Docs có thể đúng, nhưng dự án đã có tiền lệ số liệu sai được trích lại nhiều lượt (con số
2,7 % — đo trên model chưa train). Trước khi để những con số này quyết định thiết kế thì phải
đọc lại từ dữ liệu.

KHÔNG train gì, KHÔNG cần GPU, KHÔNG cần CLIP. Chỉ đọc annotation.

CHẠY (TRÊN SERVER)
------------------
  python tools/check_data_facts.py --out /mnt/disk1/aiotlab/haitn/log/data_facts.json
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection  # noqa: E402


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="data_facts.json")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    root = a.data_root or cfg["data"]["root"]
    size = cfg["data"]["image_size"]
    grid = size // 16                      # ViT-B/16 -> 32 ô mỗi chiều
    print(f"root={root}  image_size={size}  grid={grid}x{grid}", flush=True)

    res = {}
    for split in ("train", "val", "test"):
        ds = CE130Detection(root, split, size)
        n_per_img, w_cells, h_cells, area_pct, nn_dist_rel = [], [], [], [], []
        n_over_half = 0

        for i in range(len(ds)):
            # need_image=False: KHÔNG giải mã JPEG, chỉ cần annotation.
            s = ds.__getitem__(i, need_image=False)
            b = np.asarray(s["boxes"], dtype=np.float64)   # cxcywh trong [0,1]
            n_per_img.append(len(b))
            if len(b) == 0:
                continue
            w_cells.extend(b[:, 2] * grid)                 # -> ĐƠN VỊ Ô LƯỚI
            h_cells.extend(b[:, 3] * grid)
            area_pct.extend(b[:, 2] * b[:, 3] * 100)
            n_over_half += int(((b[:, 2] * b[:, 3]) > 0.5).sum())

            # Khoảng cách tới vật gần nhất / kích thước vật -- đại lượng mà docs/01 mục 4.2
            # dùng để lập luận "box<->box suy ra được chỗ hợp lý".
            if len(b) >= 2:
                c = b[:, :2]
                d = np.linalg.norm(c[:, None] - c[None], axis=-1)
                np.fill_diagonal(d, np.inf)
                sz = np.sqrt(b[:, 2] * b[:, 3])
                nn_dist_rel.extend(d.min(axis=1) / np.maximum(sz, 1e-9))

        n_per_img = np.asarray(n_per_img)
        w_cells = np.asarray(w_cells)
        h_cells = np.asarray(h_cells)
        area_pct = np.asarray(area_pct)
        nn_dist_rel = np.asarray(nn_dist_rel)

        res[split] = {
            "n_images": len(ds),
            "n_boxes_total": int(n_per_img.sum()),
            "boxes_per_image": {
                "median": float(np.median(n_per_img)), "mean": float(n_per_img.mean()),
                "p10": pct(n_per_img, 10), "p90": pct(n_per_img, 90),
                "max": int(n_per_img.max()),
                "n_images_zero_box": int((n_per_img == 0).sum()),
            },
            "box_width_cells": {
                "median": float(np.median(w_cells)), "mean": float(w_cells.mean()),
                "p10": pct(w_cells, 10), "p90": pct(w_cells, 90),
            },
            "box_height_cells": {
                "median": float(np.median(h_cells)), "mean": float(h_cells.mean()),
                "p10": pct(h_cells, 10), "p90": pct(h_cells, 90),
            },
            "box_area_pct_of_image": {
                "median": float(np.median(area_pct)), "mean": float(area_pct.mean()),
                "p90": pct(area_pct, 90),
            },
            "n_boxes_over_half_image": n_over_half,
            "nn_dist_over_size": {
                "median": float(np.median(nn_dist_rel)) if len(nn_dist_rel) else float("nan"),
                "p10": pct(nn_dist_rel, 10), "p90": pct(nn_dist_rel, 90),
            },
            "n_classes": len({it["text"] for it in ds.items}),
        }

    # ------------------------------------------------------------------ in
    print()
    print("  split | ảnh  |  box   | box/ảnh (median/mean/p90/max) | rộng ô (med/p10-p90)")
    print("  ------+------+--------+-------------------------------+---------------------")
    for k, v in res.items():
        bp, bw = v["boxes_per_image"], v["box_width_cells"]
        print(f"  {k:5s} | {v['n_images']:4d} | {v['n_boxes_total']:6d} | "
              f"{bp['median']:6.1f} / {bp['mean']:5.1f} / {bp['p90']:5.0f} / {bp['max']:5d} | "
              f"{bw['median']:5.2f} ({bw['p10']:.2f}-{bw['p90']:.2f})")
    print()
    print("  split | cao ô (med)  | diện tích % ảnh (med) | box >50% ảnh | kc/kích thước (med, p10-p90)")
    print("  ------+--------------+-----------------------+--------------+------------------------------")
    for k, v in res.items():
        bh, ar, nn = v["box_height_cells"], v["box_area_pct_of_image"], v["nn_dist_over_size"]
        print(f"  {k:5s} | {bh['median']:11.2f} | {ar['median']:21.3f} | "
              f"{v['n_boxes_over_half_image']:12d} | "
              f"{nn['median']:.2f}  ({nn['p10']:.2f}-{nn['p90']:.2f})")
    print()
    print("  SO VỚI SỐ ĐANG GHI TRONG DOCS (kiểm xem có khớp không):")
    print("    docs: box/ảnh trung vị 20-30, mean 37,6-48,5")
    print("    docs: box rộng 2,4-4,4 ô lưới")
    print("    docs: vật chiếm ~0,4 % diện tích ảnh")
    print("    docs: test có 855 box >50% ảnh, train/val = 0")
    print("    docs: kc tới vật gần nhất / kích thước = median 0,83, p10-p90 0,60-1,22")
    print()

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
