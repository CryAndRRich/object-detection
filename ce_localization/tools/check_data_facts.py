#!/usr/bin/env python3
"""Đo lại TỪ DỮ LIỆU THẬT các con số hình học của CE-130 hay bị trích sai.

Dự án đã có tiền lệ số liệu sai được trích lại nhiều lượt ("box rộng 2,4–4,4 ô lưới" — đúng
là 1,96 ô @512px). Cần số chính xác thì chạy tool này, đừng trích lại từ docs.
Không train, không GPU, không CLIP — chỉ đọc annotation.

  python tools/check_data_facts.py --out /mnt/disk1/aiotlab/haitn/output/data/data_facts.json

Hình học theo letterbox của `data/dataset.py` (scale = min(T/W, T/H), box kẹp vào vùng ảnh thật).
Mặc định `--image-size 512 --cell 16` = lưới 32×32 ô 16 px, đơn vị của con số "1,96 ô" trong
CLAUDE.md; ô P5 ở canvas 1024 là `--image-size 1024 --cell 32`.
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.dataset import scale_boxes, scan_ce130  # noqa: E402


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data/all_phase2_V2")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--cell", type=int, default=16, help="cạnh một ô lưới, pixel canvas")
    ap.add_argument("--out", default="data_facts.json")
    a = ap.parse_args()

    root, size = a.data_root, a.image_size
    grid = size // a.cell
    print(f"root={root}  image_size={size}  grid={grid}x{grid} (ô {a.cell} px)", flush=True)

    res = {}
    for split in ("train", "val", "test"):
        items = scan_ce130(root, split)
        n_per_img, w_cells, h_cells, area_pct, nn_dist_rel = [], [], [], [], []
        n_over_half = 0

        for it in items:
            W, H = Image.open(it["img_path"]).size        # chỉ đọc header, không giải mã JPEG
            sc = min(size / W, size / H)
            bx = scale_boxes(it["boxes_xyxy_px"], sc, int(W * sc), int(H * sc)) / size
            b = np.concatenate([(bx[:, :2] + bx[:, 2:]) / 2, bx[:, 2:] - bx[:, :2]], 1)   # cxcywh [0,1]
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
            "n_images": len(items),
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
            "n_classes": len({it["text"] for it in items}),
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
