#!/usr/bin/env python3
"""ALPHA3: dựng chỉ mục density MỘT lần — mỗi ảnh gốc (iid) -> các bản density trong
`samples/{train,test}/density/{iid}_{k}.png`, sắp theo diện tích blob giảm dần (bản đầu = `full`).
Giải mã jet + đếm pixel mức > 0 (`alpha/density.py`). Kiểm luôn: phủ đủ 3 split detect, mọi màu
thuộc bảng jet, kích thước density == ảnh gốc.

~26k PNG: vài phút với 8 worker (tính cả đọc đĩa) -> chạy nền trên server:
  cd object-detection/ce_localization
  LOG=/mnt/disk1/aiotlab/haitn/log/density_index_$(date +%m%d_%H%M).log
  nohup python tools/build_density_index.py --samples ../data/samples --ce130 ../data/all_phase2_V2 \\
      --out ../data/density_index.json --workers 8 > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.density import build_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--ce130", default="../data/all_phase2_V2", help="để kiểm độ phủ + kích thước")
    ap.add_argument("--out", default="../data/density_index.json")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    t0 = time.time()
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    log(f"[density] quét {a.samples} ...")
    idx = build_index(a.samples, workers=a.workers, log=log)
    V = idx["variants"]
    n_var = np.array([len(v) for v in V.values()])
    log(f"[density] {idx['n_files']} file, {len(V)} ảnh | bản/ảnh: min {n_var.min()} trung vị "
        f"{int(np.median(n_var))} max {n_var.max()} | ảnh chỉ 1 bản {np.mean(n_var == 1) * 100:.1f} % | "
        f"màu xa bảng jet nhất {idx['max_color_dist']:.2f} (0 = khớp tuyệt đối) | {time.time() - t0:.0f} s")
    empty_full = sum(v[0][1] == 0 for v in V.values())
    log(f"[density] ảnh mà bản full cũng trống: {empty_full} ({empty_full / len(V) * 100:.1f} %)")

    ok = True
    if a.ce130 and os.path.isdir(a.ce130):
        for split in ("train", "val", "test"):
            first = {}
            for br in sorted(glob.glob(os.path.join(a.ce130, split, "*"))):
                first.setdefault(os.path.basename(br).split("_b")[0], br)
            miss = [i for i in first if i not in V]
            bad_size, ratio = 0, []
            for iid, br in first.items():
                if iid not in V:
                    continue
                gt = os.path.join(br, "ground_truth.jpg")
                if os.path.exists(gt) and list(Image.open(gt).size) != V[iid][0][2]:
                    bad_size += 1
                full = V[iid][0][1]
                if full > 0 and len(V[iid]) > 1:
                    ratio.append(V[iid][-1][1] / full)
            ok &= not miss
            log(f"[density] {split:5s}: {len(first)} ảnh | thiếu density {len(miss)} {miss[:5]} | "
                f"kích thước khác ảnh gốc {bad_size} | diện tích partial/full trung vị "
                f"{np.median(ratio) if ratio else float('nan'):.2f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(idx, f)
    log(f"[density] -> {a.out} ({os.path.getsize(a.out) / 2 ** 20:.1f} MB) | {time.time() - t0:.0f} s"
        + ("" if ok else " | ⚠️ CÓ ẢNH THIẾU DENSITY"))


if __name__ == "__main__":
    main()
