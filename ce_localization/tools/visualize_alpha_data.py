#!/usr/bin/env python3
"""Cửa G2 của EXPERIMENT ALPHA: vẽ ảnh ĐÚNG như model nhận (letterbox 512 như CE-Loc gốc) kèm box
GT và ranh giới vùng ảnh thật, để xem bằng mắt trước khi train. CPU, không cần model / CLIP.

Mỗi hình: canvas 512 (đã bỏ chuẩn hoá để xem), box GT xanh lá, vạch đỏ = mép dưới vùng ảnh thật,
lưới xám = ô P5 (32 px) — ALPHA2 lấy mỗi ô thật làm một token.

  cd object-detection/ce_localization
  python tools/visualize_alpha_data.py --split train --n 12 --out ../../output/alpha_data_viz/train
  python tools/visualize_alpha_data.py --split test  --n 12 --out ../../output/alpha_data_viz/test

ALPHA3 (`--density full|partial|empty|mix`): density đúng như kênh 4 model nhận, tô ĐỎ chồng lên ảnh
(đậm = mật độ cao) — blob phải nằm trên vật, trong box GT. `mix` rút theo `--epoch`.
  python tools/visualize_alpha_data.py --split train --n 12 --image-size 1024 --density mix \
      --out ../../output/alpha_data_viz/alpha3_train_mix
"""

import argparse
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.alpha.data import IMAGENET_MEAN, IMAGENET_STD, AlphaCE130  # noqa: E402
from ce_localization.alpha.density import MODES, DensityIndex  # noqa: E402


def to_uint8(img_chw):
    x = img_chw[:3].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    if img_chw.shape[0] == 4:                       # density: trộn về đỏ theo mật độ
        d = img_chw[3].numpy()[..., None]
        x = x * (1 - 0.6 * d) + np.array([1.0, 0.0, 0.0]) * 0.6 * d
    return (x.clip(0, 1) * 255).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../data/all_phase2_V2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", default=None, choices=MODES)
    ap.add_argument("--density-root", default="../data/samples")
    ap.add_argument("--density-index", default="../data/density_index.json")
    ap.add_argument("--epoch", type=int, default=0, help="chỉ cho --density mix")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dindex = DensityIndex(a.density_index, a.density_root) if a.density else None
    ds = AlphaCE130(a.root, a.split, a.image_size, density=a.density, density_index=dindex)
    ds.epoch = a.epoch
    rng = np.random.default_rng(a.seed)
    idx = sorted(rng.choice(len(ds), size=min(a.n, len(ds)), replace=False).tolist())
    os.makedirs(a.out, exist_ok=True)
    n_tok = []
    for i in idx:
        s = ds[i]
        nh, nw = s["valid_hw"]
        canvas = Image.fromarray(to_uint8(s["image"]))
        d = ImageDraw.Draw(canvas)
        for k in range(0, a.image_size + 1, 32):
            d.line([(k, 0), (k, a.image_size)], fill=(90, 90, 90))
            d.line([(0, k), (a.image_size, k)], fill=(90, 90, 90))
        for x1, y1, x2, y2 in s["boxes"].tolist():
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0))
        d.line([(0, nh), (nw, nh)], fill=(255, 0, 0), width=2)
        rows = math.ceil(nh / 32)
        n_tok.append(rows * math.ceil(nw / 32))
        dk = "" if s["density_kind"] is None else f" | density {s['density_kind']}"
        d.text((4, 4), f"{s['image_id']} {s['text']} | {len(s['boxes'])} box | nh={nh} nw={nw} | "
                       f"{rows}x{math.ceil(nw / 32)} ô P5{dk}", fill=(255, 255, 0))
        canvas.save(os.path.join(a.out, f"{s['image_id']}.png"))
        print(f"  {s['image_id']:>6s} {s['text']:>14s} | {len(s['boxes']):4d} box | nh {nh:3d} nw {nw:3d} | "
              f"token lưới {n_tok[-1]}" + ("" if s["density_kind"] is None else
                                           f" | density {s['density_kind']}"), flush=True)
    print(f"{len(idx)} hình -> {a.out} | token lưới P5: trung vị {int(np.median(n_tok))}, "
          f"min {min(n_tok)}, max {max(n_tok)}")


if __name__ == "__main__":
    main()
