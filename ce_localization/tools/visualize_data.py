#!/usr/bin/env python3
"""Cửa G2 của EXPERIMENT ALPHA: vẽ ảnh ĐÚNG như model nhận (letterbox 512 như CE-Loc gốc) kèm box
GT và ranh giới vùng ảnh thật, để xem bằng mắt trước khi train. CPU, không cần model / CLIP.

Mỗi hình: canvas 512 (đã bỏ chuẩn hoá để xem), box GT xanh lá, vạch đỏ = mép dưới vùng ảnh thật,
lưới xám = ô P5 (32 px) — ALPHA2 lấy mỗi ô thật làm một token.

  cd object-detection/ce_localization
  python tools/visualize_data.py --split train --n 12 --out ../../output/alpha/data_viz/train
  python tools/visualize_data.py --split test  --n 12 --out ../../output/alpha/data_viz/test

ALPHA3 (`--density full|partial|empty|mix`): density đúng như kênh 4 model nhận, tô ĐỎ chồng lên ảnh
(đậm = mật độ cao) — blob phải nằm trên vật, trong box GT. `mix` rút theo `--epoch`.
  python tools/visualize_data.py --split train --n 12 --image-size 1024 --density mix \
      --out ../../output/alpha/data_viz/alpha3_train_mix

BETA (`--config config/beta/beta0.yaml`: đích điểm, lấy `data.points` + `data.pseudo_size` từ config): vẽ
thêm ĐỈNH density (chấm đỏ) + BOX GIẢ (xanh dương) mà model học, cạnh box GT (xanh lá, chỉ để chấm).
  python tools/visualize_data.py --config config/beta/beta0.yaml --split train --n 12 --image-size 1024 \
      --out /mnt/disk1/aiotlab/haitn/output/beta/data_viz/train

GAMMA (`--config config/gamma/gamma0.yaml`, bài add): ảnh inpaint lượt t đúng như model nhận (canvas + density của
config), lỗ MỚI NHẤT (đích train) đỏ, lỗ cũ cam, vật đang có xanh lá; `--image original` vẽ ảnh gốc của cùng nhánh.
  python tools/visualize_data.py --config config/gamma/gamma0.yaml --split train --n 12 \
      --out /mnt/disk1/aiotlab/haitn/output/gamma/data_viz/train
"""

import argparse
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.dataset import IMAGENET_MEAN, IMAGENET_STD, CE130Dataset  # noqa: E402
from ce_localization.data.density import MODES, DensityIndex  # noqa: E402
from ce_localization.data.points import PointTable  # noqa: E402
from ce_localization.data.turns import IMAGE_KINDS, CE130AddDataset, TurnIndex  # noqa: E402


def to_uint8(img_chw):
    x = img_chw[:3].numpy().transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    if img_chw.shape[0] == 4:                       # density: trộn về đỏ theo mật độ
        d = img_chw[3].numpy()[..., None]
        x = x * (1 - 0.6 * d) + np.array([1.0, 0.0, 0.0]) * 0.6 * d
    return (x.clip(0, 1) * 255).round().astype(np.uint8)


def draw_add(cfg, a):
    d = cfg["data"]
    dens = d.get("density")
    if a.image == "original" and dens:
        dens = "full"
    dindex = DensityIndex(d["density_index"], d["density_root"]) if dens == "full" else None
    ds = CE130AddDataset(TurnIndex(d["turn_index"]), d["root"], d["samples_root"], a.split, d["image_size"],
                         density=dens, density_index=dindex, image=a.image)
    rng = np.random.default_rng(a.seed)
    idx = sorted(rng.choice(len(ds), size=min(a.n, len(ds)), replace=False).tolist())
    os.makedirs(a.out, exist_ok=True)
    for i in idx:
        s = ds[i]
        nh, nw = s["valid_hw"]
        canvas = Image.fromarray(to_uint8(s["image"]))
        dr = ImageDraw.Draw(canvas)
        for x1, y1, x2, y2 in s["objects"].tolist():
            dr.rectangle([x1, y1, x2, y2], outline=(0, 255, 0))
        for k, (x1, y1, x2, y2) in enumerate(s["holes"].tolist()):
            last = k == len(s["holes"]) - 1
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0) if last else (255, 160, 0), width=2 if last else 1)
        dr.line([(0, nh), (nw, nh)], fill=(255, 0, 0), width=2)
        dr.text((4, 4), f"{s['image_id']} {s['text']} | lượt {s['t']} | {len(s['objects'])} vật | {a.image} | "
                        f"density {dens}", fill=(255, 255, 0))
        canvas.save(os.path.join(a.out, f"{s['image_id']}_{a.image}.png"))
        print(f"  {s['image_id']:>14s} {s['text']:>14s} | lượt {s['t']} | {len(s['objects']):3d} vật | nh {nh} nw {nw}",
              flush=True)
    print(f"{len(idx)} hình -> {a.out}")


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
    ap.add_argument("--config", default=None, help="BETA: lấy data.targets / points / pseudo_size từ config")
    ap.add_argument("--image", default="inpainted", choices=IMAGE_KINDS, help="GAMMA: ảnh inpaint / ảnh gốc")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.config:
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
        if cfg.get("task") == "add":
            return draw_add(cfg, a)

    dindex = DensityIndex(a.density_index, a.density_root) if a.density else None
    targets, ptable, pseudo = "box", None, None
    if a.config:
        with open(a.config) as f:
            dcfg = yaml.safe_load(f)["data"]
        targets = dcfg.get("targets", "box")
        if targets == "point":
            ptable, pseudo = PointTable(dcfg["points"]), dcfg["pseudo_size"]
    ds = CE130Dataset(a.root, a.split, a.image_size, density=a.density, density_index=dindex,
                    targets=targets, points=ptable, pseudo=pseudo)
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
        for x1, y1, x2, y2 in s["gt_boxes"].tolist():
            d.rectangle([x1, y1, x2, y2], outline=(0, 255, 0))
        if targets == "point":                               # box giả (đích train) + đỉnh = tâm box giả
            for x1, y1, x2, y2 in s["boxes"].tolist():
                d.rectangle([x1, y1, x2, y2], outline=(0, 128, 255))
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 0, 0))
        d.line([(0, nh), (nw, nh)], fill=(255, 0, 0), width=2)
        rows = math.ceil(nh / 32)
        n_tok.append(rows * math.ceil(nw / 32))
        dk = "" if s["density_kind"] is None else f" | density {s['density_kind']}"
        npt = "" if targets == "box" else f" | {len(s['boxes'])} đỉnh"
        d.text((4, 4), f"{s['image_id']} {s['text']} | {len(s['gt_boxes'])} box GT{npt} | nh={nh} nw={nw} | "
                       f"{rows}x{math.ceil(nw / 32)} ô P5{dk}", fill=(255, 255, 0))
        canvas.save(os.path.join(a.out, f"{s['image_id']}.png"))
        print(f"  {s['image_id']:>6s} {s['text']:>14s} | {len(s['gt_boxes']):4d} box GT{npt} | nh {nh:3d} nw {nw:3d} | "
              f"token lưới {n_tok[-1]}" + ("" if s["density_kind"] is None else
                                           f" | density {s['density_kind']}"), flush=True)
    print(f"{len(idx)} hình -> {a.out} | token lưới P5: trung vị {int(np.median(n_tok))}, "
          f"min {min(n_tok)}, max {max(n_tok)}")


if __name__ == "__main__":
    main()
