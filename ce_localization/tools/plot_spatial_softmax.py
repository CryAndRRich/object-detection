#!/usr/bin/env python3
"""Vẽ output SpatialSoftmax của CE-Loc gốc (checkpoint của bài) — 2 hàng × 3 cột mỗi ảnh:
cột  Original (ảnh vào) | Density Map (kênh density model nhận, thang [0, 1]) | SpatialSoftmax Output
hàng 1 = density đầy đủ của ảnh ; hàng 2 = density trống ĐÚNG kiểu dữ liệu (PNG toàn nền jet (0, 0, 127)).
Text không vào nhánh ảnh (ResNet18 + SpatialSoftmax) nên không có trong tool này.

SpatialSoftmax Output: mỗi chấm = một kênh trong 512 kênh của layer4 ResNet18 (lưới 16×16 trên canvas 512), vị trí = toạ
độ kỳ vọng của softmax kênh đó (đổi đúng trục: toạ độ đầu của bài là DỌC; −1 / +1 = tâm ô đầu / cuối ⇒ pixel = 16 +
240·(u+1)). Màu = số ô hiệu dụng exp(entropy) (thang log 1–256: 1 = nhìn đúng 1 ô, 256 = trải đều — chấm khi đó chỉ là
trung bình, rơi về tâm); cỡ chấm ∝ ‖W_projection[:, 2c:2c+2]‖ (kênh model dùng nhiều thì to). Chỉ chấm vàng (nhọn) mới
đọc được là "nhìn vào đây".

`--image original` (mặc định): ảnh gốc CHƯA xoá (`ground_truth.jpg`), density = bản đầy đủ nhất của ảnh gốc (`full`, chỉ mục
ALPHA3); `--image inpainted`: ảnh của `samples/` + density của chính mẫu.

Chỉ nạp ResNet18 + vài ảnh, < 1 phút, CPU được (không cần CLIP):
  python tools/plot_spatial_softmax.py --ckpt ../weights/add/paper/best_model.pth \\
      --files test/images/6246_11.png:apple --out ../../output/add/viz/spatial_softmax_original
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.turns import image_inputs  # noqa: E402
from ce_localization.models.box_policy import BoxPolicy  # noqa: E402
from ce_localization.tools.viz_common import SHARP, dot_sizes, ss_scatter, tidy  # noqa: E402

T = 512


def build_inputs(a, rel, iid, img, dens):
    """-> tensor [4,T,T] đầu vào kiểu bài (`to_tensor`, density `.convert("L")`)."""
    if dens == "blank":
        d = Image.new("RGB", img.size, (0, 0, 127))
    elif a.image == "original":
        di = DensityIndex(a.density_index, a.samples)
        d = di.path(di.pick(iid, "full")[0])
    else:
        split, stem = rel.split("/")[0], os.path.splitext(os.path.basename(rel))[0]
        d = os.path.join(a.samples, split, "density", stem + ".png")
    return image_inputs(img, d, T)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint CE-Loc gốc của bài (weights/add/paper/best_model.pth)")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--ce130", default="../data/all_phase2_V2")
    ap.add_argument("--density-index", default="../data/density_index.json")
    ap.add_argument("--image", default="original", choices=["original", "inpainted"])
    ap.add_argument("--files", nargs="+", required=True, help="<split>/images/<file>.png ...")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model, _, info = BoxPolicy.load_celoc_paper(a.ckpt)
    if info["in_channels"] != 4:
        sys.exit("checkpoint không có kênh density — tool này cho checkpoint 4 kênh của bài")
    enc = model.vision.to(a.device).eval()
    size, wnorm = dot_sizes(enc.projection)
    os.makedirs(a.out, exist_ok=True)
    summary = []
    for f in a.files:
        rel = f.partition(":")[0]
        stem = os.path.splitext(os.path.basename(rel))[0]
        iid = stem.rsplit("_", 1)[0]
        with open(os.path.join(a.samples, rel.split("/")[0], "annotation", stem + ".json")) as fh:
            cls = json.load(fh)["class"]
        if a.image == "original":
            br = sorted(glob.glob(os.path.join(a.ce130, "*", f"{iid}_b*")))
            if not br:
                sys.exit(f"không thấy nhánh {iid}_b* trong {a.ce130}")
            img = Image.open(os.path.join(br[0], "ground_truth.jpg")).convert("RGB")
        else:
            img = Image.open(os.path.join(a.samples, rel)).convert("RGB")
        rows = ["full" if a.image == "original" else "sample", "blank"]
        fig, axes = plt.subplots(2, 3, figsize=(13.5, 9.2), squeeze=False)
        for r, dens in enumerate(rows):
            x = build_inputs(a, rel, iid, img, dens)
            xy, att = enc.keypoints(x[None].to(a.device))
            xy, att = xy[0].cpu(), att[0].cpu()
            bg = x[:3].numpy().transpose(1, 2, 0)
            den = x[3].numpy()
            axes[r, 0].imshow(bg)
            axes[r, 0].set_title(f"Original ({a.image} image)", fontsize=9)
            im = axes[r, 1].imshow(den, cmap="viridis", vmin=0, vmax=1)
            axes[r, 1].set_title(f"Density Map ({dens})\nmin {den.min():.3f} max {den.max():.3f}", fontsize=9)
            fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.02)
            eff, px, py, sharp = ss_scatter(fig, axes[r, 2], bg, xy, att, size, T,
                                            label="effective cells (1 = sharp, 256 = uniform)")
            axes[r, 2].set_title(f"SpatialSoftmax Output (512 channels)\nsharp (<{SHARP} cells): {sharp * 100:.1f}% | "
                                 f"median cells {np.median(eff):.0f}", fontsize=9)
            tidy(axes[r], T, T)
            summary.append({"file": rel, "image": a.image, "class": cls, "density": dens,
                            "sharp_frac": sharp, "median_cells": float(np.median(eff)),
                            "weighted_center_xy": [float(np.average(px, weights=wnorm)), float(np.average(py, weights=wnorm))]})
        fig.suptitle(f"CE-Loc paper checkpoint | {rel} (true class '{cls}') | dot = one channel's expected (x, y), "
                     f"color = effective cells, size = projection weight", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, f"{a.image}_{stem}.png"), dpi=110)
        plt.close(fig)
        print(f"{rel:28s} | " + " | ".join(f"{s['density']:6s} sharp {s['sharp_frac'] * 100:5.1f}% median {s['median_cells']:5.0f}"
                                          for s in summary[-2:]), flush=True)
    with open(os.path.join(a.out, "spatial_softmax.json"), "w") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
