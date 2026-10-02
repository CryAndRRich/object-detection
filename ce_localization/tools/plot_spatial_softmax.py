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
      --files test/images/6246_11.png:apple --out ../../output/gamma/viz/spatial_softmax_original
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.turns import image_inputs  # noqa: E402
from ce_localization.models.box_policy import BoxPolicy  # noqa: E402

T = 512
SHARP = 16                                     # < 16 ô hiệu dụng = kênh "nhọn"


def keypoint_pixels(xy, H):
    """[C,2] (dọc, ngang) trong [−1, 1] -> (x_px, y_px) trên canvas T; −1 / +1 = tâm ô đầu / cuối của lưới H."""
    cell = T / H
    to_px = lambda u: cell / 2 + (u + 1) / 2 * (T - cell)  # noqa: E731
    return to_px(xy[:, 1]), to_px(xy[:, 0])


def effective_cells(att):
    """[C,H,W] softmax -> exp(entropy) [C]."""
    p = att.reshape(att.shape[0], -1).clamp_min(1e-12)
    return torch.exp(-(p * p.log()).sum(-1))


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
    return image_inputs(img, d, T, "paper")[0]


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
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter, NullFormatter

    model, _, info = BoxPolicy.load_celoc_paper(a.ckpt)
    if info["in_channels"] != 4:
        sys.exit("checkpoint không có kênh density — tool này cho checkpoint 4 kênh của bài")
    enc = model.vision.to(a.device).eval()
    wnorm = enc.projection.weight.detach().reshape(enc.projection.out_features, -1, 2).norm(dim=(0, 2)).cpu().numpy()
    size = 4 + 60 * wnorm / wnorm.max()
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
            eff = effective_cells(att).numpy()
            px, py = keypoint_pixels(xy.numpy(), att.shape[1])
            bg = x[:3].numpy().transpose(1, 2, 0)
            den = x[3].numpy()
            axes[r, 0].imshow(bg)
            axes[r, 0].set_title(f"Original ({a.image} image)", fontsize=9)
            im = axes[r, 1].imshow(den, cmap="viridis", vmin=0, vmax=1)
            axes[r, 1].set_title(f"Density Map ({dens})\nmin {den.min():.3f} max {den.max():.3f}", fontsize=9)
            fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.02)
            axes[r, 2].imshow(bg)
            order = np.argsort(-eff)                                        # chấm nhọn vẽ sau cùng (nằm trên)
            sc = axes[r, 2].scatter(px[order], py[order], c=eff[order], s=size[order], cmap="plasma_r",
                                    norm=LogNorm(vmin=1, vmax=att.shape[1] * att.shape[2]), edgecolors="black",
                                    linewidths=0.3, alpha=0.9)
            cb = fig.colorbar(sc, ax=axes[r, 2], fraction=0.046, pad=0.02, label="effective cells (1 = sharp, 256 = uniform)")
            # nhãn trục log dạng số thường ("1", "10", "100"), KHÔNG mathtext "$10^{k}$": matplotlib cũ + pyparsing ≥ 3.3
            # (Kaggle) cảnh báo PyparsingDeprecationWarning mỗi lần dựng / parse mathtext
            cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            cb.ax.yaxis.set_minor_formatter(NullFormatter())
            sharp = float((eff < SHARP).mean())
            axes[r, 2].set_title(f"SpatialSoftmax Output (512 channels)\nsharp (<{SHARP} cells): {sharp * 100:.1f}% | "
                                 f"median cells {np.median(eff):.0f}", fontsize=9)
            for c in range(3):
                axes[r, c].set_xlim(0, T)
                axes[r, c].set_ylim(T, 0)
                axes[r, c].set_xticks([])
                axes[r, c].set_yticks([])
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
