#!/usr/bin/env python3
"""Hình minh hoạ lọc box vật SAM của CE-CoCount (`data/cocount.filter_objects`, docs/EXPERIMENT_GAMMA.md mục 18). Không model, chỉ đọc
ảnh + JSON. Bảng 2 hàng × 4 cột: hàng trên = trước lọc (mọi box SAM: xanh = giữ, cam = bị bỏ vì cạnh > K × cạnh TB exemplar), hàng dưới
= sau lọc (box giữ + 3 exemplar gán tay, xanh lá). Mỗi cột một mẫu: `--names`, mặc định `NAMES` (người dùng chọn 2026-10-10); `--candidates` vẽ bảng ứng viên.

  cd object-detection/ce_localization
  python tools/plot_cocount_filter.py --cocount-root ../data/cocount --out ../../output/add/cocount/figure/sam_filter.png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import read_cocount  # noqa: E402

KEEP, DROP, EXEMPLAR = "#2a78d6", "#eb6834", "#0ca30c"
INK = "#0b0b0b"


def side(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None))


def stats(root, name, k):
    """-> (box SAM gốc, mask giữ, exemplar, tỉ lệ cạnh box / cạnh TB exemplar) của mẫu `name`."""
    raw = read_cocount(root, name)
    ex = raw["exemplars"]
    ratio = side(raw["objects"]) / side(ex).mean() if len(ex) else np.ones(len(raw["objects"]))
    return raw["objects"], ratio <= k, ex, ratio


def candidates(root, k, kind, n_obj=(10, 60), min_side=0.03):
    """Danh sách mẫu theo kiểu (0 = SAM hỏng gần hết, 1 = vài box hỏng, 2 = không box hỏng): ảnh ngang ≥ 1,5, n_obj vật, cạnh vật
    giữ lại (trung vị) ≥ min_side × cạnh dài ảnh — vật đủ to để nhìn rõ box."""
    out = []
    for n in sorted(f[:-5] for f in os.listdir(os.path.join(root, "Anno")) if f.endswith(".json")):
        objs, keep, _, _ = stats(root, n, k)
        if not n_obj[0] <= len(objs) <= n_obj[1] or not keep.any():
            continue
        with Image.open(os.path.join(root, "Image", n + ".jpg")) as im:
            W, H = im.size
        if W < 1.5 * H or np.median(side(objs[keep])) < min_side * W:
            continue
        drop = 1 - keep.mean()
        if (kind == 0 and drop >= 0.8) or (kind == 1 and 0.05 <= drop <= 0.3) or (kind == 2 and drop == 0):
            out.append(n)
    return out


# 4 mẫu người dùng chọn (2026-10-10): (a) SAM hỏng gần hết, (b) vài box hỏng, (c) SAM hỏng gần hết cảnh khác, (d) không box hỏng.
# Chọn lại từ bảng ứng viên: `--candidates 0|1|2`.
NAMES = ["INTER_HOU_ULT0_CTB0_00043_00041_0_300_negative", "INTER_OFF_RUB0_PPC0_00268_00166_0_1234_negative",
         "INTRA_HOU_CTB1_CTB2_00095_00068_0_465_negative", "INTRA_FUN_PKC1_PKC2_00025_00051_0_20_negative"]


def draw_boxes(ax, boxes, color, lw, alpha=1.0, ls="-"):
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw, alpha=alpha, ls=ls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--k", type=float, default=3.0, help="ngưỡng: bỏ box có cạnh > K × cạnh TB exemplar (như --cocount-obj-filter)")
    ap.add_argument("--names", nargs="*", default=None, help="tên mẫu, mỗi mẫu một cột (mặc định `default_names`)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--candidates", type=int, default=None, choices=[0, 1, 2],
                    help="chỉ vẽ bảng ứng viên của một kiểu (0 hỏng gần hết / 1 vài box hỏng / 2 không hỏng) vào --out rồi dừng")
    a = ap.parse_args()
    if a.candidates is not None:
        c = candidates(a.cocount_root, a.k, a.candidates)
        sel = [c[int(i)] for i in np.linspace(0, len(c) - 1, min(8, len(c)))] if c else []
        fig, axes = plt.subplots(2, 4, figsize=(18.4, 2 * 4.6 * 9 / 16 + 1.2), gridspec_kw={"wspace": 0.04, "hspace": 0.2})
        for j, ax in enumerate(axes.flat):
            ax.set_axis_off()
            if j >= len(sel):
                continue
            objs, keep, ex, _ = stats(a.cocount_root, sel[j], a.k)
            ax.imshow(Image.open(os.path.join(a.cocount_root, "Image", sel[j] + ".jpg")).convert("RGB"))
            draw_boxes(ax, objs[keep], KEEP, 0.9)
            draw_boxes(ax, objs[~keep], DROP, 1.2, alpha=0.85)
            draw_boxes(ax, ex, EXEMPLAR, 2.0)
            ax.set_title(f"#{j + 1}  {len(objs)} boxes, {(~keep).sum()} removed\n{sel[j]}", loc="left", fontsize=8.5)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"{len(c)} ứng viên, vẽ {len(sel)}:", sel)
        return
    names = a.names or NAMES
    plt.rcParams.update({"font.family": "sans-serif"})
    fig, axes = plt.subplots(2, len(names), figsize=(4.6 * len(names), 2 * 4.6 * 9 / 16 + 0.9),   # ảnh ~16:9: khít hai hàng
                             gridspec_kw={"wspace": 0.04, "hspace": 0.14})
    axes = np.asarray(axes).reshape(2, -1)
    for col, n in enumerate(names):
        objs, keep, ex, _ = stats(a.cocount_root, n, a.k)
        img = Image.open(os.path.join(a.cocount_root, "Image", n + ".jpg")).convert("RGB")
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            ax.imshow(img)
            ax.set_axis_off()
        draw_boxes(top, objs[keep], KEEP, 0.9)
        draw_boxes(top, objs[~keep], DROP, 1.2, alpha=0.85)
        top.set_title(f"({'abcdefgh'[col]}) Before: {len(objs)} SAM boxes, {(~keep).sum()} too large", loc="left", fontsize=11,
                      fontweight="bold", color=INK)
        draw_boxes(bot, objs[keep], KEEP, 0.9)
        draw_boxes(bot, ex, EXEMPLAR, 2.4)
        bot.set_title(f"After: {keep.sum()} boxes kept", loc="left", fontsize=11, fontweight="bold", color=INK)
    handles = [Line2D([], [], color=KEEP, lw=2, label="SAM box kept"),
               Line2D([], [], color=DROP, lw=2, label=f"SAM box removed (side > {a.k:g}× mean exemplar side)"),
               Line2D([], [], color=EXEMPLAR, lw=3, label="Hand-annotated exemplar (3 per image)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=11.5, frameon=False,
               bbox_to_anchor=(0.5, min(ax.get_position().y0 for ax in axes.flat) - 0.01))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    print("mẫu:", names, "->", a.out)


if __name__ == "__main__":
    main()
