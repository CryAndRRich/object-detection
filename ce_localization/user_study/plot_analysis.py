#!/usr/bin/env python3
"""Hình cho phân tích user study theo hạng box (`analyze.py`; output/add/user_study/analysis/README.md). Chữ trên hình tiếng Anh.
Không model; đọc items + ratings (+ ảnh cho hình ví dụ). Mỗi hình ghi một PNG (220 dpi).

  fig1_distinct_boxes     số box tách biệt / ảnh: thanh ngang 100 % mỗi bộ, dải xanh theo thứ bậc 1 -> 4
  fig2_error_by_rank      lỗi theo hạng box tách biệt: cột chồng theo nhóm không trùng nhau (đúng một lý do / nhiều lý do), % trong đoạn
  fig3_boxes_by_threshold nếu NMS chỉ chạy ở MỘT ngưỡng cố định (0,3 / 0,5 / 0,7 / 1,0, không nới): tỉ lệ ảnh nhận 1 / 2 / 3 / 4 box,
                          tính lại từ 30 mẫu thô (`--boxes NAME DUMP`), không dùng nhãn chấm
  fig4_ok_by_iou          tỉ lệ box được chấm ổn theo IoU với chỗ trống GT (box không trùng GT có thật sự sai không)
  fig5_success_at_k       tỉ lệ ảnh có >= 1 box ổn trong k box tách biệt đầu (cần đưa mấy box)
  fig6_examples_ce130     (bộ đầu) 6 màn thật (a)–(f), phóng vào vùng có box: box trùng / một chỗ / box 1 ổn còn 2–4 đè vật / đủ ổn /
                          sai cỡ / vị trí không hợp lý
  fig7_examples_cocount   (bộ thứ hai) 6 màn: đủ ổn / vị trí / sai cỡ / đè vật / hai lý do một box / cả 4 không ổn

  cd object-detection/ce_localization
  U=../../output/add/user_study
  python user_study/plot_analysis.py --model gamma4 --out $U/analysis/figure \\
      --set "CE-130" $U/ce130/items.json $U/ce130/ratings.jsonl ../data/samples \\
      --set "CE-CoCount (box resize)" $U/cocount_resize/items_objsize.json $U/cocount_resize/ratings_objsize_rater3.jsonl ../data/cocount \\
      --boxes "CE-130" $U/ce130/boxes_gamma4.json --boxes "CE-CoCount (box resize)" ../../output/add/cocount/filt3/boxes_gamma4.json
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.user_study.analyze import ERRORS, dedupe  # noqa: E402
from ce_localization.user_study.app import as_reasons  # noqa: E402
from ce_localization.user_study.score import load_ratings  # noqa: E402
from ce_localization.user_study.build import to_image_px  # noqa: E402
from ce_localization.user_study.selection import MIN_SIDE, NMS_LEVELS, VOTE_IOU, _greedy, clip_to_image  # noqa: E402
from ce_localization.utils.box_ops_np import box_iou  # noqa: E402

# Màu (dataviz reference palette, đã chạy validate_palette.js trên nền trắng): 3 slot categorical đầu cho 3 lý do lỗi (aqua < 3:1
# ⇒ luôn có chú giải + nhãn số); dải xanh thứ bậc 250 / 350 / 450 / 600 (--ordinal: PASS) cho 1 -> 4 box tách biệt; trạng thái
# xanh lá / đỏ chỉ cho ổn / không ổn ở hình ví dụ, luôn kèm chữ.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
NEUTRAL = "#a3a29b"
GOOD, BAD = "#0ca30c", "#d03b3b"
OBJ = "#7fbfff"
HALO = [pe.withStroke(linewidth=3.0, foreground="#1a1a1a")]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
REASON_EN = {"on_object": "On object", "wrong_size": "Wrong size", "implausible": "Implausible location"}
DPI = 220
rng = np.random.default_rng(0)


def style():
    sns.set_theme(style="white", rc={
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 11.5,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 11, "ytick.labelsize": 10.5, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.9, "grid.linestyle": "-", "legend.frameon": False, "legend.fontsize": 10.5, "text.color": INK,
        "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0})


def load(items, ratings, model):
    with open(items) as f:
        d = json.load(f)
    rows = []
    for r in load_ratings(ratings).values():
        if r["model"] != model or r.get("repeat_of") is not None:
            continue
        sel = d["items"][r["image_id"]]["models"][model]
        kept, _ = dedupe(sel["boxes"], r["labels"])
        rows.append({"image_id": r["image_id"], "labels": r["labels"], "nms": sel["nms"], "boxes": sel["boxes"], "kept": kept})
    return d, rows


def save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.png"), dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def headline(fig, title, sub, y=0.985):
    fig.text(0.01, y, title, fontsize=14, fontweight="bold", ha="left", va="top", color=INK)
    fig.text(0.01, y - 0.075, sub, fontsize=11, ha="left", va="top", color=INK2)


def ci_bar(ax, x, p, lo, hi, label_pad, w, color, fs=11.5):
    ax.bar(x, p, width=w, color=color, zorder=3)
    ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt="none", ecolor=INK2, elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)
    ax.text(x, hi + label_pad, pct(p, 0 if p >= 0.1 or p == 0 else 1), ha="center", va="bottom", fontsize=fs,
            fontweight="bold", color=INK)


def stack_bar(ax, x, segs, bw=0.56, fit=0.035, gap=0.021, fs=8.5, y_min=0.0):
    """Một cột chồng: segs = list (giá trị, màu, màu chữ). Đoạn >= fit ghi % giữa đoạn; đoạn nhỏ hơn ghi % cạnh phải cột (đường dẫn
    mảnh, nhãn gần nhau tự giãn theo thứ tự từ dưới lên). -> đỉnh cột và đỉnh nhãn bên cạnh cao nhất."""
    bottom, side = 0.0, []
    for v, c, ink in segs:
        ax.bar(x, v, width=bw, bottom=bottom, color=c, edgecolor="white", linewidth=1.0, zorder=3)
        if v >= fit:
            ax.text(x, bottom + v / 2, pct(v), ha="center", va="center", fontsize=fs, color=ink, zorder=4)
        elif v > 0:
            side.append((bottom + v / 2, v))
        bottom += v
    y_prev = -1.0
    for yc, v in side:
        y = max(yc, y_prev + gap, y_min)                  # y_min: nhãn không nằm sát đường trục
        x0, x1 = x + bw / 2, x + bw / 2 + 0.1
        ax.plot([x0, x0 + 0.04, x1 - 0.01], [yc, y, y], color=MUTED, lw=0.7, zorder=4, clip_on=False)
        ax.text(x1, y, pct(v), ha="left", va="center", fontsize=fs - 0.5, color=INK2, zorder=4)
        y_prev = y
    return bottom, y_prev


# ---------------------------------------------------------------- fig 1
def fig_distinct(sets, out):
    fig, ax = plt.subplots(figsize=(8.2, 1.25 + 0.75 * len(sets)))
    fig.subplots_adjust(left=0.27, right=0.98, top=0.62, bottom=0.12)
    ax.grid(False)
    for si, (name, _, rows) in enumerate(sets):
        share = np.bincount([len(r["kept"]) for r in rows], minlength=5)[1:] / len(rows)
        y, left = len(sets) - 1 - si, 0.0
        for k, v in enumerate(share):
            if v > 0:
                ax.barh(y, v, left=left, height=0.62, color=RAMP[k], edgecolor="white", linewidth=1.5, zorder=3)
                if v >= 0.05:
                    ax.text(left + v / 2, y, pct(v), ha="center", va="center", fontsize=11 if v >= 0.09 else 9,
                            fontweight="bold", color="white" if k >= 1 else INK)
                left += v
        ax.text(-0.01, y, f"{name}\n", ha="right", va="center", fontsize=11.5, fontweight="bold", color=INK,
                transform=ax.get_yaxis_transform())
        ax.text(-0.01, y - 0.14, f"{len(rows):,} images", ha="right", va="center", fontsize=10, color=MUTED,
                transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(sets) - 0.5)
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", labelsize=9.5, labelcolor=MUTED)
    handles = [Patch(color=c, label=f"{k} distinct box{'es' if k > 1 else ''}") for k, c in enumerate(RAMP, 1)]
    ax.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(-0.36, 1.02), fontsize=10.5, handlelength=1.1,
              columnspacing=1.6)
    headline(fig, "How many of the 4 shown boxes are actually distinct",
             'Share of images. Boxes rated "Other" (duplicate) are merged into the lower-numbered box.', y=0.99)
    for si, (_, _, rows) in enumerate(sets):                # 1.3 % quá hẹp để ghi bên trong: ghi cạnh thanh
        share = np.bincount([len(r["kept"]) for r in rows], minlength=5)[1:] / len(rows)
        if 0 < share[0] < 0.05:
            ax.annotate(pct(share[0]), xy=(share[0] / 2, len(sets) - 1 - si + 0.31), xytext=(share[0] / 2, len(sets) - 1 - si + 0.48),
                        fontsize=9.5, color=INK2, ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    save(fig, out, "fig1_distinct_boxes")


# ---------------------------------------------------------------- fig 2
def fig_error_by_rank(sets, out):
    """Cột chồng 100 % theo nhóm KHÔNG trùng nhau (đúng một lý do / hai lý do trở lên) ⇒ mỗi box lỗi nằm trong đúng một đoạn,
    tổng chiều cao = tỉ lệ lỗi. % ghi trong đoạn đủ cao; đoạn quá thấp ghi cạnh phải cột (đường dẫn mảnh); % tổng trên đầu cột."""
    cats = [("only", e, REASON_EN[e] + " only", SERIES[i]) for i, e in enumerate(ERRORS)] + \
           [("multi", None, "Two or more reasons", NEUTRAL)]
    ink_on = {SERIES[0]: "white", SERIES[1]: "white", SERIES[2]: INK, NEUTRAL: INK}   # chữ trong đoạn theo độ sáng nền
    fig, axes = plt.subplots(1, len(sets), figsize=(10, 4.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.70, bottom=0.16, wspace=0.10)
    axes = np.atleast_1d(axes)
    for ax, (name, _, rows) in zip(axes, sets):
        for rank in range(1, 5):
            errs = [set(r["kept"][rank - 1][1]) & set(ERRORS) for r in rows if len(r["kept"]) >= rank]
            segs = [(np.mean([x == {e} for x in errs]) if kind == "only" else np.mean([len(x) > 1 for x in errs]), c, ink_on[c])
                    for kind, e, _, c in cats]
            bottom, y_prev = stack_bar(ax, rank, segs)
            ax.text(rank, max(bottom, y_prev) + 0.012, pct(bottom), ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                    color=INK)
            ax.text(rank, -0.11, f"n = {len(errs):,}", ha="center", va="top", fontsize=9, color=MUTED,
                    transform=ax.get_xaxis_transform())
        ax.set_xticks(range(1, 5), [f"Box {k}" for k in range(1, 5)])
        ax.tick_params(axis="x", pad=5)
        ax.set_title(name, fontsize=12.5, pad=6)
        ax.set_xlim(0.45, 4.55)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylabel("Boxes rated not OK")
    axes[0].set_ylim(0, 0.45)
    handles = [Patch(color=c, label=lab) for _, _, lab, c in cats[::-1]]
    fig.legend(handles=handles, loc="upper left", ncol=4, bbox_to_anchor=(0.005, 0.845), fontsize=10.5, handlelength=1.1,
               columnspacing=1.8)
    headline(fig, "Error rate of each distinct box, by rank (GAMMA4)",
             "Box k = k-th distinct box (duplicates merged). Each not-OK box counted once; bar height = total error rate.")
    save(fig, out, "fig2_error_by_rank")


# ---------------------------------------------------------------- fig 3 / 4
def boxes_at(boxes, wh, thr, k=4):
    """Số box NMS nhận được nếu CHỈ chạy ở ngưỡng `thr` (không nới), tối đa k — cùng các bước `selection.select_boxes`
    (kẹp vào ảnh, bỏ box cạnh < MIN_SIDE, phiếu IoU >= VOTE_IOU, NMS tham lam theo phiếu)."""
    b = clip_to_image(boxes, wh)
    v = np.flatnonzero((b[:, 2] - b[:, 0] >= MIN_SIDE) & (b[:, 3] - b[:, 1] >= MIN_SIDE))
    if not len(v):
        return 0
    iou = box_iou(b[v], b[v])[0]
    return len(_greedy(np.argsort(-(iou >= VOTE_IOU).sum(1), kind="stable"), iou, thr, [], k))


def fig_boxes_by_threshold(panels, out):
    """panels: list (tên, items.json đã nạp, dump box thô). Mỗi ngưỡng NMS cố định một cột chồng 100 %: tỉ lệ ảnh nhận được
    1 / 2 / 3 / 4 box (dải màu như fig 1). Tính từ 30 mẫu của model, không dùng nhãn chấm."""
    fig, axes = plt.subplots(1, len(panels), figsize=(10, 4.9), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.70, bottom=0.17, wspace=0.12)
    axes = np.atleast_1d(axes)
    for ax, (name, d, dump, model) in zip(axes, panels):
        recs = {r["image_id"]: r for r in dump["results"][d["models"][model]["key"]]}
        for i, t in enumerate(NMS_LEVELS):
            n = [boxes_at(to_image_px(recs[iid]["boxes"], recs[iid]["wh"], it["wh"]), it["wh"], t) for iid, it in d["items"].items()]
            share = np.bincount(np.clip(n, 0, 4), minlength=5)[1:] / len(n)
            stack_bar(ax, i, [(v, RAMP[k], INK if k == 0 else "white") for k, v in enumerate(share)], bw=0.56, fit=0.06,
                      gap=0.045, y_min=0.03)
        ax.set_xticks(range(len(NMS_LEVELS)), [f"{t:.1f}" for t in NMS_LEVELS])
        ax.set_xlabel("NMS IoU threshold (fixed, no relaxing)", labelpad=8)
        ax.set_title(f"{name}, {len(d['items']):,} images", fontsize=12.5, pad=6)
        ax.set_xlim(-0.5, len(NMS_LEVELS) - 0.3)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Share of images")
    handles = [Patch(color=c, label=f"{k} box{'es' if k > 1 else ''}") for k, c in enumerate(RAMP, 1)][::-1]
    fig.legend(handles=handles, loc="upper left", ncol=4, bbox_to_anchor=(0.005, 0.845), fontsize=10.5, handlelength=1.1,
               columnspacing=1.8)
    headline(fig, "How many boxes NMS keeps at a fixed threshold (GAMMA4)",
             "Boxes kept out of 4 per image, from the model's 30 samples (no ratings). The app used 0.3 and relaxed only to fill up 4.")
    save(fig, out, "fig3_boxes_by_threshold")


IOU_BINS = ("0", "0–0.1", "0.1–0.3", "0.3–0.5", "≥ 0.5")


def iou_bin(iou):
    """IoU -> nhãn khoảng: đúng 0 | (0; 0,1) | [0,1; 0,3) | [0,3; 0,5) | >= 0,5."""
    return IOU_BINS[0 if iou <= 0 else 1 if iou < 0.1 else 2 if iou < 0.3 else 3 if iou < 0.5 else 4]


def fig_ok_by_iou(sets, out):
    """Tỉ lệ box (giữ, sau gộp) được chấm ổn theo IoU lớn nhất với chỗ trống GT (CE-130: lỗ inpaint; CE-CoCount: 10 loc_bbox),
    mỗi bộ một panel — box không trùng GT có thật sự sai không."""
    fig, axes = plt.subplots(1, len(sets), figsize=(10, 4.7), sharey=True)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.76, bottom=0.2, wspace=0.10)
    axes = np.atleast_1d(axes)
    for ax, (name, d, rows) in zip(axes, sets):
        by = {lab: [] for lab in IOU_BINS}
        for r in rows:
            it = d["items"][r["image_id"]]
            holes = np.asarray(it["holes"], float).reshape(-1, 4)
            for i, reasons, _ in r["kept"]:
                iou = float(box_iou(np.asarray([r["boxes"][i]], float), holes)[0].max()) if len(holes) else 0.0
                by[iou_bin(iou)].append(not (set(reasons) & set(ERRORS)))
        for i, lab in enumerate(IOU_BINS):
            v = by[lab]
            p_ = float(np.mean(v)) if v else np.nan
            ax.bar(i, p_, width=0.6, color=SERIES[0], zorder=3)
            ax.text(i, p_ + 0.015, pct(p_, 0), ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=INK)
            ax.text(i, -0.11, f"{len(v):,} boxes", ha="center", va="top", fontsize=9, color=MUTED, transform=ax.get_xaxis_transform())
        ax.set_xticks(range(len(IOU_BINS)), IOU_BINS)
        ax.tick_params(axis="x", pad=5)
        ax.set_xlabel("IoU with the nearest ground-truth empty spot", labelpad=24)
        ax.set_title(name, fontsize=12.5, pad=6)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Boxes rated OK")
    headline(fig, "Boxes that miss the ground-truth spot are often still OK (GAMMA4)",
             "Share of distinct boxes rated OK. GT spot: inpainted holes (CE-130), the 10 annotated spots (CE-CoCount).")
    save(fig, out, "fig4_ok_by_iou")


def fig_success_at_k(sets, out):
    """Tỉ lệ ảnh có >= 1 box ổn trong k box tách biệt đầu (k = 1..4), mỗi bộ một đường."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.11, right=0.76, top=0.76, bottom=0.14)
    ks = range(1, 5)
    for si, (name, _, rows) in enumerate(sets):
        y = [np.mean([any(not (set(x) & set(ERRORS)) for _, x, _ in r["kept"][:k]) for r in rows]) for k in ks]
        ax.plot(ks, y, color=SERIES[si], lw=2.2, marker="o", ms=8, mec="white", mew=2, zorder=3)
        for k, v in zip(ks, y):
            ax.text(k, v + (0.008 if si == 0 else -0.008), pct(v), ha="center", va="bottom" if si == 0 else "top", fontsize=9.5,
                    color=INK2)
        ax.text(4.42, y[-1], f"{name}\n{len(rows):,} images", ha="left", va="center", fontsize=10, color=INK,
                fontweight="bold", clip_on=False)
    ax.set_xticks(list(ks), [f"First {k}" if k > 1 else "Box 1 only" for k in ks])
    ax.set_xlim(0.6, 4.35)
    ax.set_ylim(0.8, 1.0)
    ax.set_yticks(np.arange(0.8, 1.001, 0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylabel("Images with ≥ 1 OK box")
    ax.set_xlabel("Distinct boxes shown to the user", labelpad=6)
    headline(fig, "One box is usually enough (GAMMA4)",
             "Share of images where at least one of the first k distinct boxes is rated OK.")
    save(fig, out, "fig5_success_at_k")


# ---------------------------------------------------------------- fig 5
def _pick(rows, d, want, n_obj=(8, 40), item_ok=lambda it: True, frac=1 / 3):
    """Màn thoả điều kiện, chọn cố định ở vị trí `frac` của danh sách theo image_id — không chọn tay."""
    cands = [r for r in rows if n_obj[0] <= len(d["items"][r["image_id"]]["objects"]) <= n_obj[1]
             and item_ok(d["items"][r["image_id"]]) and want(r)]
    return sorted(cands, key=lambda r: r["image_id"])[min(int(len(cands) * frac), len(cands) - 1)] if cands else None


def _crop(boxes, W, H, aspect=4 / 3, min_frac=0.5):
    """Khung nhìn quanh các box (rộng >= min_frac ảnh, tỉ lệ `aspect`, nằm trong ảnh)."""
    b = np.asarray(boxes, float)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    w, h = (x2 - x1) * 1.5, (y2 - y1) * 1.5
    w = max(w, h * aspect, min_frac * W)
    h = min(w / aspect, H, W / aspect)                   # khung luôn đúng tỉ lệ `aspect` và nằm trong ảnh
    w = h * aspect
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    l, t = np.clip(cx - w / 2, 0, W - w), np.clip(cy - h / 2, 0, H - h)
    return l, t, w, h


def _place(ax, txt, placed, x_right=None):
    """Nhãn tràn mép phải khung nhìn -> căn phải theo `x_right`; chạm nhãn đã đặt -> dời xuống dưới tới khi không chạm."""
    r_ = ax.figure.canvas.get_renderer()
    if x_right is not None and txt.get_window_extent(r_).x1 > ax.get_window_extent(r_).x1:
        txt.set_ha("right")
        txt.set_position((min(x_right, ax.get_xlim()[1]), txt.get_position()[1]))
    for _ in range(8):
        bb = txt.get_window_extent(r_).expanded(1.02, 1.08)
        if not any(bb.overlaps(o) for o in placed):
            placed.append(bb)
            return
        x, y = txt.get_position()
        dy = ax.transData.inverted().transform((0, 0))[1] - ax.transData.inverted().transform((0, bb.height))[1]
        txt.set_position((x, y + abs(dy)))
    placed.append(txt.get_window_extent(r_))


def _draw(ax, img_path, it, r, title):
    ax.imshow(Image.open(img_path).convert("RGB"))
    ax.set_axis_off()
    W, H = it["wh"]
    l, t, w, h = _crop(r["boxes"], W, H)
    ax.set_xlim(l, l + w)
    ax.set_ylim(t + h, t)
    ax.set_title(title, fontsize=11.5, loc="left", fontweight="bold", pad=6)
    for x1, y1, x2, y2 in it["objects"]:
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=OBJ, lw=0.8, alpha=0.55, zorder=2))
    kept = {i: reasons for i, reasons, _ in r["kept"]}
    pad = 0.012 * h
    placed = []
    for i in [i for i in range(len(r["boxes"])) if i not in kept] + [i for i in range(len(r["boxes"])) if i in kept]:
        x1, y1, x2, y2 = r["boxes"][i]
        if i in kept:
            bad = set(kept[i]) & set(ERRORS)
            c = BAD if bad else GOOD
            txt = f"{i + 1}  " + (", ".join(REASON_EN[e].lower() for e in ERRORS if e in bad) if bad else "OK")
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=2.8, zorder=4))
            above = y1 - pad > t + 0.08 * h
            _place(ax, ax.text(x1, y1 - pad if above else y2 + pad, txt, fontsize=10, fontweight="bold", color="white", zorder=6,
                               va="bottom" if above else "top",
                               bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc=c, ec="none")), placed, x2)
        else:
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.8, ls=(0, (4, 3)), zorder=3,
                                   path_effects=HALO))
    dups = [i for i in range(len(r["boxes"])) if i not in kept]
    if dups:                                              # một nhãn chung cho các box trùng, dưới box trùng thấp nhất
        b = np.asarray([r["boxes"][i] for i in dups])
        _place(ax, ax.text(b[:, 2].max(), b[:, 3].max() + pad, ", ".join(str(i + 1) for i in dups) + "  duplicate",
                           fontsize=9.5, color=INK, va="top", ha="right", zorder=5,
                           bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc="white", ec="none", alpha=0.92)), placed)


def _cases(kind):
    """6 trường hợp (tiêu đề, điều kiện trên màn, n_obj, điều kiện trên item) cho mỗi loại bộ."""
    ok = lambda x: not (set(x) & set(ERRORS))  # noqa: E731
    errs = lambda r: [set(x) & set(ERRORS) for _, x, _ in r["kept"]]  # noqa: E731
    has = lambda r, e: any(e in x for x in errs(r))  # noqa: E731
    clean = lambda r: not any(amb for _, _, amb in r["kept"])  # noqa: E731
    dup = lambda r: any("other" in as_reasons(lab) for lab in r["labels"])  # noqa: E731
    landscape = lambda it: it["wh"][0] >= it["wh"][1]  # noqa: E731
    big = lambda it: landscape(it) and np.median([np.sqrt((o[2] - o[0]) * (o[3] - o[1])) for o in it["objects"]]) \
        >= 0.05 * max(it["wh"])  # noqa: E731  ảnh ngang, vật đủ to để nhìn thấy box

    def kept_boxes(r):
        return np.asarray([r["boxes"][i] for i, _, _ in r["kept"]], float)

    def tight_dups(r, thr=0.7):                           # mọi box trùng chồng gần khít (IoU >= thr) lên một box giữ
        drop = [i for i in range(len(r["boxes"])) if i not in {k for k, _, _ in r["kept"]}]
        return bool(drop) and box_iou(np.asarray([r["boxes"][i] for i in drop], float), kept_boxes(r))[0].max(1).min() >= thr

    def separated(r, thr=0.05):                           # các box giữ tách hẳn nhau
        iou = box_iou(kept_boxes(r), kept_boxes(r))[0]
        np.fill_diagonal(iou, 0)
        return iou.max() <= thr
    if kind == "ce130":
        return [
            ("4 boxes shown, 2 distinct (duplicates merged)",
             lambda r: len(r["kept"]) == 2 and clean(r) and dup(r) and all(ok(x) for _, x, _ in r["kept"]) and tight_dups(r),
             (8, 40), landscape),
            ("All 4 boxes at one place, rated OK",
             lambda r: len(r["kept"]) == 1 and clean(r) and ok(r["kept"][0][1]), (5, 60), landscape),
            ("Box 1 OK, boxes 2–4 on an object",
             lambda r: len(r["kept"]) == 4 and ok(r["kept"][0][1]) and all(x == {"on_object"} for x in errs(r)[1:]),
             (8, 40), landscape),
            ("All 4 distinct boxes OK", lambda r: len(r["kept"]) == 4 and all(ok(x) for _, x, _ in r["kept"]) and separated(r),
             (8, 40), landscape),
            ("Wrong size", lambda r: len(r["kept"]) == 4 and sum(x == {"wrong_size"} for x in errs(r)) >= 2, (8, 40), landscape),
            ("Implausible location", lambda r: len(r["kept"]) == 4 and sum(x == {"implausible"} for x in errs(r)) >= 2,
             (8, 40), landscape)]
    return [
        ("All 4 boxes OK", lambda r: all(ok(x) for _, x, _ in r["kept"]), (5, 40), big),
        ("Implausible location", lambda r: sum(x == {"implausible"} for x in errs(r)) >= 2
         and sum(ok(x) for _, x, _ in r["kept"]) >= 1, (5, 60), landscape),
        ("Wrong size", lambda r: sum(x == {"wrong_size"} for x in errs(r)) >= 2, (5, 40), big),
        ("On an object", lambda r: has(r, "on_object") and sum(ok(x) for _, x, _ in r["kept"]) >= 1, (5, 40), big),
        ("Two reasons on one box", lambda r: any(len(x) > 1 for x in errs(r)), (5, 60), landscape),
        ("All 4 boxes not OK", lambda r: all(not ok(x) for _, x, _ in r["kept"]), (5, 60), landscape)]


# màn người dùng đã chọn cho từng panel (kind -> {panel: image_id}); panel không có ở đây chọn tự động (1/3 danh sách ứng viên).
# Chọn từ bảng ứng viên: `--candidates ce130:a`; ghi đè tạm: `--pick ce130:a=<image_id>`.
PICK = {"ce130": {"a": "2162_b1_t1", "d": "6962_b2_t1", "b": "3657_b2_t1", "c": "4940_b2_t1", "e": "3339_b3_t1", "f": "3658_b3_t3"},
        "cocount": {"a": "INTER_OTR_NUT0_PEG0_00014_00011_0_185_negative", "b": "INTRA_FUN_CHK1_CHK2_00058_00059_0_340_negative", "c": "INTER_OTR_IKE0_PEG0_00015_00020_0_75_positive",
                    "d": "INTRA_FUN_MAH1_MAH2_00013_00010_0_340_negative", "e": "INTER_HOU_ULT0_CTB0_00043_00041_0_160_positive",
                    "f": "INTER_OTR_BOL0_NUT0_00044_00046_0_340_positive"}}


def candidates(d, rows, root, kind, panel, out, n=9):
    """Bảng n ứng viên (trải đều danh sách) của một panel -> `_candidates_<kind>_<panel>.png` để người dùng chọn."""
    k = "abcdef".index(panel)
    title, want, n_obj, item_ok = _cases(kind)[k]
    cands = sorted([r for r in rows if n_obj[0] <= len(d["items"][r["image_id"]]["objects"]) <= n_obj[1]
                    and item_ok(d["items"][r["image_id"]]) and want(r)], key=lambda r: r["image_id"])
    sel = [cands[int(i)] for i in np.linspace(0, len(cands) - 1, min(n, len(cands)))] if cands else []
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 12.4))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95, bottom=0.005, wspace=0.03, hspace=0.12)
    for j, ax in enumerate(axes.flat):
        if j >= len(sel):
            ax.set_axis_off()
            continue
        it = d["items"][sel[j]["image_id"]]
        _draw(ax, os.path.join(root, it["image"]), it, sel[j], f"#{j + 1}  {sel[j]['image_id']}")
    fig.text(0.005, 0.995, f"Candidates for {kind} ({panel}) {title}: {len(cands)} screens match, {len(sel)} shown",
             fontsize=13, fontweight="bold", va="top")
    path = f"_candidates_{kind}_{panel}"
    save(fig, out, path)
    return path + ".png", [r["image_id"] for r in sel]


def fig_examples(name, d, rows, root, kind, out, fname, pick=None):
    """Lưới 2 × 3: 6 màn thật (a)–(f), mỗi màn một trường hợp, ảnh không trùng nhau; không tiêu đề chung, không chú giải."""
    pick = {**PICK.get(kind, {}), **(pick or {})}
    by_id = {r["image_id"]: r for r in rows}
    used = set(pick.values())
    panels = []
    for k, (title, want, n_obj, item_ok) in enumerate(_cases(kind)):
        fixed = pick.get("abcdef"[k])
        r = by_id.get(fixed) if fixed else _pick([x for x in rows if x["image_id"] not in used], d, want, n_obj=n_obj,
                                                  item_ok=item_ok)
        if r is not None:
            used.add(r["image_id"])
            panels.append((r, title))
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005, wspace=0.03, hspace=0.12)
    for k, ax in enumerate(axes.flat):
        if k >= len(panels):
            ax.set_axis_off()
            continue
        r, title = panels[k]
        it = d["items"][r["image_id"]]
        _draw(ax, os.path.join(root, it["image"]), it, r, f"({'abcdef'[k]}) {title}")
    save(fig, out, fname)
    return [r["image_id"] for r, _ in panels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", nargs=4, action="append", required=True, metavar=("NAME", "ITEMS", "RATINGS", "IMAGE_ROOT"))
    ap.add_argument("--model", default="gamma4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boxes", nargs=2, action="append", default=[], metavar=("NAME", "DUMP"),
                    help="fig 3: file --dump-boxes (30 mẫu thô) của bộ NAME — khoá lấy từ items.json")
    ap.add_argument("--candidates", nargs="+", default=None, metavar="KIND:PANEL",
                    help="chỉ vẽ bảng ứng viên cho panel (vd ce130:a cocount:b) vào --out rồi dừng")
    ap.add_argument("--pick", nargs="+", default=[], metavar="KIND:PANEL=IMAGE_ID", help="ghi đè màn của panel (ngoài PICK)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    loaded = [(name, *load(items, rat, a.model), root) for name, items, rat, root in a.set]
    sets = [(name, d, rows) for name, d, rows, _ in loaded]
    kinds = (("ce130", "fig6_examples_ce130"), ("cocount", "fig7_examples_cocount"))
    if a.candidates:
        for spec in a.candidates:
            kind, panel = spec.split(":")
            name, d, rows, root = loaded[[k for k, _ in kinds].index(kind)]
            print(spec, *candidates(d, rows, root, kind, panel, a.out))
        return
    picks = {}
    for spec in a.pick:
        kp, iid = spec.split("=")
        kind, panel = kp.split(":")
        picks.setdefault(kind, {})[panel] = iid
    fig_distinct(sets, a.out)
    fig_error_by_rank(sets, a.out)
    if a.boxes:
        dumps = dict((n, p) for n, p in a.boxes)
        panels = []
        for name, d, _ in sets:
            if name in dumps:
                with open(dumps[name]) as f:
                    panels.append((name, d, json.load(f), a.model))
        fig_boxes_by_threshold(panels, a.out)
    fig_ok_by_iou(sets, a.out)
    fig_success_at_k(sets, a.out)
    for (name, d, rows, root), (kind, fname) in zip(loaded, kinds):
        print("ví dụ", name, fig_examples(name, d, rows, root, kind, a.out, fname, picks.get(kind)))
    print(f"-> {a.out}: {sorted(os.listdir(a.out))}")


if __name__ == "__main__":
    main()
