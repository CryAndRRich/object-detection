#!/usr/bin/env python3
"""Vẽ box CE-Loc TRƯỚC / SAU lọc thủ công (web rà box, `output/add/review/<run>/pred_<tập>_celoc.jsonl`) — KHÔNG model, KHÔNG GPU.

Mỗi ảnh một hình 2 panel, cả hai hiện đủ K box model viền đỏ nhạt; viền đỏ đậm: trái = 5 box chọn sẵn của web rà (C-NLL thấp nhất
sau NMS IoU 0,5 — `pre` trong file, thiếu thì tính lại bằng `box_review.sources.preselect`), phải = box người rà giữ lại. Xanh dương = vật cùng lớp đang có, xanh lá nét đứt = lỗ GT (CE-130 `holes[:t]`, CoCount `loc_bbox`).
Tiêu đề panel ghi min C-NLL (lấy thẳng từ file rà, `cnll`); chữ trong hình chỉ ký tự ASCII.
Xếp ảnh theo mức tăng C-NLL TB-min khi lọc (min box giữ − min K box), chỉ ảnh có >= 1 box giữ.

  cd object-detection/ce_localization
  python tools/plot_review_filter.py --dataset ce130 --top 1 --out ../../output/add/viz/review_filter/ce130_test_all
  python tools/plot_review_filter.py --dataset cocount --ids INTER_... --out ../../output/add/viz/review_filter/cocount_review5
-> <hạng>_<ảnh>.png + rank.json (mọi ảnh: before / after / tăng / số box giữ).
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # object-detection/
sys.path.insert(0, ROOT)

from ce_localization.box_review.sources import preselect  # noqa: E402
from ce_localization.utils.plot import draw_boxes  # noqa: E402
from ce_localization.data.cocount import load_review, read_cocount  # noqa: E402
from ce_localization.data.turns import TurnIndex  # noqa: E402

TOP = os.path.dirname(ROOT)                                                               # multi_condition/
C_OBJ, C_HOLE, C_BOX = "#1e90ff", "#00c853", "#ff1a1a"


def rank_rows(rows):
    out = []
    for r in rows:
        if r.get("cnll") is None or not r.get("selected"):
            continue
        c = np.asarray(r["cnll"], dtype=np.float64)
        sel = list(r["selected"])
        b, a = int(np.argmin(c)), sel[int(np.argmin(c[sel]))]
        out.append({"id": r["id"], "before": float(c[b]), "after": float(c[a]), "gain": float(c[a] - c[b]), "best": b, "best_sel": a,
                    "n_sel": len(sel), "n_obj": len(r["objects"]), "cnll_gt": r["cnll_gt"]})
    return sorted(out, key=lambda x: -x["gain"])


def holes_of(r, dataset, index, cocount_root, review):
    if dataset == "ce130":
        e = index.turns[r["id"]]
        return np.asarray(index.branches[e["branch"]]["holes"][:e["t"]], dtype=np.float64).reshape(-1, 4)
    return np.asarray(read_cocount(cocount_root, r["id"], review)["loc"], dtype=np.float64).reshape(-1, 4)


def plot(r, k, holes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    im = Image.open(os.path.join(TOP, r["image"])).convert("RGB")
    W, H = im.size
    boxes = np.asarray(r["model"], dtype=np.float64)
    sel = list(r["selected"])
    pre = r["pre"] if r.get("pre") is not None else preselect(r["model"], r["cnll"], hidden=set(r.get("hidden_dup") or []))
    panels = [(f"Before Filtering - {len(boxes)} boxes", sorted(pre), k["before"]),
              (f"After Filtering - {len(sel)} kept", sel, k["after"])]
    pw = 6.0
    fig, axes = plt.subplots(1, 2, figsize=(2 * pw, pw * H / W + 0.9))
    for ax, (title, bold, val) in zip(axes, panels):
        ax.imshow(im)
        draw_boxes(ax, r["objects"], C_OBJ, 0.8, alpha=0.8)
        draw_boxes(ax, holes, C_HOLE, 1.2, ls="--")
        draw_boxes(ax, boxes[[i for i in range(len(boxes)) if i not in bold]], C_BOX, 0.6, alpha=0.45)
        draw_boxes(ax, boxes[bold], C_BOX, 2.0)
        ax.set_title(f"{title} - CNLL: {val:.2f}", fontsize=10)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.axis("off")
    fig.legend(handles=[Line2D([], [], color=C_OBJ, lw=1, label="existing objects"),
                        Line2D([], [], color=C_HOLE, lw=1.2, ls="--", label="GT hole"),
                        Line2D([], [], color=C_BOX, lw=0.8, alpha=0.6, label="CE-Loc boxes"),
                        Line2D([], [], color=C_BOX, lw=2.0, label="top 5 after NMS / kept")],
               loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("ce130", "cocount"), required=True)
    ap.add_argument("--pred", default=None, help="mặc định output/add/review/delta8_ep010/pred_<tập>_celoc.jsonl")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--ids", nargs="*", default=None, help="vẽ đúng các ảnh này (bỏ qua --top)")
    ap.add_argument("--turn-index", default=os.path.join(ROOT, "data/turn_index.json"))
    ap.add_argument("--cocount-root", default=os.path.join(ROOT, "data/cocount"))
    ap.add_argument("--cocount-review", default=os.path.join(ROOT, "data/cocount_review.jsonl"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pred = a.pred or os.path.join(TOP, f"output/add/review/delta8_ep010/pred_{a.dataset}_celoc.jsonl")
    rows = {r["id"]: r for r in map(json.loads, open(pred))}
    ranked = rank_rows(rows.values())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "rank.json"), "w") as f:
        json.dump(ranked, f, indent=1)
    index = TurnIndex(a.turn_index) if a.dataset == "ce130" else None
    review = load_review(a.cocount_review) if a.dataset == "cocount" else None
    pos = {k["id"]: i for i, k in enumerate(ranked)}
    pick = [ranked[pos[i]] for i in a.ids] if a.ids else ranked[:a.top]
    for k in pick:
        r = rows[k["id"]]
        path = os.path.join(a.out, f"{pos[k['id']] + 1:03d}_{k['id']}.png")
        plot(r, k, holes_of(r, a.dataset, index, a.cocount_root, review), path)
        print(f"{k['id']}: {k['before']:.2f} -> {k['after']:.2f} ({k['n_sel']} kept) -> {path}")


if __name__ == "__main__":
    main()
