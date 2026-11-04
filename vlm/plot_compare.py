#!/usr/bin/env python3
"""So box các VLM trên ảnh pilot: hình 2 × 3 / ảnh (chữ trong hình tiếng Anh).

  hàng 1: Original (ảnh model nhận + vật đang có xanh dương + box GT xanh lá nét đứt) | GPT-6 Luna | GLM-5.3-Flash
  hàng 2: gpt-4.1-mini | Perceptron Mk1.5 | Elorian Glimpse

Ô model chỉ vẽ box dự đoán (đỏ mảnh) + box `score` cao nhất (đỏ đậm, = box chấm chính ZETA5 / 6). Tiêu đề "<tên>, CNLL = <C-NLL
của box đó>"; Original = C-NLL thấp nhất của box GT. VLM = `output/add/vlm/<model>/<tập>/<name>.jsonl` (1 dòng / ảnh, box xếp theo
score); pilot chấm trước 2026-11-04 không có file này ⇒ đọc dump cũ `<name>_boxes.json` (K = 30; box lặp vòng được gộp lại khi vẽ). Box dump là pixel canvas 512 (letterbox góc trên trái) -> chia tỉ lệ về pixel ảnh gốc. Chỉ đọc JSON + ảnh.

  cd object-detection/vlm
  python plot_compare.py --dataset ce130 --n 1                 # thử 1 ảnh
  python plot_compare.py --dataset all                         # mọi ảnh pilot của cả hai tập
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # object-detection/
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from ce_localization.data.turns import TurnIndex, iter_jsonl, letterbox_size, present_objects  # noqa: E402

OUT = os.path.join(ROOT, "..", "output", "add")
MODELS = [("gpt-6-luna", "GPT-6 Luna"), ("glm-5.3-flash", "GLM-5.3-Flash"), ("gpt-4.1-mini", "gpt-4.1-mini"),
          ("perceptron-mk1.5", "Perceptron Mk1.5"), ("elorian-glimpse", "Elorian Glimpse")]
DSET = {"ce130": ("ce130_test_all", "inpainted"), "cocount": ("cocount_review5", "cocount")}
BLUE, GREEN, RED = "#1e6fff", "#00b050", "#e8000b"


def load_vlm(root, model, dset_dir, name, key):
    """-> {image_id: {"boxes" (canvas 512), "cnll" (từng box), "cnll_gt"}}, box xếp theo score."""
    jl = os.path.join(root, model, dset_dir, f"{name}.jsonl")
    if os.path.exists(jl):
        out = {}
        for r in iter_jsonl(jl):
            out[r["image_id"]] = {"boxes": [c["box_512"] for c in r["boxes"]], "cnll_gt": r.get("cnll_gt"),
                                  **({"cnll": [c["cnll"] for c in r["boxes"]]} if r["boxes"] and "cnll" in r["boxes"][0] else {})}
        return out
    with open(os.path.join(root, model, dset_dir, f"{name}_boxes.json")) as f:
        return {r["image_id"]: r for r in json.load(f)["results"][key]}


def scene(dataset, image_id, a, cache={}):
    """-> (đường dẫn ảnh model nhận, cỡ ảnh gốc, vật đang có, box GT) pixel ảnh gốc."""
    if dataset == "ce130":
        if "ti" not in cache:
            cache["ti"] = TurnIndex(a.turn_index)
        ti = cache["ti"]
        e = ti.turns[image_id]
        b = ti.branches[e["branch"]]
        t = e["t"]
        objs = present_objects(b, t)
        holes = np.asarray(b["holes"][:t], dtype=float).reshape(-1, 4)
        return os.path.join(a.samples_root, e["sample"]), tuple(b["wh"]), objs, holes
    from ce_localization.data.cocount import load_review, read_cocount
    if "rev" not in cache:
        cache["rev"] = load_review(a.cocount_review)
    r = read_cocount(a.cocount_root, image_id, review=cache["rev"])
    path = os.path.join(a.cocount_root, "Image", image_id + ".jpg")
    with Image.open(path) as im:
        wh = im.size
    return path, wh, np.asarray(r["objects"], dtype=float).reshape(-1, 4), np.asarray(r["loc"], dtype=float).reshape(-1, 4)


def to_orig(boxes, wh, image_size=512):
    return np.asarray(boxes, dtype=float).reshape(-1, 4) / letterbox_size(wh, image_size)[0]


def unique(boxes, cnll):
    """Gộp box trùng hệt (VLM thiếu box được lặp vòng cho đủ K) — giữ lần đầu."""
    seen, keep = set(), []
    for i, b in enumerate(np.round(boxes, 2)):
        k = tuple(b)
        if k not in seen:
            seen.add(k)
            keep.append(i)
    return boxes[keep], np.asarray(cnll, dtype=float)[keep]


def draw(ax, img, boxes, color, lw, ls="-", alpha=1.0):
    for b in boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, linewidth=lw, linestyle=ls,
                               alpha=alpha))


def panel(ax, img, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    ax.axis("off")


def model_panel(ax, img, name, boxes, cnll):
    if not len(boxes):
        panel(ax, img, f"{name}, no box")
        return
    panel(ax, img, f"{name}, CNLL = {cnll[0]:.2f}")                    # box[0] = score cao nhất
    draw(ax, img, boxes[1:], RED, 0.7, alpha=0.75)
    draw(ax, img, boxes[:1], RED, 2.4)


def figure(dataset, image_id, a, vlm, out_path):
    path, wh, objs, holes = scene(dataset, image_id, a)
    img = Image.open(path).convert("RGB")
    pw = 5.0                                                     # inch / ô
    ph = pw * wh[1] / wh[0]
    top, gap = 1.0, 0.45                                         # inch: chú thích + tên ảnh | tiêu đề ô hàng 2
    H = top + 2 * ph + gap + 0.35
    fig, axes = plt.subplots(2, 3, figsize=(3 * pw + 0.4, H))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - (top + 0.3) / H, wspace=0.03, hspace=gap / ph)
    gt = [min(r["cnll_gt"]) for r in (v.get(image_id) for v in vlm.values()) if r and r.get("cnll_gt")]
    panel(axes[0, 0], img, f"Original, CNLL = {gt[0]:.2f}" if gt else "Original")
    draw(axes[0, 0], img, objs, BLUE, 0.9)
    draw(axes[0, 0], img, holes, GREEN, 1.6, ls="--")
    for ax, (key, name) in zip(axes.flat[1:], MODELS):
        r = vlm[key].get(image_id)
        if r is None:
            panel(ax, img, f"{name}, missing")
            continue
        model_panel(ax, img, name, *unique(to_orig(r["boxes"], wh), r.get("cnll") or [float("nan")] * len(r["boxes"])))
    handles = [Line2D([], [], color=BLUE, lw=1.2, label="Existing objects"),
               Line2D([], [], color=GREEN, lw=1.6, ls="--", label="Ground-truth boxes"),
               Line2D([], [], color=RED, lw=1.2, label="Predicted boxes (bold = highest confidence)")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 1 - 0.05 / H))
    fig.text(0.5, 1 - 0.62 / H, f"{image_id}  |  class: {scene_class(dataset, image_id, a)}", ha="center", va="top", fontsize=11)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def scene_class(dataset, image_id, a, cache={}):
    if dataset == "ce130":
        if "ti" not in cache:
            cache["ti"] = TurnIndex(a.turn_index)
        return cache["ti"].branches[cache["ti"].turns[image_id]["branch"]]["class"]
    with open(os.path.join(a.cocount_root, "Anno", image_id + ".json")) as f:
        return json.load(f)["class_name"].rstrip(". ")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ce130", choices=["ce130", "cocount", "all"])
    ap.add_argument("--name", default="pilot10", help="tên lần chạy VLM (file <name>_boxes.json)")
    ap.add_argument("--n", type=int, default=None, help="chỉ vẽ n ảnh đầu (theo thứ tự trong dump VLM)")
    ap.add_argument("--ids", nargs="*", default=None, help="chỉ vẽ các image_id này")
    ap.add_argument("--vlm-root", default=os.path.join(OUT, "vlm"))
    ap.add_argument("--out", default=os.path.join(OUT, "viz", "vlm_compare"))
    ap.add_argument("--turn-index", default=os.path.join(ROOT, "data", "turn_index.json"))
    ap.add_argument("--samples-root", default=os.path.join(ROOT, "data", "samples"))
    ap.add_argument("--cocount-root", default=os.path.join(ROOT, "data", "cocount"))
    ap.add_argument("--cocount-review", default=os.path.join(ROOT, "data", "cocount_review.jsonl"))
    a = ap.parse_args(argv)
    for dataset in (["ce130", "cocount"] if a.dataset == "all" else [a.dataset]):
        dset_dir, key = DSET[dataset]
        vlm = {m: load_vlm(a.vlm_root, m, dset_dir, a.name, key) for m, _ in MODELS}
        ids = a.ids or list(vlm[MODELS[0][0]])
        for image_id in ids[:a.n] if a.n else ids:
            out = os.path.join(a.out, dset_dir, f"{image_id}.png")
            figure(dataset, image_id, a, vlm, out)
            print(out)


if __name__ == "__main__":
    main()
