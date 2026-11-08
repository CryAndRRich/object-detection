#!/usr/bin/env python3
"""So box các VLM: hình 2 × 3 / ảnh (chữ trong hình tiếng Anh).

  hàng 1: Original (ảnh model nhận + vật đang có xanh dương + box GT xanh lá nét đứt) | GLM-5.3-Flash | gpt-4.1-mini
  hàng 2: GPT-6 Luna | LLaVA-OV-2-8B | Qwen3-VL-4B   (`--models` chọn tối đa 5 model khác, vd RePlan / InternVL)

Ô model vẽ box dự đoán (đỏ mảnh) + một box đỏ đậm: `--bold min` (mặc định) = box C-NLL thấp nhất, `--bold top` = box `score` cao
nhất (= box chấm chính ZETA5 / 6). Tiêu đề "<tên>, CNLL = <C-NLL box đậm>"; Original = C-NLL thấp nhất của box GT. VLM =
`output/add/vlm/<model>/<tập>/<name>.jsonl` (1 dòng / ảnh, box xếp theo score); model chưa có `<name>.jsonl` (vd mới chạy pilot)
⇒ lấy `pilot<K>.jsonl` K lớn nhất; dump cũ `<name>_boxes.json` (pilot chấm trước 2026-11-04) vẫn đọc được. Box lặp vòng (VLM thiếu
box) gộp lại khi vẽ. Box dump là pixel canvas 512 (letterbox góc trên trái) -> chia tỉ lệ về pixel ảnh gốc. Chỉ đọc JSON + ảnh.

  cd object-detection/vlm
  python plot_compare.py --dataset ce130 --pilot 20 --n 1      # thử 1 ảnh
  python plot_compare.py --dataset all --pilot 20              # 20 ảnh pilot (hash seed 0, như run.py --pilot 20) cả hai tập

`--grid`: hình bài báo 2 hàng (1 ảnh CE-130 | 1 ảnh CoCount) × 8 cột — CE-Loc | 7 baseline (`GRID`),
mỗi ô 5 box C-NLL thấp nhất, viền đỏ đậm, tiêu đề = tên phương pháp. Ra `output/add/viz/method_grid/<ce130 id>__<cocount id>.png`.

  python plot_compare.py --grid --seed 0                        # ảnh ngẫu nhiên mỗi tập
  python plot_compare.py --grid --ids 1050_b1_t1 <cocount_id>   # chọn ảnh
  python plot_compare.py --grid --rank 50 --dataset all         # 50 ứng viên / tập CE-Loc vượt trội -> method_grid/candidates/
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # object-detection/
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from ce_localization.utils.plot import draw_boxes  # noqa: E402
from PIL import Image  # noqa: E402

from ce_localization.data.turns import TurnIndex, iter_jsonl, letterbox_size, present_objects  # noqa: E402

OUT = os.path.join(ROOT, "..", "output", "add")
NAMES = {"glm-5.3-flash": "GLM-5.3-Flash", "gpt-4.1-mini": "gpt-4.1-mini", "gpt-6-luna": "GPT-6 Luna", "llava-ov2-8b": "LLaVA-OV-2-8B",
         "qwen3-vl-4b": "Qwen3-VL-4B", "internvl3.5-8b": "InternVL3.5-8B", "replan-7b": "RePlan (ZETA4)"}
MODELS = [(m, NAMES[m]) for m in ("glm-5.3-flash", "gpt-4.1-mini", "gpt-6-luna", "llava-ov2-8b", "qwen3-vl-4b")]
DSET = {"ce130": ("ce130_test_all", "inpainted"), "cocount": ("cocount_review5", "cocount")}
BLUE, GREEN, RED = "#1e6fff", "#00b050", "#e8000b"
# --grid: cột (tên trong hình, nguồn) — nguồn ("vlm", model) | ("run", run, snapshot) = output/add/<run>/<tập>/<snapshot>_boxes.json.
# CE-Loc bắt buộc (cột đầu); còn lại: closed-source, open-source (chọn theo C-NLL min / 30 tốt), finetune. Không cột Groundtruth.
# CE-Loc = bản qua lọc tay (box `selected` của output/add/review/delta8_ep010/pred_<ce130|cocount>_celoc.jsonl, pixel ảnh gốc).
GRID = [("CE-Loc", ("celoc", "delta8_ep010")),                       # thứ tự cột: closed -> open-source -> finetune, trong nhóm cũ -> mới
        ("gpt-4.1-mini", ("vlm", "gpt-4.1-mini")), ("GPT-6 Luna", ("vlm", "gpt-6-luna")),
        ("Qwen3-VL-4B", ("vlm", "qwen3-vl-4b")), ("GLM-5.3-Flash", ("vlm", "glm-5.3-flash")),
        ("CSANet", ("run", "zeta2", "ep010")), ("BootPlace", ("run", "zeta1_1", "ep006")),
        ("HiddenObjects", ("run", "zeta3", "ep009"))]


def load_vlm(root, model, dset_dir, name, key):
    """-> {image_id: {"boxes" (canvas 512), "cnll" (từng box), "cnll_gt"}}, box xếp theo score."""
    jl = os.path.join(root, model, dset_dir, f"{name}.jsonl")
    if not os.path.exists(jl):                                 # chưa chạy <name> (vd full) -> pilot K lớn nhất đã chấm
        pil = [(int(m.group(1)), f) for f in glob.glob(os.path.join(root, model, dset_dir, "pilot*.jsonl"))
               if (m := re.fullmatch(r"pilot(\d+)\.jsonl", os.path.basename(f)))]
        if pil:
            jl = max(pil)[1]
    if os.path.exists(jl):
        out = {}
        for r in iter_jsonl(jl):
            out[r["image_id"]] = {"boxes": [c["box_512"] for c in r["boxes"]], "cnll_gt": r.get("cnll_gt"),
                                  **({"cnll": [c["cnll"] for c in r["boxes"]]} if r["boxes"] and "cnll" in r["boxes"][0] else {})}
        return out
    old = os.path.join(root, model, dset_dir, f"{name}_boxes.json")
    if not os.path.exists(old):
        return {}
    with open(old) as f:
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


def panel(ax, img, title):
    ax.imshow(img)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, img.size[0])
    ax.set_ylim(img.size[1], 0)
    ax.axis("off")


def model_panel(ax, img, name, boxes, cnll, bold="min"):
    if not len(boxes):
        panel(ax, img, f"{name}, no box")
        return
    # box[0] = score cao nhất; min = box C-NLL thấp nhất (nan -> không chọn)
    i = int(np.nanargmin(cnll)) if bold == "min" and np.isfinite(cnll).any() else 0
    panel(ax, img, f"{name}, CNLL = {cnll[i]:.2f}")
    draw_boxes(ax, np.delete(boxes, i, axis=0), RED, 0.7, alpha=0.75)
    draw_boxes(ax, boxes[i:i + 1], RED, 2.4)


def figure(dataset, image_id, a, vlm, out_path, models=MODELS):
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
    draw_boxes(axes[0, 0], objs, BLUE, 0.9)
    draw_boxes(axes[0, 0], holes, GREEN, 1.6, ls="--")
    for ax, (key, name) in zip(axes.flat[1:], models):
        r = vlm[key].get(image_id)
        if r is None:
            panel(ax, img, f"{name}, missing")
            continue
        model_panel(ax, img, name, *unique(to_orig(r["boxes"], wh), r.get("cnll") or [float("nan")] * len(r["boxes"])), bold=a.bold)
    handles = [Line2D([], [], color=BLUE, lw=1.2, label="Existing objects"),
               Line2D([], [], color=GREEN, lw=1.6, ls="--", label="Ground-truth boxes"),
               Line2D([], [], color=RED, lw=1.2, label="Predicted boxes (bold = " + ("lowest CNLL" if a.bold == "min" else "highest confidence") + ")")]
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


def load_grid(a, dset_dir, key):
    """-> {tên: {image_id: {"boxes" (canvas 512), "cnll"}}} cho mọi phương pháp của GRID."""
    out = {}
    for name, src in GRID:
        if src[0] == "vlm":
            out[name] = load_vlm(a.vlm_root, src[1], dset_dir, a.name, key)
        elif src[0] == "celoc":                                  # chỉ box đã chọn (lọc tay), pixel ảnh gốc
            ds = "ce130" if dset_dir.startswith("ce130") else "cocount"
            out[name] = {}
            for r in iter_jsonl(os.path.join(OUT, "review", src[1], f"pred_{ds}_celoc.jsonl")):
                sel, c = r.get("selected") or [], r.get("cnll") or []          # cnll None: ảnh < 5 vật (không chấm)
                out[name][r["image_id"]] = {"boxes": r.get("selected_boxes") or [], "orig": True,
                                            "cnll": [c[i] if i < len(c) else float("nan") for i in sel],
                                            "cnll_gt": r.get("cnll_gt")}
        else:
            with open(os.path.join(OUT, src[1], dset_dir, f"{src[2]}_boxes.json")) as f:
                out[name] = {r["image_id"]: r for r in json.load(f)["results"][key]}
    return out


def _iou(a, b):
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    i = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - i + 1e-9)


def pick(r, top=5, nms_iou=0.3):
    """-> chỉ số `top` box của r["boxes"]: C-NLL thấp nhất trước, bỏ box IoU > nms_iou với box đã lấy (box trùng / dí sát nhau —
    kể cả trong box `selected` lọc tay của CE-Loc), bỏ C-NLL nan. IoU không đổi theo thang ⇒ dùng thẳng toạ độ dump."""
    b = np.asarray(r.get("boxes") or [], dtype=float).reshape(-1, 4)
    c = np.asarray(r.get("cnll") or [np.nan] * len(b), dtype=float)
    keep = []
    for i in np.argsort(np.where(np.isfinite(c), c, np.inf), kind="stable"):
        if not np.isfinite(c[i]) or len(keep) == top:
            break
        if not keep or _iou(b[i], b[keep]).max() <= nms_iou:
            keep.append(i)
    return keep


def rank_ids(preds, max_obj, top=5):
    """Ảnh CE-Loc vượt trội: điểm = (TB C-NLL 5 box `pick` của phương pháp khác TỐT NHẤT) / (của CE-Loc), chỉ giữ ảnh CE-Loc còn đủ 5
    box `selected` (lọc tay) sau `pick`, TB 5 box CE-Loc <= TB C-NLL box GT, n vật <= max_obj. -> [(điểm, image_id)] giảm dần."""
    def m(r):
        k = pick(r, top)
        return np.mean(np.asarray(r["cnll"], dtype=float)[k]) if len(k) == top else np.inf
    out = []
    nobj = preds["CSANet"]                                       # n_obj có trong dump baseline
    for k, r in preds["CE-Loc"].items():
        if k not in nobj or nobj[k].get("n_obj", 0) > max_obj or not all(k in v for v in preds.values()):
            continue
        ce = m(r)
        if not np.isfinite(ce) or ce > np.mean(r.get("cnll_gt") or [np.inf]):
            continue
        others = min(m(v[k]) for name, v in preds.items() if name != "CE-Loc")
        out.append((others / max(ce, 1e-3), k))
    return sorted(out, reverse=True)


def grid_figure(rows, a, out_path, top=5):
    """rows = [(dataset, image_id, preds)] — mỗi hàng một ảnh (CE-130, CoCount); cột = GRID; tiêu đề chỉ ở hàng đầu."""
    cw, gap, head, pad = 2.6, 0.06, 0.28, 0.03                 # inch: rộng ô | khe hàng | tiêu đề | khe cột; ô = đúng tỉ lệ ảnh
    scenes = [scene(dataset, image_id, a) for dataset, image_id, _ in rows]
    hs = [cw * sc[1][1] / sc[1][0] for sc in scenes]
    ncol = len(GRID)
    W, H = ncol * cw + (ncol - 1) * pad, head + sum(hs) + gap * (len(rows) - 1)
    fig = plt.figure(figsize=(W, H))
    axes, y = np.empty((len(rows), ncol), dtype=object), H - head
    for ri, h in enumerate(hs):
        y -= h
        for ci in range(ncol):
            axes[ri, ci] = fig.add_axes([ci * (cw + pad) / W, y / H, cw / W, h / H])
        y -= gap
    for ri, ((dataset, image_id, preds), (path, wh, _, _)) in enumerate(zip(rows, scenes)):
        img = Image.open(path).convert("RGB")
        for ci, name in enumerate(g[0] for g in GRID):
            ax = axes[ri, ci]
            panel(ax, img, name if ri == 0 else "")
            r = preds[name].get(image_id)
            if r is None or not r.get("boxes"):
                continue
            b = np.asarray(r["boxes"], dtype=float).reshape(-1, 4)
            draw_boxes(ax, (b if r.get("orig") else to_orig(b, wh))[pick(r, top)], RED, 2.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="ce130", choices=["ce130", "cocount", "all"])
    ap.add_argument("--name", default="test", help="tên lần chạy VLM (file <name>.jsonl; thiếu -> pilot<K>.jsonl K lớn nhất)")
    ap.add_argument("--pilot", type=int, default=None, help="chỉ vẽ K ảnh pilot (hash seed 0, cùng tập run.py --pilot K)")
    ap.add_argument("--bold", default="min", choices=["min", "top"], help="box đậm: C-NLL thấp nhất | score cao nhất")
    ap.add_argument("--models", nargs="+", default=None, choices=list(NAMES), help="tối đa 5 model (mặc định: 5 model ZETA5 / 6 đã chạy)")
    ap.add_argument("--n", type=int, default=None, help="chỉ vẽ n ảnh đầu (theo thứ tự trong dump VLM)")
    ap.add_argument("--ids", nargs="*", default=None, help="chỉ vẽ các image_id này")
    ap.add_argument("--grid", action="store_true", help="hình 2 hàng (CE-130 | CoCount) × GRID (8 cột), 5 box C-NLL thấp nhất")
    ap.add_argument("--rank", type=int, default=None, help="--grid: vẽ N ảnh ứng viên / tập CE-Loc vượt trội (1 hàng / hình)")
    ap.add_argument("--max-obj", type=int, nargs=2, default=[40, 80], help="--rank: số vật tối đa CE-130, CoCount")
    ap.add_argument("--seed", type=int, default=None, help="--grid: seed chọn ảnh ngẫu nhiên (không --ids)")
    ap.add_argument("--vlm-root", default=os.path.join(OUT, "vlm"))
    ap.add_argument("--out", default=os.path.join(OUT, "viz", "vlm_compare"))
    ap.add_argument("--turn-index", default=os.path.join(ROOT, "data", "turn_index.json"))
    ap.add_argument("--samples-root", default=os.path.join(ROOT, "data", "samples"))
    ap.add_argument("--cocount-root", default=os.path.join(ROOT, "data", "cocount"))
    ap.add_argument("--cocount-review", default=os.path.join(ROOT, "data", "cocount_review.jsonl"))
    a = ap.parse_args(argv)
    if a.grid:                                                   # hàng 1 CE-130, hàng 2 CoCount; ảnh ngẫu nhiên (--seed) nếu không --ids
        if a.rank:                                               # ứng viên: a.rank ảnh / tập CE-Loc vượt trội, mỗi hình 1 hàng
            for dataset in (["ce130", "cocount"] if a.dataset == "all" else [a.dataset]):
                dset_dir, key = DSET[dataset]
                preds = load_grid(a, dset_dir, key)
                ranked = rank_ids(preds, a.max_obj[dataset == "cocount"])
                print(f"[{dataset}] {len(ranked)} ảnh qua lọc -> vẽ {min(a.rank, len(ranked))}")
                for i, (sc, image_id) in enumerate(ranked[:a.rank], 1):
                    grid_figure([(dataset, image_id, preds)], a,
                                os.path.join(OUT, "viz", "method_grid", "candidates", dset_dir, f"{i:02d}_{image_id}.png"))
            return
        import random
        rng, rows = random.Random(a.seed), []
        for i, dataset in enumerate(("ce130", "cocount")):
            dset_dir, key = DSET[dataset]
            preds = load_grid(a, dset_dir, key)
            ids = [k for k in preds["CE-Loc"] if all(k in v for v in preds.values())]
            rows.append((dataset, a.ids[i] if a.ids else rng.choice(sorted(ids)), preds))
        out = os.path.join(OUT, "viz", "method_grid", "__".join(r[1] for r in rows) + ".png")
        grid_figure(rows, a, out)
        print(out)
        return
    for dataset in (["ce130", "cocount"] if a.dataset == "all" else [a.dataset]):
        dset_dir, key = DSET[dataset]
        models = [(m, NAMES[m]) for m in a.models[:5]] if a.models else MODELS
        vlm = {m: load_vlm(a.vlm_root, m, dset_dir, a.name, key) for m, _ in models}
        ids = a.ids or max((list(v) for v in vlm.values()), key=len)              # thứ tự gốc của tập (file full)
        if a.pilot:
            from vlm.runner import select
            ids = [s["image_id"] for s in select([{"image_id": i} for i in ids], 0, a.pilot)]
        for image_id in ids[:a.n] if a.n else ids:
            out = os.path.join(a.out, dset_dir, f"{image_id}.png")
            figure(dataset, image_id, a, vlm, out, models)
            print(out)


if __name__ == "__main__":
    main()
