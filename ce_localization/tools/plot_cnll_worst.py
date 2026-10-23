#!/usr/bin/env python3
"""Vẽ các mẫu có C-NLL TỆ nhất (hoặc tốt nhất) từ dump `<out>_boxes.json` của `eval.py` — CE-CoCount (`--dataset cocount`) hoặc CE-130
(`--dataset ce130`) — KHÔNG model, KHÔNG GPU.

Dựng lại hình học từng mẫu như lúc eval (CE-CoCount: `data/cocount.cocount_geometry`, cùng `--cocount-review` / `--cocount-obj-filter`;
CE-130: `ce130_geometry` = `CE130AddDataset` ảnh inpaint), tính C-NLL TỪNG box bằng `engine/add_eval.cnll` (như eval: quan hệ với vật cả
hai lớp ở CE-CoCount) rồi xếp mẫu theo `--rank-by`: `min` = C-NLL box tốt nhất (số của bảng) | `median` = trung vị K box.
Mỗi mẫu một hình: ảnh + box + vài dòng chữ cùng màu box — cyan = vật lớp cần thêm, cam = vật lớp kia (CE-CoCount), xanh lá nét đứt =
chỗ trống / lỗ GT, đỏ = `--select` box model chọn TUẦN TỰ như bài Add One Take One (`add_eval.select_sequential`).

  cd object-detection/ce_localization
  python tools/plot_cnll_worst.py ../../output/add/cocount/review5/test_delta6_2_boxes.json --key cocount_mock \\
      --cocount-review ../../output/add/box_review/cocount.jsonl --out ../../output/add/viz/cnll_worst/delta6_2_review5_mock
  python tools/plot_cnll_worst.py ../../output/add/ce130_test_all/test_delta6_2_boxes.json --dataset ce130 --key inpainted_mock \\
      --top-pct 1 --out ../../output/add/viz/cnll_worst/delta6_2_ce130_mock
  -> worst_01_<mẫu>.png … + rank.json (mọi mẫu: trung vị / min C-NLL K box, trung vị box GT, số vật, box đã chọn + C-NLL lúc chọn)
"""

import argparse
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import clean_class, cocount_geometry, load_review, twin_name  # noqa: E402
from ce_localization.data.turns import TurnIndex, scale_boxes  # noqa: E402
from ce_localization.engine.add_eval import clip_boxes, cnll, other_class, select_sequential  # noqa: E402

__all__ = ["ce130_geometry", "score_samples", "plot_sample"]


def ce130_geometry(index, samples_root, key, image_size=512):
    """Mẫu CE-130 (nhánh, lượt) ảnh inpaint -> hình học như `CE130AddDataset` (pixel canvas, float32 như lúc eval) + ảnh + chú thích."""
    e = index.turns[key]
    b = index.branches[e["branch"]]
    t = e["t"]
    w, h = b["wh"]
    s = min(image_size / w, image_size / h)
    nw, nh = int(w * s), int(h * s)
    removed = np.asarray(b["removed"], dtype=int)
    objs = np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)[(removed == 0) | (removed > t)]
    sb = lambda x: scale_boxes(x, s, nw, nh).astype(np.float32).astype(np.float64)  # noqa: E731
    return {"wh": np.array([nw, nh], dtype=np.float64), "objects": sb(objs), "holes": sb(b["holes"][:t]),
            "image": os.path.join(samples_root, e["sample"]), "cls": b["class"],
            "labels": [(b["class"], "cyan"), (f"GT holes ({b['class']})", "lime")]}


def score_samples(rows, geo, k=5):
    """rows = `dump["results"][khoá]`, geo = {image_id: hình học} -> list dict mỗi mẫu đủ vật: image_id, boxes (đã kẹp), c (C-NLL từng
    box), median, min, gt_median (lỗ GT), n_obj, sel / sel_cnll (k box chọn tuần tự + C-NLL lúc chọn)."""
    out = []
    for x in rows:
        g = geo[x["image_id"]]
        if not np.allclose(g["wh"], x["wh"], atol=1e-3):
            raise ValueError(f"{x['image_id']}: vùng ảnh {g['wh'].tolist()} khác dump {x['wh']} — sai --image-size / cách lọc box vật?")
        b, _ = clip_boxes(x["boxes"])
        others = other_class(g["objects"], g["objects_all"]) if "objects_all" in g else None
        wh = tuple(g["wh"])
        c = cnll(b, g["objects"], wh, others=others)
        if c is None:
            continue
        gt = cnll(g["holes"], g["objects"], wh, others=others)
        sel, sel_c = select_sequential(b, g["objects"], wh, k, others=others)
        out.append({"image_id": x["image_id"], "boxes": b, "c": c, "median": float(np.median(c)), "min": float(c.min()),
                    "gt_median": float(np.median(gt)), "n_obj": len(g["objects"]), "sel": sel, "sel_cnll": sel_c})
    return out


def _rect(ax, b, color, lw=1.5, ls="-"):
    import matplotlib.patches as patches
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, linewidth=lw, linestyle=ls))


def plot_sample(it, g, path):
    """MỘT mẫu -> một hình: ảnh + box vật (+ lớp kia) + chỗ trống / lỗ GT + box model đã chọn; chữ = `labels` + dòng box model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nw, nh = (int(round(v)) for v in g["wh"])
    fig = plt.figure(figsize=(9, 9 * nh / nw))
    ax = fig.add_axes([0, 0, 1, 1])
    with Image.open(g["image"]) as im:
        ax.imshow(np.asarray(im.convert("RGB").resize((nw, nh), resample=Image.BILINEAR)))
    for b in other_class(g["objects"], g["objects_all"]) if "objects_all" in g else ():
        _rect(ax, b, "orange", lw=1.0)
    for b in g["objects"]:
        _rect(ax, b, "cyan", lw=1.2)
    for b in g["holes"]:
        _rect(ax, b, "lime", lw=1.6, ls="--")
    for i in it["sel"]:
        _rect(ax, it["boxes"][i], "red", lw=2.4)
    ax.set_xlim(0, nw)
    ax.set_ylim(nh, 0)
    ax.axis("off")
    box = {"facecolor": "black", "alpha": 0.6, "pad": 3, "edgecolor": "none"}
    for k, (txt, col) in enumerate(g["labels"] + [(f"model: add {g['cls']}", "red")]):      # dòng cách 30 pt
        ax.text(0.015, 0.985 - k * 30 / (9 * nh / nw * 72), txt, color=col, fontsize=14, fontweight="bold", va="top",
                transform=ax.transAxes, bbox=box)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _class(root, name):
    with open(os.path.join(root, "Anno", name + ".json")) as f:
        return clean_class(json.load(f)["class_name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="<out>_boxes.json của eval.py")
    ap.add_argument("--dataset", default="cocount", choices=["cocount", "ce130"])
    ap.add_argument("--key", default=None, help="khoá trong dump (mặc định cocount | inpainted; vd cocount_mock, inpainted_ddim)")
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--cocount-review", default=None, help="PHẢI giống lúc eval (review5)")
    ap.add_argument("--cocount-obj-filter", type=float, default=None, help="PHẢI giống lúc eval (filt3: 3)")
    ap.add_argument("--turn-index", default="../data/turn_index.json", help="CE-130")
    ap.add_argument("--samples-root", default="../data/samples", help="CE-130: ảnh inpaint của mẫu")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--rank-by", default="min", choices=["min", "median"], help="min = box tốt nhất (số của bảng) | median = trung vị K box")
    ap.add_argument("--order", default="worst", choices=["worst", "best"])
    ap.add_argument("--top", type=int, default=24, help="số mẫu vẽ (mỗi mẫu một hình)")
    ap.add_argument("--top-pct", type=float, default=None, help="thay --top: vẽ p %% số mẫu có C-NLL (làm tròn lên)")
    ap.add_argument("--select", type=int, default=5, help="số box model vẽ, chọn tuần tự như bài")
    ap.add_argument("--out", required=True, help="thư mục ghi hình + rank.json")
    a = ap.parse_args()
    if a.cocount_review and a.cocount_obj_filter is not None:
        sys.exit("--cocount-review thay cho --cocount-obj-filter: chọn một (giống lúc eval)")
    with open(a.dump) as f:
        dump = json.load(f)
    a.key = a.key or ("inpainted" if a.dataset == "ce130" else "cocount")
    if a.key not in dump["results"]:
        sys.exit(f"không có khoá {a.key} trong dump (có: {sorted(dump['results'])})")
    rows = dump["results"][a.key]
    if a.dataset == "ce130":
        index = TurnIndex(a.turn_index)
        geo = {x["image_id"]: ce130_geometry(index, a.samples_root, x["image_id"], a.image_size) for x in rows}
    else:
        review = load_review(a.cocount_review) if a.cocount_review else None
        geo = {}
        for x in rows:
            n = x["image_id"]
            cls = _class(a.cocount_root, n)
            geo[n] = {**cocount_geometry(a.cocount_root, n, a.image_size, a.cocount_obj_filter, review),
                      "image": os.path.join(a.cocount_root, "Image", n + ".jpg"), "cls": cls,
                      "labels": [(cls, "cyan"), (_class(a.cocount_root, twin_name(n)), "orange"), (f"GT empty spots ({cls})", "lime")]}
    items = score_samples(rows, geo, a.select)
    items.sort(key=lambda d: d[a.rank_by], reverse=a.order == "worst")
    for r, it in enumerate(items, 1):
        it["rank"] = r
    print(f"{a.key}: {len(items)} / {len(rows)} mẫu có C-NLL | TB box tốt nhất / ảnh {np.mean([d['min'] for d in items]):.3f} (= bảng) | "
          f"trung vị {np.median([d['min'] for d in items]):.3f}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "rank.json"), "w") as f:
        json.dump({"dump": a.dump, "dataset": a.dataset, "key": a.key, "rank_by": a.rank_by, "order": a.order,
                   "review": a.cocount_review, "obj_filter": a.cocount_obj_filter, "select": a.select,
                   "samples": [{k: it[k] for k in ("rank", "image_id", "median", "min", "gt_median", "n_obj", "sel", "sel_cnll")}
                               for it in items]}, f, indent=1)
    top = math.ceil(len(items) * a.top_pct / 100) if a.top_pct is not None else a.top
    w = max(2, len(str(top)))
    for it in items[:top]:
        plot_sample(it, geo[it["image_id"]], os.path.join(a.out, f"{a.order}_{it['rank']:0{w}d}_{it['image_id']}.png"))
    print(f"-> {min(top, len(items))} hình + rank.json trong {a.out}")


if __name__ == "__main__":
    main()
