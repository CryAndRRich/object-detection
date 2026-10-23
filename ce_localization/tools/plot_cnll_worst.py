#!/usr/bin/env python3
"""Vẽ các mẫu CE-CoCount có C-NLL TỆ nhất (hoặc tốt nhất) từ box dump của `eval.py --dataset cocount --dump-boxes` — KHÔNG model, KHÔNG
GPU (người dùng 2026-10-23: soi case xấu của DELTA5.4 sau lọc box tay).

Dựng lại record từng mẫu như lúc eval (`rescore_cocount.cocount_geometry`, cùng `--cocount-review` / `--cocount-obj-filter`), tính C-NLL
TỪNG box bằng đúng `engine/add_eval.cnll` (F1, hoặc F1all: quan hệ với vật cả hai lớp) rồi xếp mẫu theo TRUNG VỊ C-NLL 30 box của mẫu
(chỉ số bảng = trung vị gộp mọi box mọi mẫu). Mỗi mẫu một hình, chỉ có ảnh + box + 4 dòng chú thích cùng màu box (tên hai lớp, chỗ trống GT, box model thêm lớp nào):
  cyan  = box vật lớp cần thêm (tên cyan), cam = box vật lớp kia (tên cam), xanh lá nét đứt = 10 chỗ trống GT,
  đỏ    = `--select` box model chọn TUẦN TỰ như bài Add One Take One (`add_eval.select_sequential`: box C-NLL thấp nhất = q cao nhất,
          thêm nó vào tập vật, fit lại, chọn tiếp).

  cd object-detection/ce_localization
  python tools/plot_cnll_worst.py ../../output/add/cocount/review5/boxes_delta5_4_mockddim.json --key cocount_mock --kind F1all \\
      --cocount-review ../../output/add/box_review/cocount.jsonl --cocount-root ../data/cocount \\
      --out ../../output/add/viz/cnll_worst/delta5_4_review5_mock_F1all
  -> worst_01_<mẫu>.png … + rank.json (mọi mẫu: trung vị / min C-NLL 30 box, trung vị box GT, số vật, box đã chọn + C-NLL lúc chọn)
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import clean_class, load_review, twin_name  # noqa: E402
from ce_localization.engine.add_eval import clip_boxes, cnll, other_class, select_sequential  # noqa: E402
from ce_localization.tools.rescore_cocount import cocount_geometry  # noqa: E402
from ce_localization.tools.viz_common import rect  # noqa: E402

__all__ = ["score_samples", "plot_sample"]


def score_samples(rows, geo, kind="F1", k=5):
    """rows = `dump["results"][khoá]`, geo = {image_id: `cocount_geometry`} -> list dict mỗi mẫu đủ vật: image_id, boxes (đã kẹp như
    add_metrics), c (C-NLL từng box), median, min, gt_median (10 chỗ trống GT, cùng Gaussian), n_obj, sel / sel_cnll (k box chọn tuần tự
    như bài + C-NLL lúc chọn). `kind`: F1 | F1all."""
    out = []
    for x in rows:
        g = geo[x["image_id"]]
        if not np.allclose(g["wh"], x["wh"], atol=1e-3):
            raise ValueError(f"{x['image_id']}: vùng ảnh {g['wh'].tolist()} khác dump {x['wh']} — sai --image-size / cách lọc box vật?")
        b, _ = clip_boxes(x["boxes"])
        others = other_class(g["objects"], g["objects_all"]) if kind == "F1all" else None
        wh = tuple(g["wh"])
        c = cnll(b, g["objects"], wh, "F1", others=others)
        if c is None:
            continue
        gt = cnll(g["holes"], g["objects"], wh, "F1", others=others)
        sel, sel_c = select_sequential(b, g["objects"], wh, k, "F1", others=others)
        out.append({"image_id": x["image_id"], "boxes": b, "c": c, "median": float(np.median(c)), "min": float(c.min()),
                    "gt_median": float(np.median(gt)), "n_obj": len(g["objects"]), "sel": sel, "sel_cnll": sel_c})
    return out


def _class(root, name):
    p = os.path.join(root, "Anno", name + ".json")
    if not os.path.exists(p):
        return "?"
    with open(p) as f:
        return clean_class(json.load(f)["class_name"])


def plot_sample(it, geo, root, path):
    """MỘT mẫu -> một hình: ảnh + box vật hai lớp + chỗ trống GT + box model đã chọn; chữ duy nhất = tên hai lớp (cùng màu box)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = geo[it["image_id"]]
    nw, nh = (int(round(v)) for v in g["wh"])
    fig = plt.figure(figsize=(9, 9 * nh / nw))
    ax = fig.add_axes([0, 0, 1, 1])
    with Image.open(os.path.join(root, "Image", it["image_id"] + ".jpg")) as im:
        ax.imshow(np.asarray(im.convert("RGB").resize((nw, nh), resample=Image.BILINEAR)))
    for b in other_class(g["objects"], g["objects_all"]):
        rect(ax, b, "orange", lw=1.0)
    for b in g["objects"]:
        rect(ax, b, "cyan", lw=1.2)
    for b in g["holes"]:
        rect(ax, b, "lime", lw=1.6, ls="--")
    for i in it["sel"]:
        rect(ax, it["boxes"][i], "red", lw=2.4)
    ax.set_xlim(0, nw)
    ax.set_ylim(nh, 0)
    ax.axis("off")
    box = {"facecolor": "black", "alpha": 0.6, "pad": 3, "edgecolor": "none"}
    cls = _class(root, it["image_id"])
    lines = [(cls, "cyan"), (_class(root, twin_name(it["image_id"])), "orange"), (f"GT empty spots ({cls})", "lime"),
             (f"model: add {cls}", "red")]
    for k, (txt, col) in enumerate(lines):                              # dòng cách 30 pt
        ax.text(0.015, 0.985 - k * 30 / (9 * nh / nw * 72), txt, color=col, fontsize=14, fontweight="bold", va="top",
                transform=ax.transAxes, bbox=box)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="boxes_<run>.json của eval.py --dump-boxes (CE-CoCount)")
    ap.add_argument("--key", default="cocount_mock", help="khoá trong dump (vd cocount_mock | cocount_ddim | cocount_mock_objsize)")
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--cocount-review", default=None, help="PHẢI giống lúc eval (review5: jsonl rà box tay)")
    ap.add_argument("--cocount-obj-filter", type=float, default=None, help="PHẢI giống lúc eval (filt3: 3)")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--kind", default="F1", choices=["F1", "F1all"], help="F1 = hàng 'C-NLL', F1all = 'C-NLL all box'")
    ap.add_argument("--order", default="worst", choices=["worst", "best"])
    ap.add_argument("--top", type=int, default=24, help="số mẫu vẽ (mỗi mẫu một hình)")
    ap.add_argument("--select", type=int, default=5, help="số box model vẽ, chọn tuần tự như bài (C-NLL thấp nhất, thêm vào vật, lặp)")
    ap.add_argument("--out", required=True, help="thư mục ghi hình + rank.json")
    a = ap.parse_args()
    if a.cocount_review and a.cocount_obj_filter is not None:
        sys.exit("--cocount-review thay cho --cocount-obj-filter: chọn một (giống lúc eval)")
    with open(a.dump) as f:
        dump = json.load(f)
    if a.key not in dump["results"]:
        sys.exit(f"không có khoá {a.key} trong dump (có: {sorted(dump['results'])})")
    review = load_review(a.cocount_review) if a.cocount_review else None
    rows = dump["results"][a.key]
    geo = {x["image_id"]: cocount_geometry(a.cocount_root, x["image_id"], a.image_size, a.cocount_obj_filter, review) for x in rows}
    items = score_samples(rows, geo, a.kind, a.select)
    items.sort(key=lambda d: d["median"], reverse=a.order == "worst")
    for r, it in enumerate(items, 1):
        it["rank"] = r
    allc = np.concatenate([it["c"] for it in items])
    print(f"{a.key}: {len(items)} / {len(rows)} mẫu có C-NLL | trung vị gộp mọi box {np.median(allc):.3f} (= bảng) | trung vị theo mẫu: "
          f"p50 {np.median([d['median'] for d in items]):.2f}, p90 {np.percentile([d['median'] for d in items], 90):.2f}, "
          f"max {max(d['median'] for d in items):.1f}")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "rank.json"), "w") as f:
        json.dump({"dump": a.dump, "key": a.key, "kind": a.kind, "order": a.order, "review": a.cocount_review,
                   "obj_filter": a.cocount_obj_filter, "select": a.select,
                   "samples": [{k: it[k] for k in ("rank", "image_id", "median", "min", "gt_median", "n_obj", "sel", "sel_cnll")}
                               for it in items]}, f, indent=1)
    for it in items[:a.top]:
        plot_sample(it, geo, a.cocount_root, os.path.join(a.out, f"{a.order}_{it['rank']:02d}_{it['image_id']}.png"))
    print(f"-> {min(a.top, len(items))} hình {a.order}_NN_<mẫu>.png + rank.json trong {a.out}")


if __name__ == "__main__":
    main()
