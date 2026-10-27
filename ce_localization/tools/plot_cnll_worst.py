#!/usr/bin/env python3
"""Vẽ các mẫu có C-NLL TỆ nhất (hoặc tốt nhất) từ dump `<out>_boxes.json` của `eval.py` — CE-CoCount (`--dataset cocount`) hoặc CE-130
(`--dataset ce130`) — KHÔNG model, KHÔNG GPU.

Dựng lại hình học từng mẫu như lúc eval (CE-CoCount: `data/cocount.cocount_geometry`, cùng `--cocount-review` / `--cocount-obj-filter`;
CE-130: `ce130_geometry` = `CE130AddDataset` ảnh inpaint), tính C-NLL TỪNG box bằng `engine/add_eval.cnll` (như eval: quan hệ với vật cả
hai lớp ở CE-CoCount) rồi xếp mẫu theo `--rank-by`: `min` = C-NLL box tốt nhất (số của bảng) | `median` = trung vị K box.
Mỗi mẫu một hình: ảnh + box + vài dòng chữ cùng màu box — cyan = vật lớp cần thêm, cam = vật lớp kia (CE-CoCount), xanh lá nét đứt =
chỗ trống / lỗ GT, đỏ = `--select` box model C-NLL thấp nhất sau NMS (`--nms-iou`, bỏ box trùng giữa K mẫu; `add_eval.select_nms`).

  cd object-detection/ce_localization
  python tools/plot_cnll_worst.py ../../output/add/cocount/review5/test_delta6_2_boxes.json --key cocount_mock \\
      --cocount-review ../../output/add/box_review/cocount.jsonl --out ../../output/add/viz/cnll_worst/delta6_2_review5_mock
  python tools/plot_cnll_worst.py ../../output/add/ce130_test_all/test_delta6_2_boxes.json --dataset ce130 --key inpainted_mock \\
      --top-pct 1 --out ../../output/add/viz/cnll_worst/delta6_2_ce130_mock
  -> worst_01_<mẫu>.png … + rank.json (mọi mẫu: trung vị / min C-NLL K box, trung vị box GT, số vật, box đã chọn + C-NLL của chúng)
Nhiều dump `TÊN=đường dẫn` (cùng tập mẫu, vd các model): vẽ mẫu nằm trong `--intersect-pct` % tệ nhất của MỌI dump, mỗi mẫu một lưới
(Ground truth + từng setting) -> worst_01_<mẫu>.png … + intersect.json.
`--trend` + nhiều dump theo thứ tự thời gian (vd snapshot `ep010=… ep020=…`): xếp mẫu theo C-NLL box tốt nhất tăng từ dump đầu tới dump
cuối, vẽ `--top` mẫu tăng mạnh nhất, mỗi mẫu một lưới (mỗi dump một panel: box C-NLL thấp nhất đậm + K box nhạt) -> trend_01_<mẫu>.png
… + trend.json (mọi mẫu: C-NLL theo dump, mức tăng, tương quan hạng với thứ tự dump).
"""

import argparse
import json
import math
import os
import re
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import clean_class, cocount_geometry, load_review, twin_name  # noqa: E402
from ce_localization.data.turns import TurnIndex, scale_boxes  # noqa: E402
from ce_localization.engine.add_eval import clip_boxes, cnll, other_class, select_nms  # noqa: E402

__all__ = ["ce130_geometry", "score_samples", "plot_sample", "plot_trend"]


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


def score_samples(rows, geo, k=5, nms_iou=0.5):
    """rows = `dump["results"][khoá]`, geo = {image_id: hình học} -> list dict mỗi mẫu đủ vật: image_id, boxes (đã kẹp), c (C-NLL từng
    box), median, min, gt_median (lỗ GT), n_obj, sel / sel_cnll (k box C-NLL thấp nhất sau NMS + C-NLL của chúng)."""
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
        sel = select_nms(b, c, k, nms_iou)
        sel_c = [float(c[i]) for i in sel]
        out.append({"image_id": x["image_id"], "boxes": b, "c": c, "median": float(np.median(c)), "min": float(c.min()),
                    "gt_median": float(np.median(gt)), "n_obj": len(g["objects"]), "sel": sel, "sel_cnll": sel_c})
    return out


def _rect(ax, b, color, lw=1.5, ls="-"):
    import matplotlib.patches as patches
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, linewidth=lw, linestyle=ls))


def _image(g):
    nw, nh = (int(round(v)) for v in g["wh"])
    with Image.open(g["image"]) as im:
        return np.asarray(im.convert("RGB").resize((nw, nh), resample=Image.BILINEAR))


def _draw(ax, g, img, boxes=(), faint=()):
    """Ảnh + box vật (+ lớp kia, cam) + chỗ trống / lỗ GT (xanh lá nét đứt) + `boxes` của model (đỏ) + `faint` (đỏ nhạt, mảnh)."""
    nw, nh = g["wh"]
    ax.imshow(img)
    _rect(ax, (0, 0, nw, nh), "#333333", lw=1.2)                         # viền ảnh (ảnh nền trắng đặt cạnh nhau)
    for b in other_class(g["objects"], g["objects_all"]) if "objects_all" in g else ():
        _rect(ax, b, "orange", lw=1.0)
    for b in g["objects"]:
        _rect(ax, b, "cyan", lw=1.2)
    for b in g["holes"]:
        _rect(ax, b, "lime", lw=1.6, ls="--")
    for b in faint:
        _rect(ax, b, (1.0, 0.25, 0.25, 0.45), lw=0.8)
    for b in boxes:
        _rect(ax, b, "red", lw=2.4)
    ax.set_xlim(-2, nw + 2)                                                # chừa lề cho viền
    ax.set_ylim(nh + 2, -2)
    ax.axis("off")


def plot_sample(it, g, path):
    """MỘT mẫu -> một hình: ảnh + box vật (+ lớp kia) + chỗ trống / lỗ GT + box model đã chọn; chữ = `labels` + dòng box model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nw, nh = (int(round(v)) for v in g["wh"])
    fig = plt.figure(figsize=(9, 9 * nh / nw))
    ax = fig.add_axes([0, 0, 1, 1])
    _draw(ax, g, _image(g), [it["boxes"][i] for i in it["sel"]])
    box = {"facecolor": "black", "alpha": 0.6, "pad": 3, "edgecolor": "none"}
    for k, (txt, col) in enumerate(g["labels"] + [(f"model: add {g['cls']}", "red")]):      # dòng cách 30 pt
        ax.text(0.015, 0.985 - k * 30 / (9 * nh / nw * 72), txt, color=col, fontsize=14, fontweight="bold", va="top",
                transform=ax.transAxes, bbox=box)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_grid(g, gt_cnll, settings, path, ncol=4):
    """MỘT mẫu, nhiều setting -> lưới: panel 1 = Ground truth (chỗ trống / lỗ GT, C-NLL box GT tốt nhất), mỗi panel sau = một setting
    (box model đã chọn, C-NLL box tốt nhất trong K). settings = [(tên, item của `score_samples`)]. Chú thích màu dưới hình."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    nw, nh = g["wh"]
    nrow = -(-(len(settings) + 1) // ncol)
    pw = 4.2
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * pw, nrow * (pw * nh / nw + 0.75) + 0.6), squeeze=False)
    img = _image(g)
    panels = [("Ground truth", f"C-NLL {gt_cnll:.2f}", ())] + [
        (name, f"C-NLL {it['min']:.2f}", [it["boxes"][i] for i in it["sel"]]) for name, it in settings]
    for ax, (name, sub, boxes) in zip(axes.ravel(), panels):
        _draw(ax, g, img, boxes)
        ax.set_title(f"{name}\n{sub}", fontsize=10.5)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    handles = [Line2D([], [], color="cyan", lw=1.5, label=f"{g['cls']} (existing)"),
               Line2D([], [], color="lime", lw=1.5, ls="--", label="GT empty spots" if "objects_all" in g else "GT holes"),
               Line2D([], [], color="red", lw=2.4, label=f"model: add {g['cls']}")]
    if "objects_all" in g:
        handles.insert(1, Line2D([], [], color="orange", lw=1.2, label=g["labels"][1][0]))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.6 / fig.get_figheight(), 1, 1))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_trend(g, gt_cnll, panels, path, ncol=6):
    """MỘT mẫu qua nhiều dump theo thời gian -> lưới: mỗi panel = một dump (tên, C-NLL box tốt nhất), box C-NLL thấp nhất đỏ đậm, K box
    còn lại đỏ nhạt. panels = [(tên, item của `score_samples`)]. Tiêu đề: mẫu, lớp, số vật, C-NLL box GT tốt nhất."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    nw, nh = g["wh"]
    ncol = min(ncol, len(panels))
    nrow = -(-len(panels) // ncol)
    pw = 3.6
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * pw, nrow * (pw * nh / nw + 0.45) + 1.3), squeeze=False)
    img = _image(g)
    for ax, (name, it) in zip(axes.ravel(), panels):
        best = int(np.argmin(it["c"]))
        _draw(ax, g, img, [it["boxes"][best]], [b for i, b in enumerate(it["boxes"]) if i != best])
        m = re.fullmatch(r"ep(\d+)", name)
        ax.set_title(f"{f'epoch {int(m.group(1))}' if m else name}  |  C-NLL {it['min']:.1f}", fontsize=10.5)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    k = len(panels[0][1]["boxes"])
    handles = [Line2D([], [], color="cyan", lw=1.5, label=f"{g['cls']} (existing)"),
               Line2D([], [], color="lime", lw=1.5, ls="--", label="GT empty spots" if "objects_all" in g else "GT holes"),
               Line2D([], [], color="red", lw=2.4, label="model: lowest C-NLL box"),
               Line2D([], [], color=(1.0, 0.25, 0.25, 0.45), lw=1.0, label=f"model: other {k - 1} boxes")]
    if "objects_all" in g:
        handles.insert(1, Line2D([], [], color="orange", lw=1.2, label=g["labels"][1][0]))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10.5)
    fig.suptitle(f"Add {g['cls']}  |  {len(g['objects'])} existing objects  |  best GT empty spot C-NLL {gt_cnll:.2f}", fontsize=12.5,
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.5 / fig.get_figheight(), 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _class(root, name):
    with open(os.path.join(root, "Anno", name + ".json")) as f:
        return clean_class(json.load(f)["class_name"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", nargs="+", help="<out>_boxes.json của eval.py; NHIỀU dump dạng TÊN=đường dẫn (cùng tập mẫu) -> chế độ giao")
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
    ap.add_argument("--intersect-pct", type=float, default=1.0,
                    help="nhiều dump: vẽ mẫu nằm trong p %% tệ / tốt nhất (round(n·p/100) mẫu) của MỌI dump, mỗi mẫu một lưới setting")
    ap.add_argument("--trend", action="store_true",
                    help="nhiều dump theo thứ tự thời gian: vẽ --top mẫu có C-NLL box tốt nhất tăng mạnh nhất từ dump đầu tới cuối")
    ap.add_argument("--select", type=int, default=5, help="số box model vẽ: C-NLL thấp nhất sau NMS")
    ap.add_argument("--nms-iou", type=float, default=0.5, help="NMS trước khi chọn: bỏ box IoU > ngưỡng với box C-NLL thấp hơn")
    ap.add_argument("--out", required=True, help="thư mục ghi hình + rank.json")
    a = ap.parse_args()
    if a.cocount_review and a.cocount_obj_filter is not None:
        sys.exit("--cocount-review thay cho --cocount-obj-filter: chọn một (giống lúc eval)")
    a.key = a.key or ("inpainted" if a.dataset == "ce130" else "cocount")
    dumps = []
    for spec in a.dump:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.basename(spec), spec)
        with open(path) as f:
            res = json.load(f)["results"]
        if a.key not in res:
            sys.exit(f"{path}: không có khoá {a.key} (có: {sorted(res)})")
        dumps.append((name, path, res[a.key]))
    rows = dumps[0][2]
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
    os.makedirs(a.out, exist_ok=True)
    meta = {"dataset": a.dataset, "key": a.key, "rank_by": a.rank_by, "order": a.order, "review": a.cocount_review,
            "obj_filter": a.cocount_obj_filter, "select": a.select, "nms_iou": a.nms_iou}
    if a.trend:
        return trend(a, dumps, geo, meta)
    if len(dumps) > 1:
        return intersect(a, dumps, geo, meta)
    items = score_samples(rows, geo, a.select, a.nms_iou)
    items.sort(key=lambda d: d[a.rank_by], reverse=a.order == "worst")
    for r, it in enumerate(items, 1):
        it["rank"] = r
    print(f"{a.key}: {len(items)} / {len(rows)} mẫu có C-NLL | TB box tốt nhất / ảnh {np.mean([d['min'] for d in items]):.3f} (= bảng) | "
          f"trung vị {np.median([d['min'] for d in items]):.3f}")
    with open(os.path.join(a.out, "rank.json"), "w") as f:
        json.dump({"dump": dumps[0][1], **meta,
                   "samples": [{k: it[k] for k in ("rank", "image_id", "median", "min", "gt_median", "n_obj", "sel", "sel_cnll")}
                               for it in items]}, f, indent=1)
    top = math.ceil(len(items) * a.top_pct / 100) if a.top_pct is not None else a.top
    w = max(2, len(str(top)))
    for it in items[:top]:
        plot_sample(it, geo[it["image_id"]], os.path.join(a.out, f"{a.order}_{it['rank']:0{w}d}_{it['image_id']}.png"))
    print(f"-> {min(top, len(items))} hình + rank.json trong {a.out}")


def intersect(a, dumps, geo, meta):
    """Nhiều dump (cùng tập mẫu): mẫu nằm trong `--intersect-pct` % tệ / tốt nhất của MỌI dump -> mỗi mẫu một lưới (GT + từng setting)."""
    scored = []
    for name, _, rows in dumps:
        items = {it["image_id"]: it for it in score_samples(rows, geo, a.select, a.nms_iou)}
        order = sorted(items, key=lambda k: items[k][a.rank_by], reverse=a.order == "worst")
        scored.append((name, items, {k: r for r, k in enumerate(order, 1)}))
    n = len(scored[0][1])
    k = int(round(n * a.intersect_pct / 100))
    common = set.intersection(*(set(k_ for k_, r in rk.items() if r <= k) for _, _, rk in scored))
    common = sorted(common, key=lambda i: np.mean([rk[i] for _, _, rk in scored]))
    print(f"{a.key}: {n} mẫu, {a.order} {a.intersect_pct:g} % = {k} mẫu / setting | giao {len(scored)} setting: {len(common)} mẫu")
    out = []
    for j, iid in enumerate(common, 1):
        g = geo[iid]
        gt = cnll(g["holes"], g["objects"], tuple(g["wh"]),
                  others=other_class(g["objects"], g["objects_all"]) if "objects_all" in g else None)
        plot_grid(g, float(gt.min()), [(name, items[iid]) for name, items, _ in scored],
                  os.path.join(a.out, f"{a.order}_{j:02d}_{iid}.png"))
        out.append({"image_id": iid, "gt_cnll": float(gt.min()), "n_obj": len(g["objects"]),
                    "settings": {name: {"rank": rk[iid], "min": items[iid]["min"], "sel_cnll": items[iid]["sel_cnll"]}
                                 for name, items, rk in scored}})
    with open(os.path.join(a.out, "intersect.json"), "w") as f:
        json.dump({"dumps": {name: path for name, path, _ in dumps}, **meta, "intersect_pct": a.intersect_pct, "per_setting": k,
                   "samples": out}, f, indent=1)
    print(f"-> {len(common)} hình + intersect.json trong {a.out}")



def trend(a, dumps, geo, meta):
    """Nhiều dump theo thời gian (cùng tập mẫu): xếp mẫu theo C-NLL box tốt nhất (dump cuối − dump đầu) giảm dần -> `--top` lưới."""
    if len(dumps) < 2:
        sys.exit("--trend cần >= 2 dump theo thứ tự thời gian (TÊN=đường dẫn)")
    scored = [(name, {it["image_id"]: it for it in score_samples(rows, geo, a.select, a.nms_iou)}) for name, _, rows in dumps]
    ids = sorted(set.intersection(*(set(items) for _, items in scored)))
    m = np.array([[items[i]["min"] for _, items in scored] for i in ids])
    rank = lambda v: np.argsort(np.argsort(v, axis=-1), axis=-1).astype(float)  # noqa: E731
    r = rank(m) - (len(dumps) - 1) / 2
    t = np.arange(len(dumps)) - (len(dumps) - 1) / 2
    rho = (r @ t) / np.sqrt((r ** 2).sum(1) * (t ** 2).sum() + 1e-12)            # tương quan hạng với thứ tự dump
    rise = m[:, -1] - m[:, 0]
    order = np.argsort(-rise)
    print(f"{a.key}: {len(ids)} mẫu đủ {len(dumps)} dump | C-NLL tăng > 10: {(rise > 10).sum()}, giảm > 10: {(rise < -10).sum()} | "
          f"TB theo dump: {' '.join(f'{v:.1f}' for v in m.mean(0))}")
    samples = []
    for j, q in enumerate(order, 1):
        iid = ids[q]
        d = {"rank": j, "image_id": iid, "rise": float(rise[q]), "rho": float(rho[q]), "min": dict(zip([n for n, _ in scored], m[q].tolist()))}
        if j <= a.top:
            g = geo[iid]
            gt = cnll(g["holes"], g["objects"], tuple(g["wh"]),
                      others=other_class(g["objects"], g["objects_all"]) if "objects_all" in g else None)
            d.update(gt_cnll=float(gt.min()), n_obj=len(g["objects"]))
            plot_trend(g, d["gt_cnll"], [(name, items[iid]) for name, items in scored],
                       os.path.join(a.out, f"trend_{j:02d}_{iid}.png"))
        samples.append(d)
    with open(os.path.join(a.out, "trend.json"), "w") as f:
        json.dump({"dumps": {name: path for name, path, _ in dumps}, **meta, "samples": samples}, f, indent=1)
    print(f"-> {min(a.top, len(ids))} hình + trend.json trong {a.out}")


if __name__ == "__main__":
    main()
