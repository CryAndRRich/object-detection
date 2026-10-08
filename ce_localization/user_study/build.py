#!/usr/bin/env python3
"""Gộp box thô của các model (`eval.py --dump-boxes`) -> `items.json` cho web chấm (docs/EXPERIMENT_GAMMA.md mục 17).

Mỗi mẫu test (nhánh, lượt t; ảnh inpaint `samples/`): box của từng model quy về PIXEL ẢNH GỐC rồi chọn K = 4 box
(`selection.py`, chung mọi model, không nhìn lỗ / box vật). Ghi kèm box vật đang có (hiện cho người chấm) và lỗ GT = box các vật
đã xoá tới lượt t (KHÔNG hiện — chỉ để `score.py` phân tích).
Màn = (model, mẫu), mã `<model>|<image_id>` (màn chấm lại thêm `|repeat`). Mỗi model một dãy màn RIÊNG: khoá thứ tự ngẫu nhiên theo
RNG của (`--seed`, mã model) ⇒ web chấm được từng model / vài model (nhiều model thì các dãy xen ngẫu nhiên), dừng giữa chừng vẫn là
mẫu ngẫu nhiên, và THÊM MODEL SAU (`--add-to`) không đổi màn / thứ tự / nhãn đã chấm của model cũ (người dùng, 2026-10-04: sau này
còn thử nghiệm khác). `--repeat` phần mẫu mỗi model được chấm lại, khoá thứ tự sau bản gốc >= `--min-gap` (phần của dãy).
Không nạp model, không GPU (đọc JSON, vài chục giây).

  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  O=/mnt/disk1/aiotlab/haitn/output/gamma/user_study
  python user_study/build.py --turn-index ../data/turn_index.json --out $O/items.json \\
      --model paper        $O/boxes_paper.json        inpainted      "CE-Loc (paper)" \\
      --model gamma2_celoc $O/boxes_gamma2_celoc.json inpainted      "CE-Loc (masked SpatialSoftmax)" \\
      --model gamma2       $O/boxes_gamma2.json       inpainted_t100 "CE-Loc (frozen) + Refiner" \\
      --model gamma3       $O/boxes_gamma3.json       inpainted_t100 "CE-Loc (frozen) + Refiner + object-box geometry" \\
      --model gamma4       $O/boxes_gamma4.json       inpainted      "CE-Loc + object-box attention"
Tên hiển thị (mục 13.0) hiện trên web; mã model (vd gamma4) chỉ dùng trong ratings.jsonl / score.json.
Thêm model vào items.json đã có (cùng tập mẫu, màn cũ giữ nguyên):
  python user_study/build.py --turn-index ../data/turn_index.json --add-to $O/items.json \\
      --model gamma5 $O/boxes_gamma5.json inpainted "<tên hiển thị>"

CE-CoCount (`--cocount-root`, dump của `eval.py --dataset cocount`; mục 17 + 18): mẫu = file `Anno/<tên>.json` (ảnh gốc
`Image/<tên>.jpg`, t = 0), box vật hiện = vật CÙNG lớp (`objects`) + vật lớp kia của cùng frame (`objects_other`, web vẽ nét đứt), lỗ GT
= 10 `loc_bbox` (không hiện). Khoá `cocount_objsize` (`--obj-size`) = bộ "box resize" — dựng thành items.json RIÊNG:
  O=../../output/gamma/user_study_cocount
  python user_study/build.py --cocount-root ../data/cocount --out $O/items.json \\
      --model paper $O/boxes_paper.json cocount "CE-Loc (paper)" ...
  python user_study/build.py --cocount-root ../data/cocount --out $O/items_objsize.json \\
      --model paper $O/boxes_paper.json cocount_objsize "CE-Loc (paper)" ...
"""

import argparse
import json
import os
import sys
import zlib

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import read_cocount  # noqa: E402
from ce_localization.data.turns import TurnIndex  # noqa: E402
from ce_localization.user_study.selection import K, select_boxes  # noqa: E402

__all__ = ["load_dump", "to_image_px", "screen_order", "cocount_item", "build_items"]


def load_dump(path, key):
    """-> (meta, {image_id: record}) của khoá kết quả `key` (vd `inpainted`, `inpainted_t100`)."""
    with open(path) as f:
        d = json.load(f)
    if key not in d["results"]:
        raise KeyError(f"{path}: không có khoá {key!r} (có {sorted(d['results'])})")
    meta = {k: d.get(k) for k in ("ckpt", "iter", "split", "n_samples", "seed")}
    return meta, {r["image_id"]: r for r in d["results"][key]}


def to_image_px(boxes, wh_canvas, wh_image):
    """Box xyxy pixel canvas (vùng ảnh thật (nw, nh)) -> pixel ảnh gốc (W, H)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    sx, sy = wh_image[0] / wh_canvas[0], wh_image[1] / wh_canvas[1]
    return b * np.array([sx, sy, sx, sy])


def screen_order(image_ids, model, seed=0, repeat=0.1, min_gap=0.05):
    """Dãy màn của MỘT model -> list {id, image_id, model, order, repeat_of}. RNG theo (seed, mã model) ⇒ không phụ thuộc model
    khác. `order` ∈ [0, 1) xáo mẫu; màn chấm lại có order = order bản gốc + U(min_gap, 1) (luôn sau bản gốc)."""
    rng = np.random.default_rng([seed, zlib.crc32(model.encode())])
    keys = rng.random(len(image_ids))
    out = [{"id": f"{model}|{i}", "image_id": i, "model": model, "order": float(k), "repeat_of": None}
           for i, k in zip(image_ids, keys)]
    n_rep = int(round(repeat * len(image_ids)))
    for j in sorted(rng.choice(len(image_ids), size=n_rep, replace=False).tolist()) if n_rep else []:
        i = image_ids[j]
        out.append({"id": f"{model}|{i}|repeat", "image_id": i, "model": model,
                    "order": float(keys[j] + rng.uniform(min_gap, 1.0)), "repeat_of": f"{model}|{i}"})
    return out


def _item(index, iid):
    e = index.turns[iid]
    b = index.branches[e["branch"]]
    t = e["t"]
    rm = np.asarray(b["removed"], dtype=int)
    objs = np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)[(rm == 0) | (rm > t)]
    return {"t": t, "class": b["class"], "image": e["sample"], "wh": b["wh"], "objects": objs.round(2).tolist(),
            "holes": [list(map(float, h)) for h in b["holes"][:t]], "models": {}}


def cocount_item(root, iid, max_obj_ratio=None):
    """Mẫu CE-CoCount `iid` (tên file) -> item: ảnh gốc, t = 0, vật cùng lớp + vật lớp kia (pixel ảnh gốc, lọc box SAM khổng lồ nếu
    `max_obj_ratio`), lỗ = 10 loc_bbox."""
    r = read_cocount(root, iid, max_obj_ratio)
    image = f"Image/{iid}.jpg"
    with Image.open(os.path.join(root, image)) as im:
        wh = list(im.size)
    n = len(r["objects"])
    return {"t": 0, "class": r["class"], "image": image, "wh": wh, "objects": r["objects"].round(2).tolist(),
            "objects_other": r["objects_all"][n:].round(2).tolist(), "holes": r["loc"].tolist(), "models": {}}


def build_items(index, dumps, k=K, seed=0, repeat=0.1, min_gap=0.05, base=None, log=print, cocount_root=None, max_obj_ratio=None):
    """dumps: list (model_id, meta, {image_id: record}) -> dict items.json. `base` (items.json đã có): THÊM các model vào, giữ nguyên
    tập mẫu + màn cũ; dump mới phải có đủ mọi mẫu của `base`. `cocount_root`: mẫu CE-CoCount (`index` bỏ qua)."""
    if base is not None:
        k, seed, repeat, min_gap = base["k"], base["seed"], base["repeat"], base["min_gap"]
        clash = [m for m, _, _ in dumps if m in base["models"]]
        if clash:
            raise ValueError(f"model đã có trong items.json: {clash}")
        ids = set(base["items"])
        for mid, _, recs in dumps:
            miss = ids - set(recs)
            if miss:
                raise ValueError(f"{mid}: thiếu {len(miss)} mẫu của items.json (vd {sorted(miss)[:3]})")
        out = base
    else:
        ids = set.intersection(*(set(r) for _, _, r in dumps))
        for mid, _, recs in dumps:
            if len(recs) != len(ids):
                log(f"  [cảnh báo] {mid}: {len(recs)} mẫu, chỉ giữ {len(ids)} mẫu chung mọi model")
        if not ids:
            raise ValueError("các dump không có mẫu chung")
        make = (lambda i: cocount_item(cocount_root, i, max_obj_ratio)) if cocount_root else (lambda i: _item(index, i))
        out = {"k": k, "seed": seed, "repeat": repeat, "min_gap": min_gap, "dataset": "cocount" if cocount_root else "ce130",
               "models": {}, "items": {iid: make(iid) for iid in sorted(ids)}, "screens": [],
               **({"cocount_obj_filter": max_obj_ratio} if cocount_root else {})}
    for mid, meta, recs in dumps:
        for iid, it in out["items"].items():
            r = recs[iid]
            if r["t"] != it["t"]:
                raise ValueError(f"{mid} {iid}: lượt {r['t']} != chỉ mục {it['t']}")
            it["models"][mid] = select_boxes(to_image_px(r["boxes"], r["wh"], it["wh"]), it["wh"], k=k)
        out["models"][mid] = meta
        out["screens"] += screen_order(sorted(out["items"]), mid, seed=seed, repeat=repeat, min_gap=min_gap)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--cocount-root", default=None, help="CE-CoCount (Image/ Anno/ Anno_with_exam_bbox/): dump của eval.py "
                                                       "--dataset cocount; bỏ qua --turn-index")
    ap.add_argument("--cocount-obj-filter", type=float, default=None, help="CE-CoCount: bỏ box vật > K × cạnh TB exemplar (như eval.py)")
    ap.add_argument("--model", nargs=4, action="append", required=True, metavar=("ID", "DUMP", "KEY", "NAME"),
                    help="mã model, file --dump-boxes, khoá kết quả (vd inpainted / inpainted_t100), tên hiển thị trên web "
                         "(docs/EXPERIMENT_GAMMA.md mục 13.0)")
    ap.add_argument("--add-to", default=None, help="items.json đã có: thêm model vào (giữ màn + nhãn cũ), ghi đè chính file đó")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=float, default=0.1, help="phần mẫu mỗi model được chấm lại (đo độ nhất quán)")
    ap.add_argument("--min-gap", type=float, default=0.05, help="màn chấm lại đứng sau bản gốc >= phần này của dãy màn")
    ap.add_argument("--out", default=None, help="items.json mới (không dùng với --add-to)")
    a = ap.parse_args()
    if bool(a.out) == bool(a.add_to):
        sys.exit("cần đúng một trong --out (dựng mới) / --add-to (thêm model)")
    if len({m[0] for m in a.model}) != len(a.model):
        sys.exit("mã model trùng nhau")
    base = None
    if a.add_to:
        with open(a.add_to) as f:
            base = json.load(f)
    if base and (base.get("dataset", "ce130") == "cocount") != bool(a.cocount_root):
        sys.exit(f"items.json là {base.get('dataset', 'ce130')}: --cocount-root phải {'có' if not a.cocount_root else 'bỏ'}")
    index = None if a.cocount_root else TurnIndex(a.turn_index)
    dumps = []
    for mid, path, key, name in a.model:
        meta, recs = load_dump(path, key)
        print(f"  {mid} ({name}): {path} [{key}] {len(recs)} mẫu, {meta['n_samples']} box / mẫu, iter {meta['iter']}", flush=True)
        dumps.append((mid, {**meta, "dump": path, "key": key, "name": name}, recs))
    splits = {d[1]["split"] for d in dumps} | ({m["split"] for m in base["models"].values()} if base else set())
    if len(splits) != 1:
        sys.exit(f"dump khác split: {sorted(splits)}")
    out = build_items(index, dumps, k=a.k, seed=a.seed, repeat=a.repeat, min_gap=a.min_gap, base=base, cocount_root=a.cocount_root,
                      max_obj_ratio=a.cocount_obj_filter)
    path = a.add_to or a.out
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f)
    os.replace(tmp, path)
    k = out["k"]
    for m in out["models"]:
        sc = [s for s in out["screens"] if s["model"] == m]
        n_rep = sum(s["repeat_of"] is not None for s in sc)
        short = sum(len(it["models"][m]["idx"]) < k for it in out["items"].values())
        div = float(np.mean([it["models"][m]["n_distinct"] for it in out["items"].values()]))
        print(f"   {m:14s} {len(sc) - n_rep} màn + {n_rep} chấm lại | mẫu thiếu < {k} box: {short} | "
              f"box khác biệt TB / 30 mẫu (NMS 0,3): {div:.1f}", flush=True)
    print(f"-> {path}: {len(out['items'])} mẫu × {len(out['models'])} model", flush=True)


if __name__ == "__main__":
    main()
