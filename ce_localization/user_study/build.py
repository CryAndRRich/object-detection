#!/usr/bin/env python3
"""Gộp box thô của các model (`eval.py --dump-boxes`) -> `items.json` cho web chấm (docs/EXPERIMENT_GAMMA.md mục 17).

Mỗi mẫu test (nhánh, lượt t; ảnh inpaint `samples/`): box của từng model quy về PIXEL ẢNH GỐC rồi chọn K = 4 box
(`selection.py`, chung mọi model, không nhìn lỗ / box vật). Ghi kèm box vật đang có (hiện cho người chấm) và lỗ GT = box các vật
đã xoá tới lượt t (KHÔNG hiện — chỉ để `score.py` phân tích). Thứ tự màn = mọi cặp (mẫu, model) xáo theo `--seed` (dừng
giữa chừng vẫn là mẫu ngẫu nhiên); `--repeat` phần màn được chấm lại ở vị trí sau bản gốc >= `--min-gap` màn (đo độ nhất quán).
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
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.turns import TurnIndex  # noqa: E402
from ce_localization.user_study.selection import K, select_boxes  # noqa: E402

__all__ = ["load_dump", "to_image_px", "screen_order", "build_items"]


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


def screen_order(image_ids, models, seed=0, repeat=0.1, min_gap=50):
    """Mọi cặp (image_id, model) xáo ngẫu nhiên + `repeat` phần cặp chấm lại -> list {image_id, model, repeat_of}
    (`repeat_of` = vị trí màn gốc, luôn đứng trước >= min_gap màn nếu đủ chỗ)."""
    rng = np.random.default_rng(seed)
    pairs = [(i, m) for i in image_ids for m in models]
    pairs = [pairs[j] for j in rng.permutation(len(pairs))]
    n = len(pairs)
    keyed = [(float(p), p, None) for p in range(n)]
    for p in sorted(rng.choice(n, size=int(round(repeat * n)), replace=False).tolist()) if n else []:
        lo = min(p + min_gap, n)
        keyed.append((float(rng.uniform(lo, n + 1)), p, p))
    keyed.sort(key=lambda x: x[0])
    pos = {}
    out = []
    for s, (_, p, rep) in enumerate(keyed):
        if rep is None:
            pos[p] = s
    for _, p, rep in keyed:
        i, m = pairs[p]
        out.append({"image_id": i, "model": m, "repeat_of": None if rep is None else pos[rep]})
    return out


def build_items(index, dumps, k=K, seed=0, repeat=0.1, min_gap=50, log=print):
    """dumps: list (model_id, meta, {image_id: record}) -> dict items.json."""
    ids = set.intersection(*(set(r) for _, _, r in dumps))
    for mid, _, recs in dumps:
        if len(recs) != len(ids):
            log(f"  [cảnh báo] {mid}: {len(recs)} mẫu, chỉ giữ {len(ids)} mẫu chung mọi model")
    if not ids:
        raise ValueError("các dump không có mẫu chung")
    items = {}
    for iid in sorted(ids):
        e = index.turns[iid]
        b = index.branches[e["branch"]]
        t = e["t"]
        rm = np.asarray(b["removed"], dtype=int)
        objs = np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)[(rm == 0) | (rm > t)]
        it = {"t": t, "class": b["class"], "image": e["sample"], "wh": b["wh"], "objects": objs.round(2).tolist(),
              "holes": [list(map(float, h)) for h in b["holes"][:t]], "models": {}}
        for mid, _, recs in dumps:
            r = recs[iid]
            if r["t"] != t:
                raise ValueError(f"{mid} {iid}: lượt {r['t']} != chỉ mục {t}")
            it["models"][mid] = select_boxes(to_image_px(r["boxes"], r["wh"], b["wh"]), b["wh"], k=k)
        items[iid] = it
    screens = screen_order(sorted(ids), [m for m, _, _ in dumps], seed=seed, repeat=repeat, min_gap=min_gap)
    return {"k": k, "seed": seed, "models": {m: meta for m, meta, _ in dumps}, "items": items, "screens": screens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--model", nargs=4, action="append", required=True, metavar=("ID", "DUMP", "KEY", "NAME"),
                    help="mã model, file --dump-boxes, khoá kết quả (vd inpainted / inpainted_t100), tên hiển thị trên web "
                         "(docs/EXPERIMENT_GAMMA.md mục 13.0)")
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--repeat", type=float, default=0.1, help="phần màn chấm lại để đo độ nhất quán")
    ap.add_argument("--min-gap", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len({m[0] for m in a.model}) != len(a.model):
        sys.exit("mã model trùng nhau")
    index = TurnIndex(a.turn_index)
    dumps = []
    for mid, path, key, name in a.model:
        meta, recs = load_dump(path, key)
        print(f"  {mid} ({name}): {path} [{key}] {len(recs)} mẫu, {meta['n_samples']} box / mẫu, iter {meta['iter']}", flush=True)
        dumps.append((mid, {**meta, "dump": path, "key": key, "name": name}, recs))
    if len({d[1]["split"] for d in dumps}) != 1:
        sys.exit(f"dump khác split: {[d[1]['split'] for d in dumps]}")
    out = build_items(index, dumps, k=a.k, seed=a.seed, repeat=a.repeat, min_gap=a.min_gap)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f)
    n_rep = sum(s["repeat_of"] is not None for s in out["screens"])
    short = {m: sum(len(it["models"][m]["idx"]) < a.k for it in out["items"].values()) for m in out["models"]}
    div = {m: float(np.mean([it["models"][m]["n_distinct"] for it in out["items"].values()])) for m in out["models"]}
    print(f"-> {a.out}: {len(out['items'])} mẫu × {len(out['models'])} model = {len(out['screens']) - n_rep} màn + {n_rep} chấm lại"
          f"\n   mẫu thiếu < {a.k} box hợp lệ: {short}\n   số box khác biệt TB / 30 mẫu (NMS 0,3): "
          + " | ".join(f"{m} {v:.1f}" for m, v in div.items()), flush=True)


if __name__ == "__main__":
    main()
