#!/usr/bin/env python3
"""Chấm lại CE-CoCount từ box dump của `eval.py --dataset cocount --dump-boxes` — KHÔNG model, KHÔNG GPU.

Dựng lại record của từng mẫu đúng như lúc eval: box thô từ dump (pixel canvas, mọi khoá kể cả `_noobj` / `_norefine` / `_objsize*` —
box `_objsize*` trong dump đã đổi cỡ), lỗ / box vật / exemplar từ `data/cocount.read_cocount` (cùng `--cocount-obj-filter`) qua cùng
letterbox; `prior` dựng lại bằng `prior_records` cùng seed + cùng thứ tự mẫu (thứ tự của dump) ⇒ trùng prior của eval. Rồi gọi
`engine/add_eval.add_metrics` HIỆN TẠI — dùng để thêm chỉ số mới cho các lần eval cũ (2026-10-17: C-NLL `F1all`, docs/EXPERIMENT_DELTA.md
mục 10.9) mà không chạy lại model.

`--check`: nếu cạnh dump có `test_<run>.json` (dump tên `boxes_<run>.json`) thì mọi chỉ số CŨ của mọi khoá (model + prior) phải khớp
(box dump làm tròn 0,01 px ⇒ sai số nhỏ; C-NLL là trung vị nên lệch rất ít) — báo khoá / chỉ số lệch quá ngưỡng.

  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  O=/mnt/disk1/aiotlab/haitn/output/add/cocount/filt3
  python tools/rescore_cocount.py $O/boxes_*.json --cocount-root ../data/cocount --cocount-obj-filter 3 \\
      --turn-index ../data/turn_index.json --check        # ghi $O/rescore_<run>.json cạnh mỗi dump
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import read_cocount  # noqa: E402
from ce_localization.data.dataset import scale_boxes  # noqa: E402
from ce_localization.data.turns import TurnIndex  # noqa: E402
from ce_localization.engine.add_eval import add_metrics, prior_records, prior_unit_boxes, resize_to_objects  # noqa: E402

__all__ = ["cocount_geometry", "rebuild_records", "size_stat", "rescore", "compare"]

CHECK_TOL = {"cnll": 0.02, "default": 1e-3}       # sai số cho phép khi so chỉ số cũ (C-NLL tương đối, còn lại tuyệt đối)


def cocount_geometry(root, name, image_size=512, max_obj_ratio=None):
    """-> dict lỗ / box vật / box vật cả hai lớp / exemplar (pixel canvas) + (nw, nh) của mẫu `name` — y `CoCountAddDataset`."""
    r = read_cocount(root, name, max_obj_ratio)
    with Image.open(os.path.join(root, "Image", name + ".jpg")) as im:
        w, h = im.size
    scale = min(image_size / w, image_size / h)
    nw, nh = int(w * scale), int(h * scale)
    sb = lambda b: scale_boxes(b, scale, nw, nh).astype(np.float32).astype(np.float64)  # noqa: E731  (eval đi qua tensor float32)
    return {"holes": sb(r["loc"]), "objects": sb(r["objects"]), "objects_all": sb(r["objects_all"]),
            "exemplars": sb(r["exemplars"]), "wh": np.array([nw, nh], dtype=np.float64)}


def rebuild_records(rows, geo):
    """rows = `dump["results"][khoá]` (image_id, t, wh, boxes), geo = {image_id: `cocount_geometry`} -> record của `add_metrics`."""
    out = []
    for x in rows:
        g = geo[x["image_id"]]
        if not np.allclose(g["wh"], x["wh"], atol=1e-3):
            raise ValueError(f"{x['image_id']}: vùng ảnh {g['wh'].tolist()} khác dump {x['wh']} — sai --image-size / ảnh?")
        out.append({**g, "image_id": x["image_id"], "t": int(x["t"]), "boxes": np.asarray(x["boxes"], dtype=np.float64)})
    return out


def size_stat(key):
    """Khoá `..._objsize` -> "mean", `..._objsize_<stat>` -> stat, khoá khác -> None (box không đổi cỡ)."""
    m = re.search(r"_objsize(?:_(\w+))?$", key)
    return None if m is None else (m.group(1) or "mean")


def rescore(dump, geo, prior_unit=None):
    """-> {"results": {khoá: chỉ số}, "prior": {khoá: chỉ số}} (prior chỉ khi có `prior_unit`). Prior dựng trên record khoá gốc."""
    K, seed = dump["n_samples"], dump.get("seed", 0)
    out = {"results": {}, "prior": {}}
    base_prior = None
    for key, rows in dump["results"].items():
        rec = rebuild_records(rows, geo)
        out["results"][key] = add_metrics(rec, with_holes=True, latest=False)
        if prior_unit is None:
            continue
        if base_prior is None:                        # prior chỉ phụ thuộc (wh, thứ tự) ⇒ mọi khoá dùng chung bộ lỗ train
            base_prior = prior_records(rec, prior_unit, K, seed=seed)
        st = size_stat(key)
        out["prior"][key] = add_metrics(base_prior if st is None else resize_to_objects(base_prior, st),
                                        with_holes=True, latest=False)
    return out


def compare(new, old):
    """Chỉ số CŨ (có trong `old`) lệch quá `CHECK_TOL` -> list chuỗi mô tả; [] = khớp."""
    bad = []
    for part in ("results", "prior"):
        for key, res in old.get(part, {}).items():
            if key not in new.get(part, {}):
                continue
            for k, v in res.items():
                nv = new[part][key].get(k)
                if not isinstance(v, (int, float)) or nv is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                tol = CHECK_TOL["cnll"] * max(abs(v), 1.0) if k.startswith("cnll") else CHECK_TOL["default"]
                if abs(nv - v) > tol:
                    bad.append(f"{part}/{key}/{k}: cũ {v:.4f} mới {nv:.4f}")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+", help="boxes_<run>.json của eval.py --dump-boxes (CE-CoCount)")
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--cocount-obj-filter", type=float, default=None, help="PHẢI giống lúc eval (filt3: 3)")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--turn-index", default=None, help="có thì dựng lại prior (lỗ train CE-130)")
    ap.add_argument("--prior-source", default="samples", help="data.split_source của checkpoint (GAMMA2+ / DELTA: samples)")
    ap.add_argument("--check", action="store_true", help="so chỉ số cũ với test_<run>.json cạnh dump")
    a = ap.parse_args()
    prior_unit = prior_unit_boxes(TurnIndex(a.turn_index), "train", a.prior_source) if a.turn_index else None
    geo, n_bad = {}, 0
    for path in a.dumps:
        t0 = time.time()
        with open(path) as f:
            dump = json.load(f)
        if not dump["results"] or not all(k.startswith("cocount") for k in dump["results"]):
            sys.exit(f"{path}: không phải dump CE-CoCount (khoá {list(dump['results'])})")
        for x in next(iter(dump["results"].values())):
            if x["image_id"] not in geo:
                geo[x["image_id"]] = cocount_geometry(a.cocount_root, x["image_id"], a.image_size, a.cocount_obj_filter)
        res = rescore(dump, geo, prior_unit)
        res.update(dump=path, ckpt=dump.get("ckpt"), iter=dump.get("iter"), cocount_obj_filter=a.cocount_obj_filter)
        base = os.path.basename(path)
        run = base[len("boxes_"):-len(".json")] if base.startswith("boxes_") else base[:-len(".json")]
        out = os.path.join(os.path.dirname(os.path.abspath(path)), f"rescore_{run}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"== {run} ({len(dump['results'])} khoá, {time.time() - t0:.0f}s) -> {out}")
        for key, r in res["results"].items():
            pr = res["prior"].get(key, {})
            print(f"   {key:36s} C-NLL F1 / F1all n1 median {r['cnll_F1_n1_median']:8.3f} / {r['cnll_F1all_n1_median']:8.3f}"
                  + (f"  | prior {pr['cnll_F1_n1_median']:8.3f} / {pr['cnll_F1all_n1_median']:8.3f}" if pr else ""))
        old_p = os.path.join(os.path.dirname(os.path.abspath(path)), f"test_{run}.json")
        if a.check:
            if not os.path.exists(old_p):
                print(f"   (không có {old_p} để kiểm)")
                continue
            with open(old_p) as f:
                bad = compare(res, json.load(f))
            n_bad += len(bad)
            print(f"   kiểm với {os.path.basename(old_p)}: " + ("khớp" if not bad else f"{len(bad)} chỉ số LỆCH"))
            for b in bad[:20]:
                print("     " + b)
    if n_bad:
        sys.exit(f"!!! {n_bad} chỉ số cũ không khớp — đừng dùng số F1all trước khi tìm ra lý do")


if __name__ == "__main__":
    main()
