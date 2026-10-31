#!/usr/bin/env python3
"""Đổi JSON eval cũ sang C-NLL mới (2026-10-29: trừ NLL tại TÂM μ thay vì min trên vật) — KHÔNG model, KHÔNG GPU, sửa TẠI CHỖ.

C-NLL mới = C-NLL cũ + ½·min_i D²(z_i) (D² = Mahalanobis² của vật z_i tới μ; chỉ phụ thuộc ảnh, ∈ [0, 2] vì tổng D² của M vật = 4M).
Mỗi `<x>.json` của `eval.py` cần `<x>_boxes.json` cạnh nó (để biết ảnh nào có C-NLL): dump cộng lượng của ảnh vào `cnll` từng box +
`cnll_gt`; file tổng cộng TB lượng đó trên đúng tập ảnh có C-NLL vào mọi khoá `cnll` / `cnll_gt` (cả `prior`, `*_objsize*` — cùng tập
ảnh, kiểm `n_cnll`). Ghi `cnll_calib: "mu"` vào cả hai file; file đã có thì bỏ qua. Dựng lại hình học như lúc eval (CE-130 ảnh inpaint:
`ce130_geometry`; CE-CoCount: `--cocount-review` / `--cocount-obj-filter` PHẢI giống lúc eval, kiểm số vật với `n_obj` của dump).
Khoá ảnh gốc CE-130 (`original*`) chưa hỗ trợ -> báo lỗi.

  cd object-detection/ce_localization
  python tools/rescore_cnll.py ../../output/add/snapshots/delta8/*/ep0??.json --cocount-review ../../output/add/box_review/cocount.jsonl
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import cocount_geometry, load_review  # noqa: E402
from ce_localization.data.turns import TurnIndex  # noqa: E402
from ce_localization.engine.add_eval import CNLL_MIN_OBJ, _cnll_fit, _features, other_class  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cnll_worst import ce130_geometry  # noqa: E402

__all__ = ["calib_shift", "shift_for"]


def calib_shift(objects, wh, others=None):
    """½·min_i D²(z_i) của ảnh (C-NLL mới − C-NLL cũ), None nếu < CNLL_MIN_OBJ vật."""
    if len(objects) < CNLL_MIN_OBJ:
        return None
    ref = objects if others is None or not len(others) else np.concatenate([objects, others])
    mu, inv = _cnll_fit(objects, ref, wh)
    d = _features(objects, ref, wh, is_objects=True) - mu
    return float(0.5 * np.einsum("ni,ij,nj->n", d, inv, d).min())


def shift_for(key, means):
    """(số ảnh, lượng dời TB) cho khoá `key` của file tổng từ `means` {khoá dump: (n, m)}. Khoá đổi cỡ (`*_objsize*`) và khoá `prior`
    không hậu tố sampler (`inpainted` khi dump có `inpainted_mock` / `_ddim`) dùng lại khoá cùng gốc: cùng tập ảnh ⇒ cùng lượng dời.
    Không có -> (None, None)."""
    if key in means:
        return means[key]
    root = key.split("_objsize")[0]
    for k, v in means.items():
        if k == root or k.startswith(root + "_"):
            return v
    return None, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("summary", nargs="+", help="<x>.json của eval.py (cần <x>_boxes.json cạnh nó)")
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--samples-root", default="../data/samples")
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--cocount-review", default=None)
    ap.add_argument("--cocount-obj-filter", type=float, default=None)
    ap.add_argument("--image-size", type=int, default=512)
    a = ap.parse_args()
    index, review, cache = None, None, {}

    def shift(key, x):
        nonlocal index, review
        ck = ("cocount" if key.startswith("cocount") else "ce130", x["image_id"])
        if ck not in cache:
            if ck[0] == "ce130":
                if not key.startswith("inpainted"):
                    sys.exit(f"khoá {key}: chỉ hỗ trợ CE-130 ảnh inpaint / CE-CoCount")
                index = index or TurnIndex(a.turn_index)
                g = ce130_geometry(index, a.samples_root, x["image_id"], a.image_size)
            else:
                if a.cocount_review and review is None:
                    review = load_review(a.cocount_review)
                g = cocount_geometry(a.cocount_root, x["image_id"], a.image_size, a.cocount_obj_filter, review)
            if not np.allclose(g["wh"], x["wh"], atol=1e-3) or len(g["objects"]) != x["n_obj"]:
                sys.exit(f"{x['image_id']}: hình học khác dump (wh / số vật) — sai --cocount-review / --cocount-obj-filter?")
            oth = other_class(g["objects"], g["objects_all"]) if "objects_all" in g else None
            cache[ck] = calib_shift(g["objects"], tuple(g["wh"]), oth)
        return cache[ck]

    for path in a.summary:
        bpath = path[:-5] + "_boxes.json"
        with open(path) as f:
            summ = json.load(f)
        with open(bpath) as f:
            dump = json.load(f)
        if summ.get("cnll_calib") == "mu" or dump.get("cnll_calib") == "mu":
            print(f"{path}: đã là C-NLL mới, bỏ qua")
            continue
        means = {}
        for key, rows in dump["results"].items():
            sh = []
            for x in rows:
                if not x.get("cnll"):
                    continue
                s = shift(key, x)
                x["cnll"] = [round(v + s, 4) for v in x["cnll"]]
                if x.get("cnll_gt"):
                    x["cnll_gt"] = [round(v + s, 4) for v in x["cnll_gt"]]
                sh.append(s)
            means[key] = (len(sh), float(np.mean(sh)) if sh else 0.0)
        def fix(block, key):
            if "cnll" not in block or not block.get("n_cnll"):
                return
            n, m = shift_for(key, means)
            if n != block["n_cnll"]:
                sys.exit(f"{path} [{key}]: n_cnll {block['n_cnll']} khác số ảnh có C-NLL trong dump ({n})")
            block["cnll"] += m
            if block.get("cnll_gt") is not None:
                block["cnll_gt"] += m

        for key, block in summ["results"].items():
            fix(block, key)
        for key, block in (summ.get("prior") or {}).items():
            fix(block, key)
        for key, rows in dump["results"].items():                        # kiểm: TB box tốt nhất dump == file tổng
            sel = [min(x["cnll"]) for x in rows if x.get("cnll")]
            if key in summ["results"] and sel and abs(np.mean(sel) - summ["results"][key]["cnll"]) > 1e-3:
                sys.exit(f"{path} [{key}]: TB dump {np.mean(sel):.4f} khác file tổng {summ['results'][key]['cnll']:.4f}")
        summ["cnll_calib"] = dump["cnll_calib"] = "mu"
        with open(bpath, "w") as f:                                     # như eval.py: dump gọn, file tổng thụt lề
            json.dump(dump, f)
        with open(path, "w") as f:
            json.dump(summ, f, indent=2, ensure_ascii=False)
        print(f"{path}: " + ", ".join(f"{k} +{m:.3f} ({n} ảnh)" for k, (n, m) in means.items()))


if __name__ == "__main__":
    main()
