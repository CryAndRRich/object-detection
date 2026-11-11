"""Chấm box của baseline bằng ĐÚNG code chấm của CE-Loc (`ce_localization/engine/add_eval.py`, dump như `eval.py`).

Box xyxy pixel ảnh gốc × s = min(512 / W, 512 / H) -> canvas letterbox của CE-Loc (không kẹp — `add_metrics` tự kẹp box suy biến, đo
`in_image`). Record dựng như `predict_add` ⇒ cùng thứ tự mẫu + seed thì mốc `prior` trùng từng số với JSON eval CE-Loc. Khoá kết quả
như `eval.py`: CE-130 `inpainted`; CoCount `cocount` (+ `cocount_objsize[_exemplar]`). Chỉ số chính `cnll` (box C-NLL thấp nhất trong K
mỗi ảnh, TB trên ảnh >= 5 vật) + `cnll_median`, `cnll_mean_trim1` (bỏ 1 % ảnh tệ nhất — cạm bẫy 9).
"""

import json
import os

import numpy as np

from ce_localization.data.turns import iter_jsonl, letterbox_size
from ce_localization.engine.add_eval import (SIZE_STATS, add_metrics, dump_rows, prior_records, prior_unit_boxes, resize_to_objects,
                                             trim_mean, write_json)


_PRIOR = {}                                       # mốc prior lần chấm gần nhất (score_records)


def result_key(dataset):
    return "inpainted" if dataset == "ce130" else "cocount"


def records_from_boxes(samples, boxes_by_id, geo_fn, image_size=512):
    """{image_id: [[x1, y1, x2, y2] × K] pixel gốc} -> (record canvas, K). `geo_fn(sample)` = `data.geometry`."""
    missing = [s["image_id"] for s in samples if s["image_id"] not in boxes_by_id]
    if missing:
        raise ValueError(f"thiếu box cho {len(missing)} / {len(samples)} mẫu (vd {missing[:3]})")
    ks = {len(boxes_by_id[s["image_id"]]) for s in samples}
    if len(ks) != 1 or 0 in ks:
        raise ValueError(f"mọi mẫu phải cùng số box K > 0, gặp {sorted(ks)}")
    recs = []
    for s in samples:
        g = geo_fn(s)
        r = {"image_id": s["image_id"], "t": s["t"], "wh": np.asarray(g["wh"], dtype=np.float64),
             "boxes": np.asarray(boxes_by_id[s["image_id"]], dtype=np.float64).reshape(-1, 4) * letterbox_size(s["wh"], image_size)[0]}
        r.update({k: np.asarray(g[k], dtype=np.float64) for k in ("holes", "objects", "objects_all", "exemplars") if k in g})
        recs.append(r)
    return recs, ks.pop()


def cnll_summary(rows):
    """Dòng dump -> TB / trung vị / TB bỏ 1 % ảnh tệ nhất của C-NLL box tốt nhất mỗi ảnh + trung vị lỗ GT."""
    best = np.sort([min(r["cnll"]) for r in rows if "cnll" in r])
    gt = [min(r["cnll_gt"]) for r in rows if "cnll_gt" in r]
    out = {}
    if len(best):
        out.update(cnll_median=float(np.median(best)), cnll_mean_trim1=trim_mean(best, 1))
    if gt:
        out["cnll_gt_median"] = float(np.median(gt))
    return out


def score_records(recs, dataset, K, *, index=None, obj_size=None, seed=0):
    """-> (results, prior, dump) theo khoá. `index` (TurnIndex có split train) -> mốc `prior`; None thì bỏ."""
    cocount = dataset == "cocount"
    for st in obj_size or ():
        if st not in SIZE_STATS:
            raise ValueError(f"obj_size {st!r} không thuộc {SIZE_STATS}")
    pkey = (dataset, K, seed, tuple(obj_size or ()), id(index), tuple(r["image_id"] for r in recs))
    known = _PRIOR.get(pkey)                      # mốc prior chỉ phụ thuộc mẫu (không phụ thuộc box) ⇒ nhiều tiêu chí / lần chấm dùng lại
    pri_base = None if index is None or known else prior_records(recs, prior_unit_boxes(index, "train"), K, seed=seed)
    results, prior, dump = {}, dict(known or {}), {}
    for suffix, st in [("", None)] + [(f"_objsize{'' if st == 'mean' else '_' + st}", st) for st in (obj_size or ())]:
        key = result_key(dataset) + suffix
        rec = recs if st is None else resize_to_objects(recs, st)
        dump[key] = dump_rows(rec)
        results[key] = {**add_metrics(rec, latest=not cocount), **cnll_summary(dump[key])}
        if pri_base is not None:
            prior[key] = add_metrics(pri_base if st is None else resize_to_objects(pri_base, st), latest=not cocount)
    if pri_base is not None:
        _PRIOR.clear()
        _PRIOR[pkey] = prior
    return results, prior, dump


def read_boxes(path, first=None):
    """jsonl {"image_id", "boxes"} (pixel gốc) -> {image_id: boxes}; `first` N: chỉ N box đầu (box[0] = box chính của baseline)."""
    return {r["image_id"]: r["boxes"][:first] if first else r["boxes"] for r in iter_jsonl(path)}


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def write_outputs(out_path, meta, results, prior, dump):
    """`<out>.json` (meta + results + prior) + `<out>_boxes.json` (dump) — đọc được bằng `cnll_outliers`, `plot_cnll_worst` của CE-Loc."""
    write_json(out_path, {**meta, "cnll_calib": "mu", "results": results, "prior": prior})
    write_json(os.path.splitext(out_path)[0] + "_boxes.json", {**meta, "cnll_calib": "mu", "results": dump}, compact=True)
