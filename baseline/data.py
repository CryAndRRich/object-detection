"""Dữ liệu ZETA, box xyxy PIXEL ẢNH GỐC (baseline tự resize theo kiểu của nó).

Mẫu TEST — cùng tập, cùng thứ tự với `ce_localization/eval.py`:
  image_id   CE-130 `<iid>_b<k>_t<t>` (khoá `turns` của turn_index.json); CE-CoCount: tên file không đuôi
  image      CE-130 `samples/test/images/…` (ảnh đã inpaint xoá vật 1..t, như CE-Loc); CE-CoCount `Image/<tên>.jpg` (ảnh gốc)
  objects    vật CÙNG lớp đang có (CE-130: `removed == 0` hoặc `> t`; CoCount: `exam_bbox` sau rà tay) — nguồn ảnh vật mẫu
  holes      lỗ GT `holes[:t]` | 10 `loc_bbox` — chỉ để chấm, baseline KHÔNG đọc
`geometry(sample)` = hình học PIXEL CANVAS 512 như lúc eval CE-Loc (dùng thẳng `ce130_geometry` / `cocount_geometry`).

Mẫu TRAIN (`train_items`, chỉ `samples/train` như CE-Loc): `image` = ảnh đã xoá vật 1..t; `before` = ảnh còn vật t (mẫu samples lượt t−1
cùng nhánh, t = 1: `ground_truth.jpg`); `gt` = `<gt_root>/<split>/<nhánh>/ground_truth.jpg` (cắt ảnh vật đã xoá); `holes` lỗ mới nhất cuối.

Ảnh vật mẫu (foreground): bài ADD chỉ cho tên lớp ⇒ cắt một vật CÙNG lớp đang có trong ảnh, mask chữ nhật đầy (vật có sẵn không có mask).
`median` = vật diện tích trung vị; `random` = mỗi lần rút một vật (RNG theo mẫu: chạy lại / đổi batch ra cùng số).
"""

import hashlib
import os

import numpy as np
from PIL import Image

from ce_localization.data.cocount import CoCountAddDataset, cocount_geometry, read_cocount
from ce_localization.data.turns import TurnIndex
from ce_localization.tools.plot_cnll_worst import ce130_geometry

__all__ = ["DATASETS", "FG_SELECT", "load_samples", "canvas_scale", "geometry", "train_items", "sample_rng", "pick_objects", "int_box"]

DATASETS = ("ce130", "cocount")
FG_SELECT = ("median", "random")


def _present(b, t):
    removed = np.asarray(b["removed"], dtype=int)
    return np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)[(removed == 0) | (removed > t)]


def load_samples(dataset, *, turn_index=None, samples_root=None, split="test_all", cocount_root=None, review=None, min_obj=None,
                 limit=None):
    """-> (list mẫu test, TurnIndex | None). TurnIndex (đủ split train) còn là nguồn mốc `prior`."""
    if dataset not in DATASETS:
        raise ValueError(f"dataset {dataset!r} không thuộc {DATASETS}")
    index = TurnIndex(turn_index) if turn_index else None
    out = []
    if dataset == "ce130":
        for key in index.keys(split):
            e = index.turns[key]
            b = index.branches[e["branch"]]
            out.append({"dataset": "ce130", "image_id": key, "t": e["t"], "image": os.path.join(samples_root, e["sample"]),
                        "wh": tuple(int(v) for v in b["wh"]), "objects": _present(b, e["t"]),
                        "holes": np.asarray(b["holes"][:e["t"]], dtype=np.float64).reshape(-1, 4)})
    else:
        for key in CoCountAddDataset(cocount_root, review=review, min_obj=min_obj).keys:
            r = read_cocount(cocount_root, key, None, review)
            path = os.path.join(cocount_root, "Image", key + ".jpg")
            with Image.open(path) as im:
                wh = im.size
            out.append({"dataset": "cocount", "image_id": key, "t": 0, "image": path, "wh": tuple(int(v) for v in wh),
                        "objects": r["objects"], "holes": r["loc"]})
    return (out[:limit] if limit else out), index


def canvas_scale(wh, image_size=512):
    """Ảnh gốc -> canvas letterbox (neo góc trên trái) của CE-Loc: s = min(T / W, T / H)."""
    return min(image_size / wh[0], image_size / wh[1])


def geometry(sample, *, index=None, samples_root=None, cocount_root=None, review=None, image_size=512):
    """Hình học pixel canvas như lúc eval CE-Loc: wh (nw, nh), holes, objects [+ objects_all, exemplars]."""
    if sample["dataset"] == "ce130":
        g = ce130_geometry(index, samples_root, sample["image_id"], image_size)
        return {k: g[k] for k in ("wh", "holes", "objects")}
    return cocount_geometry(cocount_root, sample["image_id"], image_size, review=review)


def train_items(index, samples_root, gt_root, split="train"):
    out = []
    for key in index.keys(split):
        e = index.turns[key]
        b = index.branches[e["branch"]]
        t = e["t"]
        gt = os.path.join(gt_root, e["branch"], "ground_truth.jpg")
        before = gt if t == 1 else os.path.join(samples_root, index.turns[f"{key.rsplit('_t', 1)[0]}_t{t - 1}"]["sample"])
        out.append({"image_id": key, "t": t, "image": os.path.join(samples_root, e["sample"]), "before": before, "gt": gt,
                    "wh": tuple(int(v) for v in b["wh"]), "objects": _present(b, t),
                    "holes": np.asarray(b["holes"][:t], dtype=np.float64).reshape(-1, 4)})
    return out


def sample_rng(seed, image_id):
    """RNG riêng mỗi mẫu, không phụ thuộc thứ tự / batch."""
    h = hashlib.blake2b(f"{seed}:{image_id}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(h, "little"))


def pick_objects(objects, n, mode, rng=None):
    """-> n chỉ số vật trong `objects` [M,4] (M >= 1) làm ảnh vật mẫu."""
    objects = np.asarray(objects, dtype=np.float64).reshape(-1, 4)
    if not len(objects):
        raise ValueError("ảnh không có vật cùng lớp để làm ảnh vật mẫu")
    if mode == "median":
        area = (objects[:, 2] - objects[:, 0]) * (objects[:, 3] - objects[:, 1])
        return [int(np.argsort(area, kind="stable")[(len(area) - 1) // 2])] * n
    if mode == "random":
        return [int(i) for i in rng.integers(0, len(objects), size=n)]
    raise ValueError(f"fg_select {mode!r} không thuộc {FG_SELECT}")


def int_box(box, wh):
    """Box float -> (x1, y1, x2, y2) nguyên trong ảnh (W, H), mỗi cạnh >= 1 px."""
    W, H = wh
    x1 = int(np.clip(np.floor(box[0]), 0, W - 1))
    y1 = int(np.clip(np.floor(box[1]), 0, H - 1))
    return x1, y1, int(np.clip(np.ceil(box[2]), x1 + 1, W)), int(np.clip(np.ceil(box[3]), y1 + 1, H))
