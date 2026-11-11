"""CE-CoCount — tập test phụ của bài, `data/cocount/`: 742 frame CoCount-train × 2 lớp, ẢNH GỐC KHÔNG LỖ, KHÔNG có density map.

  Image/<tên>.jpg, Anno/<tên>.json (class_name, loc_bbox = 10 box GT chỗ trống xyxy px, counting_anno.exemplars = 3 box mẫu gán tay),
  Anno_with_exam_bbox/<tên>.json (exam_bbox = box MỌI vật lớp đó, SAM + chỉnh tay); `_positive` / `_negative` = hai lớp CÙNG frame.

Một phần tử = một tên (một lớp cần thêm), cùng khuôn `CE130AddDataset`: holes = 10 chỗ trống GT (không thứ tự lượt), objects = box vật
CÙNG lớp, objects_all = vật CẢ HAI lớp [cùng lớp ; lớp kia], exemplars, t = 0, density TRỐNG (PNG jet trống `.convert("L")` = 14/255
trên vùng thật), text = `class_name` bỏ phần trong ngoặc và dấu chấm.
Lọc box vật (cả hai lớp): `review` = bản rà tay (`box_review/`, `load_review`: bỏ `bad` ∪ `auto`); `min_obj` n: chỉ giữ file còn
>= n box vật cùng lớp (review5 = review + n = 5).
"""

import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ce_localization.data.turns import _letterbox_l_u8, iter_jsonl, letterbox, letterbox_size, scale_boxes


def clean_class(name):
    """'normal tomato(Big round tomato).' -> 'normal tomato' (bỏ phần trong ngoặc, dấu chấm cuối, khoảng trắng thừa)."""
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", name)).strip().rstrip(".").strip()


def twin_name(name):
    """Tên file lớp còn lại của cùng frame."""
    stem, pol = name.rsplit("_", 1)
    return f"{stem}_{'negative' if pol == 'positive' else 'positive'}"


def load_review(path):
    """File jsonl của `box_review/app.py cocount` -> {tên file: (số box gốc, set chỉ số box bị loại = `bad` ∪ `auto`)} (dòng sau đè dòng
    trước cùng tên)."""
    out = {}
    for r in iter_jsonl(path):
        for g, n in r["n"].items():
            out[g] = (int(n), set(r["bad"].get(g, [])) | set(r.get("auto", {}).get(g, [])))
    return out


def _apply_review(objs, name, review):
    if name not in review:
        raise ValueError(f"CE-CoCount: {name} chưa có trong bản rà box")
    n, removed = review[name]
    if n != len(objs):
        raise ValueError(f"CE-CoCount: {name} có {len(objs)} box nhưng bản rà ghi {n} — bản rà không khớp dữ liệu")
    return objs[[i for i in range(len(objs)) if i not in removed]]


def read_cocount(root, name, review=None):
    """-> dict thô (pixel ảnh gốc): class, loc [10,4], objects [M,4] cùng lớp, objects_all [M',4] cả hai lớp, exemplars [3,4], count.
    `review` (`load_review`): bỏ box đã loại khi rà tay (mỗi lớp theo bản rà của file lớp đó)."""

    def load(sub, n):
        with open(os.path.join(root, sub, n + ".json")) as f:
            return json.load(f)

    def boxes_ex(n):
        objs = np.asarray([x["bbox"] for x in load("Anno_with_exam_bbox", n)["exam_bbox"]], dtype=np.float64).reshape(-1, 4)
        ex = np.asarray(load("Anno", n)["counting_anno"].get("exemplars", []), dtype=np.float64).reshape(-1, 4)
        return (_apply_review(objs, n, review) if review is not None else objs), ex

    a = load("Anno", name)
    objs, ex = boxes_ex(name)
    tw = twin_name(name)
    other = (boxes_ex(tw)[0] if os.path.exists(os.path.join(root, "Anno_with_exam_bbox", tw + ".json"))
             and os.path.exists(os.path.join(root, "Anno", tw + ".json")) else np.zeros((0, 4)))
    return {"class": clean_class(a["class_name"]), "loc": np.asarray(a["loc_bbox"], dtype=np.float64).reshape(-1, 4),
            "objects": objs, "objects_all": np.concatenate([objs, other]), "count": a["counting_anno"]["count"],
            "exemplars": ex}


def cocount_geometry(root, name, image_size=512, review=None):
    """-> dict lỗ / box vật / box vật cả hai lớp / exemplar (pixel canvas, float32 như lúc eval) + (nw, nh) — y `CoCountAddDataset`."""
    r = read_cocount(root, name, review)
    with Image.open(os.path.join(root, "Image", name + ".jpg")) as im:
        scale, nw, nh = letterbox_size(im.size, image_size)
    sb = lambda b: scale_boxes(b, scale, nw, nh).astype(np.float32).astype(np.float64)  # noqa: E731
    return {"holes": sb(r["loc"]), "objects": sb(r["objects"]), "objects_all": sb(r["objects_all"]),
            "exemplars": sb(r["exemplars"]), "wh": np.array([nw, nh], dtype=np.float64)}


class CoCountAddDataset(Dataset):
    """Đầu vào kiểu bài như `CE130AddDataset`: uint8 HWC 4 kênh (density trống)."""

    def __init__(self, root, image_size=512, review=None, min_obj=None):
        self.root, self.image_size, self.review = root, image_size, review
        self.keys = sorted(f[:-5] for f in os.listdir(os.path.join(root, "Anno")) if f.endswith(".json"))
        self.n_all = len(self.keys)
        if min_obj:                                   # chỉ file còn >= min_obj box vật cùng lớp SAU lọc
            self.keys = [k for k in self.keys if len(read_cocount(root, k, review)["objects"]) >= min_obj]
        self._cls = {}

    def __len__(self):
        return len(self.keys)

    def classes(self):
        for k in self.keys:
            if k not in self._cls:
                with open(os.path.join(self.root, "Anno", k + ".json")) as f:
                    self._cls[k] = clean_class(json.load(f)["class_name"])
        return sorted(set(self._cls.values()))

    def __getitem__(self, i):
        key = self.keys[i]
        r = read_cocount(self.root, key, self.review)
        img = Image.open(os.path.join(self.root, "Image", key + ".jpg")).convert("RGB")
        x, scale, nw, nh = letterbox(img, self.image_size)
        den = _letterbox_l_u8(np.asarray(Image.new("RGB", img.size, (0, 0, 127)).convert("L")), nw, nh, self.image_size)
        x = np.concatenate([x, den[..., None]], axis=-1)
        holes = scale_boxes(r["loc"], scale, nw, nh)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8)).permute(2, 0, 1),
            "target": torch.from_numpy(holes[0] if len(holes) else np.zeros(4)).float(),
            "holes": torch.from_numpy(holes).float(),
            "objects": torch.from_numpy(scale_boxes(r["objects"], scale, nw, nh)).float(),
            "objects_all": torch.from_numpy(scale_boxes(r["objects_all"], scale, nw, nh)).float(),
            "exemplars": torch.from_numpy(scale_boxes(r["exemplars"], scale, nw, nh)).float(),
            "valid_hw": (nh, nw),
            "text": r["class"],
            "image_id": key,
            "t": 0,
        }
