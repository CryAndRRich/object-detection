"""CE-CoCount — tập test phụ của bài (docs/EXPERIMENT_GAMMA.md mục 18), `data/cocount/`:

  Image/<tên>.jpg                     frame CoCount-train, ẢNH GỐC KHÔNG LỖ
  Anno/<tên>.json                     class_name, loc_bbox (10 box GT chỗ trống để thêm, xyxy px), counting_anno (count, points, …)
  Anno_with_exam_bbox/<tên>.json      exam_bbox = [{bbox, score}] box MỌI vật lớp đó (SAM + chỉnh tay)
  <tên> = {INTRA|INTER}_{nhóm}_{A}_{B}_{sốA}_{sốB}_…_{positive|negative}; `_positive` / `_negative` = hai lớp CÙNG frame.

Một phần tử = một tên (một lớp cần thêm), cùng khuôn `CE130AddDataset` để `eval.py` / `predict_add` dùng lại:
  holes       = 10 `loc_bbox` (đáp án; không có thứ tự lượt ⇒ chỉ số `_latest` vô nghĩa — `add_metrics(latest=False)`)
  objects     = box vật CÙNG lớp (GAMMA4 nhận làm box vật; C-NLL fit trên chúng)
  objects_all = box vật CẢ HAI lớp của frame (file `_positive` + `_negative`) — `on_object` tính trên đây
  exemplars   = 3 box mẫu GÁN TAY (`counting_anno.exemplars`) — cỡ vật đáng tin nhất (box SAM của ~10 % mẫu khoanh cả đống vật);
                chỉ dùng cho `eval.py --obj-size exemplar`
  `max_obj_ratio` k (mặc định None = giữ hết): BỎ box vật có cạnh √(wh) > k × cạnh TB 3 exemplar của chính file đó (lớp kia: exemplar
  của file kia) — box SAM ở ~15 % mẫu khoanh cả đống vật (docs/EXPERIMENT_GAMMA.md mục 18); k = 3 giữ 77 % box, cạnh vật / chỗ trống GT
  trung vị 1,00, p90 1,41 (không lọc: p90 8,0); 80 mẫu còn < 5 vật.
  t = 0, density TRỐNG (bản tải về không có density map; người dùng chốt 2026-10-07), text = `class_name` bỏ phần trong ngoặc và dấu chấm.
"""

import json
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ce_localization.data.dataset import letterbox, normalize, scale_boxes
from ce_localization.data.turns import INPUT_STYLES, _letterbox_l_u8

__all__ = ["clean_class", "twin_name", "filter_objects", "read_cocount", "CoCountAddDataset"]


def clean_class(name):
    """'normal tomato(Big round tomato).' -> 'normal tomato' (bỏ phần trong ngoặc, dấu chấm cuối, khoảng trắng thừa)."""
    return re.sub(r"\s+", " ", re.sub(r"\(.*?\)", "", name)).strip().rstrip(".").strip()


def twin_name(name):
    """Tên file lớp còn lại của cùng frame."""
    stem, pol = name.rsplit("_", 1)
    return f"{stem}_{'negative' if pol == 'positive' else 'positive'}"


def _side(b):
    return np.sqrt(np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None))


def filter_objects(objs, exemplars, max_ratio=None):
    """Bỏ box có cạnh √(wh) > max_ratio × cạnh TB của `exemplars`; None / không có exemplar: giữ nguyên."""
    if max_ratio is None or not len(exemplars) or not len(objs):
        return objs
    return objs[_side(objs) <= max_ratio * _side(exemplars).mean()]


def read_cocount(root, name, max_obj_ratio=None):
    """-> dict thô (pixel ảnh gốc): class, loc [10,4], objects [M,4] cùng lớp, objects_all [M',4] cả hai lớp, exemplars [3,4], count.
    `max_obj_ratio`: lọc box vật (`filter_objects`, mỗi lớp theo exemplar của file lớp đó)."""
    def load(sub, n):
        with open(os.path.join(root, sub, n + ".json")) as f:
            return json.load(f)

    def boxes_ex(n):
        objs = np.asarray([x["bbox"] for x in load("Anno_with_exam_bbox", n)["exam_bbox"]], dtype=np.float64).reshape(-1, 4)
        ex = np.asarray(load("Anno", n)["counting_anno"].get("exemplars", []), dtype=np.float64).reshape(-1, 4)
        return filter_objects(objs, ex, max_obj_ratio), ex

    a = load("Anno", name)
    objs, ex = boxes_ex(name)
    tw = twin_name(name)
    other = (boxes_ex(tw)[0] if os.path.exists(os.path.join(root, "Anno_with_exam_bbox", tw + ".json"))
             and os.path.exists(os.path.join(root, "Anno", tw + ".json")) else np.zeros((0, 4)))
    return {"class": clean_class(a["class_name"]), "loc": np.asarray(a["loc_bbox"], dtype=np.float64).reshape(-1, 4),
            "objects": objs, "objects_all": np.concatenate([objs, other]), "count": a["counting_anno"]["count"],
            "exemplars": ex}


class CoCountAddDataset(Dataset):
    """`style` như `CE130AddDataset`: paper = uint8 HWC (+ density trống `.convert("L")` của PNG jet trống = 14/255 trên vùng thật);
    ours = chuẩn hoá ImageNet (+ density 0). `density` None (model 3 kênh) hoặc "empty"."""

    def __init__(self, root, image_size=512, style="paper", density="empty", limit=None, max_obj_ratio=None):
        if style not in INPUT_STYLES:
            raise ValueError(f"style {style!r} không thuộc {INPUT_STYLES}")
        if density not in (None, "empty"):
            raise ValueError(f"CE-CoCount không có density map: density phải là None hoặc 'empty', không phải {density!r}")
        self.root, self.image_size, self.style, self.density = root, image_size, style, density
        self.max_obj_ratio = max_obj_ratio
        self.keys = sorted(f[:-5] for f in os.listdir(os.path.join(root, "Anno")) if f.endswith(".json"))
        if limit:
            self.keys = self.keys[:limit]
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
        r = read_cocount(self.root, key, self.max_obj_ratio)
        img = Image.open(os.path.join(self.root, "Image", key + ".jpg")).convert("RGB")
        canvas, scale, nw, nh = letterbox(img, self.image_size)
        paper = self.style == "paper"
        x = canvas if paper else normalize(canvas)
        ch = -1 if paper else 0
        if self.density == "empty":
            den = (_letterbox_l_u8(np.asarray(Image.new("RGB", img.size, (0, 0, 127)).convert("L")), nw, nh, self.image_size)
                   if paper else np.zeros((self.image_size, self.image_size), dtype=np.float32))
            x = np.concatenate([x, np.expand_dims(den, ch)], axis=ch)
        holes = scale_boxes(r["loc"], scale, nw, nh)
        return {
            "image": (torch.from_numpy(np.ascontiguousarray(x, dtype=np.uint8)).permute(2, 0, 1) if paper else
                      torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))),
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
