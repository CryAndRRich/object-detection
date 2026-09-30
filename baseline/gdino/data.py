"""Dữ liệu cho BASELINE3.2 (finetune Grounding DINO bằng Open-GroundingDino, định dạng ODVG):

- `train_odvg.jsonl`: mỗi dòng một ảnh train CE-130, instance = box GT (xyxy pixel ảnh gốc, kẹp vào
  ảnh như `scale_boxes`) mang tên lớp thật (`class_based_caption`);
- `label_map.json`: {"0": tên, ...} — 72 lớp train, sắp xếp. Caption lúc train = lớp thật của ảnh + lớp
  âm rút từ bảng này (`max_labels`), do ODVGDataset của Open-GroundingDino dựng;
- `val_internal_coco.json`: vài chục ảnh val dạng COCO, một category "object" **id 0** (= nhãn 0 mà PostProcess
  trả ra với `label_list = ["object"]`, `use_coco_eval = False` — gdino/train.py) — CHỈ để main.py của
  Open-GroundingDino chạy được vòng eval mỗi epoch (nó bắt buộc có val). Không dùng để chọn checkpoint:
  chọn bằng `gdino/predict.py --split val` với prompt đúng lớp của từng ảnh (oracle_recall);
- `datasets.json`: file `--datasets` của main.py.
GT lấy từ `scan_ce130` như mọi baseline.
"""

import json
import os
import random

from PIL import Image

from ce_localization.data.dataset import scale_boxes, scan_ce130

__all__ = ["build_label_map", "odvg_line", "write_odvg", "write_coco_subset", "prepare"]


def _name(text):
    return " ".join(str(text).lower().split())


def build_label_map(items):
    return {str(i): n for i, n in enumerate(sorted({_name(it["text"]) for it in items}))}


def _wh_boxes(it):
    with Image.open(it["img_path"]) as im:
        w, h = im.size
    return w, h, scale_boxes(it["boxes_xyxy_px"], 1.0, w, h)


def odvg_line(it, root, label_of):
    w, h, boxes = _wh_boxes(it)
    name = _name(it["text"])
    return {"filename": os.path.relpath(it["img_path"], root), "height": h, "width": w,
            "detection": {"instances": [{"bbox": [round(float(v), 2) for v in b], "label": label_of[name],
                                         "category": name} for b in boxes]}}


def write_odvg(items, root, out_jsonl, label_map):
    label_of = {n: int(k) for k, n in label_map.items()}
    n_box = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for it in items:
            line = odvg_line(it, root, label_of)
            n_box += len(line["detection"]["instances"])
            f.write(json.dumps(line) + "\n")
    return len(items), n_box


def write_coco_subset(items, root, out_json, n_images, seed=0):
    items = sorted(items, key=lambda it: it["image_id"])
    pick = random.Random(seed).sample(items, min(n_images, len(items)))
    images, anns = [], []
    for img_id, it in enumerate(sorted(pick, key=lambda it: it["image_id"]), 1):
        w, h, boxes = _wh_boxes(it)
        images.append({"id": img_id, "file_name": os.path.relpath(it["img_path"], root), "width": w, "height": h})
        for b in boxes:
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": len(anns) + 1, "image_id": img_id, "category_id": 0, "iscrowd": 0,
                         "bbox": [float(b[0]), float(b[1]), bw, bh], "area": bw * bh})
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"images": images, "annotations": anns,
                   "categories": [{"id": 0, "name": "object", "supercategory": "object"}]}, f)
    return len(images), len(anns)


def prepare(root, data_dir, internal_val_images=50, seed=0, log=print):
    """Dựng 4 file ở `data_dir` (dựng lại mỗi lần gọi; nội dung tất định). -> đường dẫn datasets.json."""
    os.makedirs(data_dir, exist_ok=True)
    root = os.path.abspath(root)
    train, val = scan_ce130(root, "train"), scan_ce130(root, "val")
    label_map = build_label_map(train)
    paths = {k: os.path.join(data_dir, f) for k, f in (("odvg", "train_odvg.jsonl"), ("label_map", "label_map.json"),
                                                        ("val", "val_internal_coco.json"),
                                                        ("datasets", "datasets.json"))}
    with open(paths["label_map"], "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=1, ensure_ascii=False)
    n_img, n_box = write_odvg(train, root, paths["odvg"], label_map)
    v_img, v_box = write_coco_subset(val, root, paths["val"], internal_val_images, seed)
    with open(paths["datasets"], "w", encoding="utf-8") as f:
        json.dump({"train": [{"root": root, "anno": paths["odvg"], "label_map": paths["label_map"],
                              "dataset_mode": "odvg"}],
                   "val": [{"root": root, "anno": paths["val"], "label_map": None, "dataset_mode": "coco"}]}, f,
                  indent=1)
    log(f"[gdino data] train {n_img} ảnh / {n_box} box / {len(label_map)} lớp | val nội bộ {v_img} ảnh / {v_box} box"
        f" -> {data_dir}")
    return paths["datasets"]
