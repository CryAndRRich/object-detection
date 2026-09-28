"""`ObjectPlacementDataset` của CE-Loc gốc (data/dataset.py) viết lại, thêm `use_density=False`
(không đọc file density). Tiền xử lý dùng chung `celoc_vision.resize_and_pad` (y hệt bản gốc).

Box: target_bbox cxcywh pixel ảnh gốc -> nhân scale -> chuẩn hoá [-1,1] theo canvas 512, KỂ CẢ
w, h (norm_w = w/512*2 - 1) như bản gốc.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ce_localization.legacy.celoc_vision import TARGET, resize_and_pad


class ObjectPlacementDataset(Dataset):
    def __init__(self, root_dir, target=TARGET, use_density=True):
        self.root_dir = root_dir
        self.target = target
        self.use_density = use_density
        self.image_dir = os.path.join(root_dir, "images")
        self.density_dir = os.path.join(root_dir, "density")
        self.annot_dir = os.path.join(root_dir, "annotation")
        self.files = sorted(f for f in os.listdir(self.image_dir) if f.endswith((".jpg", ".png", ".jpeg")))

    def __len__(self):
        return len(self.files)

    def parse_annotation(self, filename):
        with open(os.path.join(self.annot_dir, os.path.splitext(filename)[0] + ".json")) as f:
            data = json.load(f)
        return data["class"], data.get("target_bbox", [0.0, 0.0, 0, 0])

    def __getitem__(self, idx):
        filename = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        if self.use_density:
            den = Image.open(os.path.join(self.density_dir, os.path.splitext(filename)[0] + ".png")).convert("L")
        else:
            den = Image.new("L", img.size, 0)            # không dùng; chỉ để resize_and_pad chạy
        class_name, (cx, cy, w, h) = self.parse_annotation(filename)
        img_p, den_p, scale = resize_and_pad(img, den, self.target)

        T = self.target
        box = torch.tensor([(cx * scale / T) * 2 - 1, (cy * scale / T) * 2 - 1,
                            (w * scale / T) * 2 - 1, (h * scale / T) * 2 - 1], dtype=torch.float32)
        item = {
            "pixel_values": torch.from_numpy(np.asarray(img_p, dtype=np.float32) / 255.0).permute(2, 0, 1),
            "text": class_name,
            "bbox": box,
            "scale": scale,
            "index": idx,
        }
        if self.use_density:
            item["density_map"] = torch.from_numpy(np.asarray(den_p, dtype=np.float32) / 255.0)[None]
        return item
