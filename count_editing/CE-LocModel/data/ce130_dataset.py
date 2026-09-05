"""CE-130 cho bài DETECTION có điều kiện theo category.

Input:  (ground_truth.jpg, tên class)   Output: box của riêng class đó.

Phần logic (parse / dedupe / toạ độ) là NUMPY THUẦN nên test được ở local không
cần torch. `to_tensor` chỉ xảy ra ở collate.

BỐN ĐIỂM BẮT BUỘC ĐÚNG — mỗi cái là một lỗi đã đo được:

1. HAI NGUỒN, HAI ĐỊNH DẠNG BOX KHÁC NHAU:
     all_phase2_V2/*/annotation.json : all_bboxes, inpainted_bboxes -> **xyxy**
     samples/*/annotation/*.json     : target_bbox                  -> **cxcywh**
   (đo: 10.368/10.376 nhất quán xyxy; 19.998/19.998 cxcywh, và đối chiếu chéo
   giữa hai nguồn cho IoU = 1,0000)

2. GIỮ NGUYÊN `all_bboxes`, KHÔNG trừ `inpainted_bboxes`.
   `ground_truth.jpg` là ảnh GỐC CHƯA XOÁ GÌ. Bằng chứng pixel: diff giữa nó và
   `inpainted_turn_1.png` tại vùng inpainted_bboxes[0] = 51,96/255 so với toàn
   ảnh 1,41/255 -> vật CÓ trong ảnh rồi mới bị xoá ở turn sau.
   Vòng 1 trừ đi -> vứt bỏ 7-8 % vật THẬT.

3. DEDUPE THEO ẢNH. Branch _b1/_b2/_b3 dùng chung một ground_truth.jpg (md5 trùng
   186/186) và all_bboxes giống hệt (1.571/1.571) -> 1.911 / 908 / 779 ảnh.

4. PAD BẰNG CLIP MEAN, không phải đen. Đen cho -1,79 sigma sau chuẩn hoá (khối
   tối, tạo cạnh giả cho ViT); CLIP mean cho đúng 0,000.

Ngoài ra: `fixed_annotation.json` chỉ có ở val/test -> fallback annotation.json.
Lọc 14/37.110 box degenerate. Mọi ảnh cao đúng 384px nên luôn W>=H -> pad LUÔN ở
dưới, `valid_h` là một ngưỡng vô hướng.
"""

import glob
import json
import os

import numpy as np
from PIL import Image

from utils.box_ops_np import filter_degenerate, flip_horizontal, scale_to_canvas

__all__ = ["CLIP_MEAN", "CLIP_STD", "CE130Detection", "resize_and_pad"]

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def resize_and_pad(img, target=512):
    """Giữ tỉ lệ, pad góc trên-trái bằng CLIP mean. Trả về (ảnh RGB uint8, valid_h).

    Mọi ảnh CE-130 cao 384px và rộng >= 384 -> new_w luôn = target, pad ở DƯỚI.
    """
    W, H = img.size
    s = min(target / W, target / H)
    nw, nh = int(W * s), int(H * s)

    canvas = np.empty((target, target, 3), dtype=np.uint8)
    canvas[:] = (CLIP_MEAN * 255).round().astype(np.uint8)
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.uint8)
    return canvas, nh / float(target)


class CE130Detection:
    """Trả về dict numpy. Bọc torch Dataset ở tầng trên (giữ file này không cần torch)."""

    def __init__(self, root, split, target=512, flip_prob=0.0, seed=None):
        self.root = root
        self.split = split
        self.target = target
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed)
        self.items = self._quet()

    # ------------------------------------------------------------------ index

    def _quet(self):
        """Dedupe theo ảnh gốc: mỗi image-id lấy đúng một branch."""
        theo_anh = {}
        for br in sorted(glob.glob(os.path.join(self.root, self.split, "*"))):
            ann = self._doc_annotation(br)
            if ann is None:
                continue
            iid = os.path.basename(br).split("_b")[0]
            if iid not in theo_anh:                       # branch nào cũng như nhau
                theo_anh[iid] = (br, ann)

        items = []
        for iid, (br, ann) in sorted(theo_anh.items()):
            img_path = os.path.join(br, "ground_truth.jpg")
            if not os.path.exists(img_path):
                continue
            boxes = np.asarray(ann.get("all_bboxes", []), dtype=np.float64).reshape(-1, 4)
            boxes, _ = filter_degenerate(boxes)           # bỏ 14/37.110 box w/h <= 0
            items.append({
                "image_id": iid,
                "img_path": img_path,
                "boxes_xyxy_px": boxes,                   # KHÔNG trừ inpainted_bboxes
                "text": ann.get("class_based_caption", ""),
            })
        return items

    @staticmethod
    def _doc_annotation(branch_dir):
        """fixed_annotation.json chỉ có ở val/test -> fallback annotation.json."""
        for ten in ("fixed_annotation.json", "annotation.json"):
            p = os.path.join(branch_dir, ten)
            if os.path.exists(p):
                with open(p, "r") as f:
                    return json.load(f)
        return None

    # ----------------------------------------------------------------- access

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["img_path"]).convert("RGB")
        W, H = img.size

        canvas, valid_h = resize_and_pad(img, self.target)
        boxes, _, _ = scale_to_canvas(it["boxes_xyxy_px"], W, H, self.target)

        if self.flip_prob > 0 and self.rng.random() < self.flip_prob:
            canvas = canvas[:, ::-1].copy()
            boxes = flip_horizontal(boxes)
            lat = True
        else:
            lat = False

        return {
            "image": canvas,                 # uint8 [512,512,3], đã pad CLIP mean
            "boxes": boxes,                  # cxcywh [0,1] — HỆ CHUẨN
            "text": it["text"],              # 1 từ tên category
            "valid_h": valid_h,              # ranh giới vùng ảnh thật (chặn placeholder)
            "image_id": it["image_id"],
            "orig_size": (W, H),
            "flipped": lat,
        }

    # ------------------------------------------------------------ thống kê

    def thong_ke(self):
        n = [len(it["boxes_xyxy_px"]) for it in self.items]
        return {
            "so_anh": len(self.items),
            "tong_box": int(np.sum(n)),
            "box_moi_anh_median": float(np.median(n)) if n else 0.0,
            "box_moi_anh_mean": float(np.mean(n)) if n else 0.0,
            "box_moi_anh_max": int(np.max(n)) if n else 0,
            "so_class": len({it["text"] for it in self.items}),
        }


def normalize_for_clip(canvas_uint8):
    """uint8 HWC -> float32 CHW đã chuẩn hoá theo CLIP. Vùng pad ra đúng 0,000."""
    x = np.asarray(canvas_uint8, dtype=np.float32) / 255.0
    return ((x - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)
