"""
COCO-minitrain cho variant (d) — detection đa class có class head.

Vì sao KHÔNG dùng `object-detection/diffusiondet/objdet/datasets.py`:
file đó chỉ *đăng ký* dataset với **detectron2** (`register_coco_instances`) —
việc đọc ảnh/annotation là do detectron2 làm. Env `ce-locmodel` không có
detectron2 (phải build từ source cho khớp CUDA), và kéo nó về chỉ để đọc một
file JSON là quá đắt. `torchvision.datasets.CocoDetection` thì cần
`pycocotools`, cũng là dependency mới.

`instances_minitrain2017.json` là JSON thường, nên ở đây parse thẳng bằng
`json` — không thêm dependency nào. Layout đọc đúng thư mục mà
`objdet/datasets.py` mô tả, nên dùng chung `data/` với DiffusionDet:

    data/coco_minitrain/annotations/instances_minitrain2017.json
    data/coco_minitrain/images/train2017/
    data/coco/annotations/instances_val2017.json
    data/coco/val2017/

Ba chỗ dễ sai của format COCO, đã xử lý (đo trên chính file này):
  1. `bbox` là **[x, y, w, h] góc trên-trái**, KHÔNG phải cxcywh — phải đổi.
  2. `category_id` chạy **1..90 không liên tục** (80 class) — class head cần
     index liên tục 0..79, nên phải map. Giữ luôn map ngược để in tên class.
  3. `iscrowd=1` (1,13% annotation) là vùng đám đông, không phải một vật —
     DiffusionDet bỏ chúng khi train (`DiffusionDetDatasetMapper`), ở đây cũng vậy.

Khác `CE130DetectionDataset`: ở đây **không có text** (variant (d) bỏ nhánh
text, class head tự lo phân loại) và mỗi mẫu trả thêm `labels` cho class head.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class CocoDetectionDataset(Dataset):
    def __init__(self, ann_file, image_dir, target_size=(512, 512),
                 num_proposals=300, max_boxes=None, keep_crowd=False):
        """
        Args:
            ann_file: instances_*.json
            image_dir: thư mục chứa ảnh (khớp `file_name` trong json)
            num_proposals: N slot cho variant (d). COCO-minitrain có mean 7,3 /
                max 80 box mỗi ảnh nên N=300 phủ 100% ảnh (đã đo).
            max_boxes: bỏ ảnh nhiều box hơn ngưỡng (None = không lọc)
            keep_crowd: giữ annotation iscrowd=1. Mặc định BỎ, giống DiffusionDet.
        """
        self.image_dir = image_dir
        self.target_size = target_size
        self.num_proposals = num_proposals

        with open(ann_file) as f:
            data = json.load(f)

        # category_id (1..90, thưa) -> index liên tục 0..C-1 cho class head.
        cats = sorted(data["categories"], key=lambda c: c["id"])
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats)}
        self.idx_to_cat_name = [c["name"] for c in cats]
        self.num_classes = len(cats)

        imgs = {im["id"]: im for im in data["images"]}
        per_image = {}
        n_crowd = 0
        for a in data["annotations"]:
            if not keep_crowd and a.get("iscrowd", 0):
                n_crowd += 1
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:      # box suy biến
                continue
            per_image.setdefault(a["image_id"], []).append(
                (x + w / 2.0, y + h / 2.0, w, h, self.cat_id_to_idx[a["category_id"]])
            )

        self.samples = []
        n_skipped = 0
        for img_id, boxes in sorted(per_image.items()):
            if max_boxes is not None and len(boxes) > max_boxes:
                n_skipped += 1
                continue
            im = imgs[img_id]
            self.samples.append({
                "img_id": img_id,
                "file_name": im["file_name"],
                "boxes": boxes,      # list (cx, cy, w, h, label) — pixel gốc
            })

        n = [len(s["boxes"]) for s in self.samples]
        print(
            f"[COCO/{os.path.basename(ann_file)}] {len(self.samples)} ảnh "
            f"({self.num_classes} class) | box/ảnh: mean={np.mean(n):.1f} "
            f"median={int(np.median(n))} max={max(n)} | num_proposals={num_proposals}"
            + (f" | bỏ {n_crowd} ann iscrowd" if n_crowd else "")
            + (f" | bỏ {n_skipped} ảnh >{max_boxes} box" if n_skipped else "")
        )

    def __len__(self):
        return len(self.samples)

    def resize_and_pad(self, img):
        """Giống hệt CE130DetectionDataset: giữ tỉ lệ, pad góc trên-trái."""
        w, h = img.size
        tw, th = self.target_size
        scale = min(tw / w, th / h)
        img = img.resize((int(w * scale), int(h * scale)), resample=Image.BILINEAR)
        canvas = Image.new("RGB", (tw, th), (0, 0, 0))
        canvas.paste(img, (0, 0))
        return canvas, scale

    def _normalize_bbox(self, bbox_px, scale):
        """Dùng ĐÚNG quy ước chuẩn hoá của CE-Loc (kích thước ánh xạ bằng cùng
        phép affine với tâm) để tái dùng nguyên matcher/loss/DDIM đã verify.
        Hệ quả: box nhỏ hơn nửa ảnh có norm_w ÂM — xem COORDINATE SPACE trong
        utils/matcher.py."""
        cx, cy, w, h = bbox_px
        cx, cy, w, h = cx * scale, cy * scale, w * scale, h * scale
        tw, th = self.target_size
        return torch.tensor([
            (cx / tw) * 2 - 1, (cy / th) * 2 - 1,
            (w / tw) * 2 - 1, (h / th) * 2 - 1,
        ], dtype=torch.float32)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img = Image.open(os.path.join(self.image_dir, rec["file_name"])).convert("RGB")
        original_size = img.size
        canvas, scale = self.resize_and_pad(img)
        img_t = F.to_tensor(canvas)

        gt_norm = torch.stack([
            self._normalize_bbox(b[:4], scale) for b in rec["boxes"]
        ])
        gt_labels = torch.tensor([b[4] for b in rec["boxes"]], dtype=torch.long)

        N, num_gt = self.num_proposals, gt_norm.shape[0]
        if num_gt < N:
            # Padding y hệt prepare_diffusion_concat của DiffusionDet, đã chuyển
            # sang không gian [-1,1] của CE-Loc (xem CE130DetectionDataset).
            ph = torch.randn(N - num_gt, 4) / 3.0
            ph[:, 2:] = torch.clamp(ph[:, 2:], min=1e-4 * 2 - 1)
            boxes = torch.cat([gt_norm, ph], dim=0)
            labels = torch.cat([gt_labels, torch.zeros(N - num_gt, dtype=torch.long)])
            mask = torch.cat([torch.ones(num_gt, dtype=torch.bool),
                              torch.zeros(N - num_gt, dtype=torch.bool)])
        elif num_gt > N:
            keep = torch.randperm(num_gt)[:N]
            boxes, labels, mask = gt_norm[keep], gt_labels[keep], torch.ones(N, dtype=torch.bool)
        else:
            boxes, labels, mask = gt_norm, gt_labels, torch.ones(N, dtype=torch.bool)

        return {
            "pixel_values": img_t,
            # Nhánh text bị tắt ở variant (d) nên chuỗi này không được dùng;
            # giữ lại để chung interface với CE130DetectionDataset.
            "text": "",
            "boxes": boxes,
            "box_mask": mask,
            "labels": labels,          # [N] index liên tục 0..C-1, cho class head
            "bbox": gt_norm[0],
            "scale": scale,
            "original_size": original_size,
            "img_id": rec["img_id"],
            "num_boxes": num_gt,
        }
