"""
CE-130 như một bộ dữ liệu OBJECT DETECTION có điều kiện theo class.

Khác hoàn toàn `ObjectPlacementDataset` (bài toán ADD: cho ảnh đã xoá vật, hỏi
"thêm vật mới vào đâu"). Ở đây:

    input  = (ground_truth.jpg, tên class)
    target = all_bboxes — TOÀN BỘ vật cùng class đang có trong ảnh

tức là bài toán detection thật, chỉ khác là class được cho trước qua text nên
model không cần class head (mỗi forward chỉ lo đúng 1 class).

Vì sao đọc thẳng `all_phase2_V2` chứ không dùng `samples/`:
  - `samples/` chỉ có 1 `target_bbox`/ảnh (cái bị xoá), không phải cả tập vật.
  - `all_phase2_V2/{split}/{img}_b{N}/annotation.json` có `all_bboxes` (đủ tập)
    + `class_based_caption` (tên class) + `ground_truth.jpg` (ảnh gốc, còn
    nguyên mọi vật).

DEDUPE — quan trọng: các branch `{img}_b1`, `{img}_b2`, ... **dùng chung một
`ground_truth.jpg`** (khác nhau ở thứ tự xoá vật, thứ không liên quan tới
detection). Đếm theo branch sẽ nhân đôi/ba dữ liệu và làm train/val rò rỉ lẫn
nhau về mặt ảnh. Đo được: 8.829 branch nhưng chỉ **3.598 ảnh gốc** thật
(train 1.911 / val 908 / test 779, overlap giữa 3 split = 0).

`fixed_annotation.json` được ưu tiên khi có (nó sửa một điểm lệch toạ độ giữa
`all_bboxes` và `inpainted_bboxes`), nhưng **train split không có file này**
(0/4653) — không sao, vì lệch đó nằm ở `inpainted_bboxes`, thứ bài detection
không dùng. `all_bboxes` tự nó hợp lệ hình học ở 294/300 branch train đã kiểm.

KHÔNG có density map: `all_phase2_V2` không kèm density cho `ground_truth.jpg`,
và density trong `samples/` vốn được vẽ từ chính các vật đang có — tức chính
`all_bboxes`, tức chính TARGET. Đưa vào input là rò rỉ đáp án. Nên nhánh này
chạy `vision_encoder.in_channels: 3` (chỉ RGB).
"""
import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# Ảnh gốc của một branch. Mọi branch của cùng một img_id đều có file này giống nhau.
SOURCE_IMAGE = "ground_truth.jpg"


class CE130DetectionDataset(Dataset):
    def __init__(self, root_dir, split, target_size=(512, 512),
                 num_proposals=1, max_boxes=None):
        """
        Args:
            root_dir: đường dẫn tới `all_phase2_V2/`.
            split: 'train' | 'val' | 'test' — dùng đúng split có sẵn của CE-130
                (đã kiểm: không có ảnh nào xuất hiện ở 2 split).
            target_size: canvas sau resize+pad, giống hệt ObjectPlacementDataset.
            num_proposals: 1 cho variant (a)/(b) — trả về đúng 1 box lấy ngẫu
                nhiên trong `all_bboxes`. >1 cho variant (c) — trả về cả tập,
                pad/crop về đúng N slot.
            max_boxes: bỏ qua ảnh có nhiều hơn ngần này box (None = không lọc).
                CE-130 có ảnh tới 1229 box; giữ nguyên thì một ảnh cá biệt có
                thể chiếm hết ngân sách N và làm lệch thống kê.
        """
        self.root_dir = root_dir
        self.split = split
        self.target_size = target_size
        self.num_proposals = num_proposals

        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            raise FileNotFoundError(f"không thấy split {split!r} trong {root_dir}")

        # --- gom branch theo ảnh gốc, mỗi ảnh giữ đúng 1 bản ghi ---
        by_image = {}
        for branch in sorted(glob.glob(os.path.join(split_dir, "*"))):
            if not os.path.isdir(branch):
                continue
            img_path = os.path.join(branch, SOURCE_IMAGE)
            if not os.path.exists(img_path):
                continue
            fixed = os.path.join(branch, "fixed_annotation.json")
            plain = os.path.join(branch, "annotation.json")
            anno_path = fixed if os.path.exists(fixed) else plain
            if not os.path.exists(anno_path):
                continue
            img_id = os.path.basename(branch).rsplit("_b", 1)[0]
            # Ưu tiên branch có fixed_annotation; ngoài ra branch đầu theo tên.
            prev = by_image.get(img_id)
            if prev is not None and not (os.path.exists(fixed) and not prev["has_fixed"]):
                continue
            with open(anno_path) as f:
                data = json.load(f)
            boxes = data.get("all_bboxes", [])
            cls = data.get("class_based_caption")
            if not boxes or not cls:
                continue
            by_image[img_id] = {
                "img_path": img_path,
                "boxes_xyxy": boxes,
                "text": cls,
                "has_fixed": os.path.exists(fixed),
            }

        self.samples = []
        self.n_skipped_degenerate = 0
        self.n_skipped_too_many = 0
        for img_id, rec in sorted(by_image.items()):
            # Bỏ box suy biến (x2<=x1 hoặc y2<=y1). Đo được ~2% branch train có
            # ít nhất một box như vậy; bỏ riêng box đó chứ không bỏ cả ảnh.
            good = [b for b in rec["boxes_xyxy"] if b[2] > b[0] and b[3] > b[1]]
            if len(good) != len(rec["boxes_xyxy"]):
                self.n_skipped_degenerate += len(rec["boxes_xyxy"]) - len(good)
            if not good:
                continue
            if max_boxes is not None and len(good) > max_boxes:
                self.n_skipped_too_many += 1
                continue
            self.samples.append({
                "img_id": img_id,
                "img_path": rec["img_path"],
                "boxes_xyxy": good,
                "text": rec["text"],
            })

        n_box = [len(s["boxes_xyxy"]) for s in self.samples]
        print(
            f"[CE130Detection/{split}] {len(self.samples)} ảnh gốc "
            f"(từ {len(by_image)} img_id, đã dedupe branch) | "
            f"box/ảnh: mean={np.mean(n_box):.1f} median={int(np.median(n_box))} "
            f"max={max(n_box)} | num_proposals={num_proposals}"
            + (f" | bỏ {self.n_skipped_too_many} ảnh >{max_boxes} box" if max_boxes else "")
            + (f" | bỏ {self.n_skipped_degenerate} box suy biến"
               if self.n_skipped_degenerate else "")
        )

    def __len__(self):
        return len(self.samples)

    def resize_and_pad(self, img):
        """Giống hệt ObjectPlacementDataset.resize_and_pad nhưng không có density:
        giữ tỉ lệ, pad về góc trên-trái (nên toạ độ chỉ cần nhân scale, không
        phải trừ offset)."""
        w, h = img.size
        target_w, target_h = self.target_size
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
        canvas.paste(img, (0, 0))
        return canvas, scale

    def _normalize_bbox(self, bbox_px, scale):
        """(cx,cy,w,h) pixel của ảnh GỐC -> [-1,1] theo canvas.

        Dùng ĐÚNG quy ước của ObjectPlacementDataset._normalize_bbox: kích thước
        được ánh xạ bằng cùng phép affine với tâm (`(v/target)*2 - 1`). Nghĩa là
        box nhỏ hơn nửa ảnh có norm_w ÂM — xem ghi chú COORDINATE SPACE trong
        utils/matcher.py. Giữ nguyên quy ước này để tái dùng matcher/loss/DDIM
        đã viết và đã kiểm cho nhánh CE-130 add.
        """
        cx, cy, w, h = bbox_px
        cx, cy, w, h = cx * scale, cy * scale, w * scale, h * scale
        target_w, target_h = self.target_size
        return torch.tensor([
            (cx / target_w) * 2 - 1,
            (cy / target_h) * 2 - 1,
            (w / target_w) * 2 - 1,
            (h / target_h) * 2 - 1,
        ], dtype=torch.float32)

    @staticmethod
    def _xyxy_to_cxcywh(b):
        x1, y1, x2, y2 = b
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)

    def _pad_to_proposals(self, gt_norm):
        """Pad/crop tập box về đúng num_proposals, theo công thức
        prepare_diffusion_concat của DiffusionDet (detector.py:370) đã chuyển
        sang không gian [-1,1] của CE-Loc."""
        N = self.num_proposals
        num_gt = gt_norm.shape[0]
        if num_gt < N:
            # DiffusionDet: randn/6 + 0.5 trong [0,1]; ánh xạ qua v*2-1 thành
            # randn/3 quanh 0 trong [-1,1] (đã kiểm trùng phân phối).
            placeholder = torch.randn(N - num_gt, 4) / 3.0
            # Sàn w/h: box suy biến trong [-1,1] là -1 (không phải 0), nên
            # tương đương của DiffusionDet min=1e-4 là 1e-4*2-1.
            placeholder[:, 2:] = torch.clamp(placeholder[:, 2:], min=1e-4 * 2 - 1)
            boxes = torch.cat([gt_norm, placeholder], dim=0)
            mask = torch.cat([torch.ones(num_gt, dtype=torch.bool),
                              torch.zeros(N - num_gt, dtype=torch.bool)])
        elif num_gt > N:
            keep = torch.randperm(num_gt)[:N]
            boxes, mask = gt_norm[keep], torch.ones(N, dtype=torch.bool)
        else:
            boxes, mask = gt_norm, torch.ones(N, dtype=torch.bool)
        return boxes, mask

    def __getitem__(self, idx):
        rec = self.samples[idx]
        img = Image.open(rec["img_path"]).convert("RGB")
        original_size = img.size
        canvas, scale = self.resize_and_pad(img)
        img_t = F.to_tensor(canvas)  # [3, H, W]

        boxes_cxcywh = [self._xyxy_to_cxcywh(b) for b in rec["boxes_xyxy"]]
        gt_norm = torch.stack([self._normalize_bbox(b, scale) for b in boxes_cxcywh])

        sample = {
            "pixel_values": img_t,
            "text": rec["text"],
            # Variant (a)/(b): đúng 1 box, lấy ngẫu nhiên trong tập — mỗi epoch
            # ảnh này sẽ dạy model một vật khác nhau của cùng class.
            "bbox": gt_norm[torch.randint(len(gt_norm), (1,)).item()],
            "scale": scale,
            "original_size": original_size,
            "img_id": rec["img_id"],
            "num_boxes": len(gt_norm),
        }
        if self.num_proposals > 1:
            boxes, mask = self._pad_to_proposals(gt_norm)
            sample["boxes"] = boxes
            sample["box_mask"] = mask
        return sample
