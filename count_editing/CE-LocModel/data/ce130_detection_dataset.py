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


def _iou_xyxy(a, b):
    """IoU giữa hai box [x1,y1,x2,y2] pixel. Dùng để nhận diện box của vật đã
    inpaint — không so bằng đúng vì toạ độ lệch vài pixel giữa hai lần chạy
    detector."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


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
        self._n_dropped_inpainted = 0
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

            # BỎ các vật ĐƯỢC THÊM VÀO bằng inpainting.
            #
            # Bộ dữ liệu này sinh ra cho bài toán ADD của CE-Loc gốc:
            #   ground_truth.jpg      = ảnh ban đầu (ví dụ 115 con cừu)
            #   inpainted_turn_3.png  = ảnh sau khi thêm 3 con  -> 118 con
            #   inpainted_bboxes      = vị trí 3 con MỚI thêm
            #   all_bboxes            = 118 box, annotation của ảnh ĐÃ THÊM
            #
            # Nhánh detection đưa vào model `ground_truth.jpg` (115 con) nhưng
            # `all_bboxes` mô tả 118 con -> 3 box thừa nằm ở chỗ ảnh đầu vào
            # TRỐNG. Model bị dạy "có vật ở đây" trong khi không có gì -> nó học
            # sinh box vào vùng trống, đúng bài toán add chứ không phải detect.
            #
            # Đo được (2026-09-04): 98,7% box trong inpainted_bboxes có IoU>0,5
            # với một box của all_bboxes, và vùng đó chênh 40-62/255 pixel giữa
            # hai ảnh (vùng ngẫu nhiên: 5,4) -> chúng thật sự bị sửa. Trên toàn
            # bộ dataset, 7,5% target là vật không tồn tại trong ảnh đầu vào.
            #
            # Khớp bằng IoU chứ không so bằng đúng: detector chạy lại trên ảnh
            # inpainted cho toạ độ lệch vài pixel (đo được một cặp IoU 0,825).
            added = data.get("inpainted_bboxes", [])
            if added:
                keep = [b for b in boxes
                        if max((_iou_xyxy(b, a) for a in added), default=0.0) <= 0.5]
                n_drop = len(boxes) - len(keep)
                self._n_dropped_inpainted += n_drop
                boxes = keep
                if not boxes:
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
            + (f" | BỎ {self._n_dropped_inpainted} box của vật đã inpaint "
               f"(không có trong ground_truth.jpg)" if self._n_dropped_inpainted else "")
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

    def _pad_to_proposals(self, gt_norm, valid_wh=None):
        """Pad/crop tập box về đúng num_proposals, theo công thức
        prepare_diffusion_concat của DiffusionDet (detector.py:370) đã chuyển
        sang không gian [-1,1] của CE-Loc.

        valid_wh: (fw, fh) — phần canvas thật sự là ẢNH, tính theo phân số. Ảnh
        được resize giữ tỉ lệ rồi pad vào góc trên-trái, nên phần còn lại của
        canvas là ĐEN (đo được: ảnh 469x384 -> 93px dưới cùng, 18% chiều cao).
        Placeholder rải theo randn có thể rơi hẳn vào vùng đen đó, dạy model
        rằng "vật có thể ở chỗ không có ảnh". Truyền valid_wh để giới hạn tâm
        placeholder trong vùng ảnh thật.
        """
        N = self.num_proposals
        num_gt = gt_norm.shape[0]
        if num_gt < N:
            n_ph = N - num_gt
            # TÂM: giữ nguyên DiffusionDet (randn/6 + 0.5 trong [0,1], tức
            # randn/3 quanh 0 trong [-1,1]) — rải đều khắp ảnh là hợp lý.
            placeholder = torch.randn(n_ph, 4) / 3.0

            # KÍCH THƯỚC: KHÔNG dùng công thức DiffusionDet ở đây.
            #
            # `randn/6 + 0.5` cho w/h trung bình 0,5 — tức box rộng NỬA ẢNH. Trên
            # COCO thì hợp lý (vật chiếm ~0,3-0,5 ảnh), nhưng CE-130 có vật rất
            # nhỏ: trung vị 0,098 x 0,120 (đo trên 7.054 box thật). Placeholder
            # to gấp ~5x vật thật.
            #
            # Hậu quả đo được (2026-09-04): với 48,5 GT trên N=300 slot, 84% slot
            # là placeholder. Chúng KHÔNG vào loss (Hungarian chỉ match GT thật),
            # nhưng CÓ vào x_start của q_sample, nên model nhận đầu vào mà 84% là
            # box to bằng nửa ảnh và phải khử nhiễu tất cả -> học prior "box điển
            # hình thì to và ở giữa". Nhìn thấy rõ trên ảnh visualize: box dồn vào
            # vùng trống, không bám vật.
            #
            # Thay bằng phân phối lấy từ chính thống kê vật thật của ảnh này. Dùng
            # log-normal quanh trung vị của các box thật (kích thước vật luôn
            # dương và lệch phải), để placeholder nằm TRONG phân phối target.
            if num_gt > 0:
                # gt_norm ở hệ [-1,1] với size mã hoá (2f-1) -> phân số ảnh = (v+1)/2
                real_wh = (gt_norm[:, 2:] + 1.0) / 2.0
                med = real_wh.median(dim=0).values.clamp(min=1e-3)
            else:
                med = torch.tensor([0.098, 0.120])  # trung vị toàn CE-130
            # sigma 0,5 trong không gian log ~ nhân/chia 1,65 quanh trung vị
            frac_wh = (med.log() + torch.randn(n_ph, 2) * 0.5).exp().clamp(1e-4, 1.0)
            placeholder[:, 2:] = frac_wh * 2.0 - 1.0   # về hệ [-1,1]

            # Giữ TÂM placeholder trong vùng ảnh thật, không cho rơi vào dải pad đen.
            if valid_wh is not None:
                fw, fh = valid_wh
                # tâm ở hệ [-1,1]: phân số 0..fw ứng với -1 .. (2*fw-1)
                placeholder[:, 0] = placeholder[:, 0].clamp(-1.0, 2.0 * fw - 1.0)
                placeholder[:, 1] = placeholder[:, 1].clamp(-1.0, 2.0 * fh - 1.0)
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
            # phần canvas thật sự là ảnh (phần còn lại là pad đen)
            valid_wh = (original_size[0] * scale / self.target_size[0],
                        original_size[1] * scale / self.target_size[1])
            boxes, mask = self._pad_to_proposals(gt_norm, valid_wh=valid_wh)
            sample["boxes"] = boxes
            sample["box_mask"] = mask
        return sample
