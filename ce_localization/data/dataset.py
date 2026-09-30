"""Dữ liệu: ảnh gốc CE-130 (`ground_truth.jpg`) + mọi box của lớp đó.

Đầu vào y như CE-Loc gốc (`refs/repos/Count-Editing/CE-LocModel/data/dataset.py:27-46`):
  scale = min(T/W, T/H) ; nw, nh = int(W·scale), int(H·scale)
  resize (nw, nh) BILINEAR -> dán góc TRÊN-TRÁI lên canvas ĐEN T×T
KHÔNG augmentation (CE-Loc gốc không có). Khác CE-Loc gốc đúng một chỗ: chuẩn hoá mean/std
ImageNet, vì BatchNorm của R-50 bị đóng băng (docs/EXPERIMENT_ALPHA.md mục 2.1).

Box: `all_bboxes` xyxy pixel ẢNH GỐC (qua `scan_ce130`: dedupe theo ảnh, KHÔNG trừ
`inpainted_bboxes`, bỏ box suy biến) -> nhân `scale` -> kẹp vào vùng ảnh thật [0,nw]×[0,nh]
-> bỏ box rỗng sau kẹp. Box ra là xyxy PIXEL CANVAS (tuyệt đối), đúng quy ước head DiffusionDet.

CE-130 cao 384, rộng >= 384 nên nw = T và phần đệm LUÔN ở đáy; nhưng code không giả định điều
đó — `valid_hw = (nh, nw)` đi kèm mỗi ảnh.

ALPHA3 (`density` khác None): nối density [0,1] (letterbox NEAREST, `data/density.py`) làm kênh
thứ 4 -> ảnh [4,T,T]. Kênh density KHÔNG chuẩn hoá mean/std (như CE-Loc gốc: chỉ /255). Chế độ
`mix` rút bản density theo RNG seed (seed, epoch, chỉ số ảnh): tái lập khi resume, không phụ thuộc
số worker (cạm bẫy 12). `epoch` do vòng train gán TRƯỚC khi tạo DataLoader của epoch đó.

BETA (`targets="point"`, docs/EXPERIMENT_BETA.md): đích train `boxes` là BOX GIẢ dựng từ điểm density
(`data/points.py`: điểm × scale letterbox, bỏ điểm ngoài vùng thật, cỡ giả kNN theo pixel canvas,
kẹp theo `nh`); box GT KHÔNG vào đích. `gt_boxes` (mọi chế độ) = box GT, CHỈ để chấm.
"""

import glob
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ce_localization.data.density import MODES, letterbox_density, load_density_levels
from ce_localization.data.points import pseudo_boxes, pseudo_sizes
from ce_localization.utils.box_ops_np import filter_degenerate

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "TARGETS", "scan_ce130", "letterbox", "scale_boxes", "scale_points",
           "CE130Dataset", "collate", "to_device"]

TARGETS = ("box", "point")

# Chuẩn ImageNet của torchvision (weight IMAGENET1K_V1 được học trên ảnh chuẩn hoá thế này).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _read_annotation(branch_dir):
    """`fixed_annotation.json` chỉ có ở val/test -> không có thì dùng `annotation.json`."""
    for name in ("fixed_annotation.json", "annotation.json"):
        path = os.path.join(branch_dir, name)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    return None


def scan_ce130(root, split):
    """`<root>/<split>/<iid>_b*/` -> list ảnh (dedupe theo iid: mọi nhánh cùng ảnh gốc như nhau nên
    lấy nhánh đầu, KIỂM TRƯỚC khi đọc JSON — đọc hết ~3 nhánh/ảnh từng làm khởi động mất 13–14 phút
    trên đĩa server). Mỗi phần tử: image_id, img_path, boxes_xyxy_px (`all_bboxes`, KHÔNG trừ
    `inpainted_bboxes`, bỏ 14/37.110 box suy biến w/h <= 0), text (`class_based_caption`)."""
    by_image = {}
    for br in sorted(glob.glob(os.path.join(root, split, "*"))):
        iid = os.path.basename(br).split("_b")[0]
        if iid in by_image:
            continue
        ann = _read_annotation(br)
        if ann is not None:
            by_image[iid] = (br, ann)
    items = []
    for iid, (br, ann) in sorted(by_image.items()):
        img_path = os.path.join(br, "ground_truth.jpg")
        if not os.path.exists(img_path):
            continue
        boxes, _ = filter_degenerate(np.asarray(ann.get("all_bboxes", []), dtype=np.float64).reshape(-1, 4))
        items.append({"image_id": iid, "img_path": img_path, "boxes_xyxy_px": boxes,
                      "text": ann.get("class_based_caption", "")})
    return items


def letterbox(img, target=512):
    """PIL RGB -> (canvas uint8 [T,T,3], scale, nw, nh). Y như `resize_and_pad` của CE-Loc gốc."""
    w, h = img.size
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.zeros((target, target, 3), dtype=np.uint8)            # đệm ĐEN như bản gốc
    canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), resample=Image.BILINEAR), dtype=np.uint8)
    return canvas, scale, nw, nh


def scale_boxes(boxes_xyxy_px, scale, nw, nh):
    """Box xyxy pixel ảnh gốc -> xyxy pixel canvas, kẹp vào vùng ảnh thật, bỏ box rỗng."""
    b = np.asarray(boxes_xyxy_px, dtype=np.float64).reshape(-1, 4) * scale
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, nw)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, nh)
    keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    return b[keep]


def scale_points(points_px, scale, nw, nh):
    """Điểm (x, y) pixel ảnh gốc -> pixel canvas; bỏ điểm nằm ngoài vùng ảnh thật [0,nw)×[0,nh)."""
    p = np.asarray(points_px, dtype=np.float64).reshape(-1, 2) * scale
    keep = (p[:, 0] >= 0) & (p[:, 0] < nw) & (p[:, 1] >= 0) & (p[:, 1] < nh)
    return p[keep]


def normalize(canvas_uint8):
    """uint8 HWC -> float32 CHW, [0,1] rồi chuẩn hoá ImageNet."""
    x = canvas_uint8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


class CE130Dataset(Dataset):
    """Một phần tử = một ẢNH (đã dedupe) với mọi box của lớp trong ảnh."""

    def __init__(self, root, split, image_size=512, density=None, density_index=None, seed=0,
                 targets="box", points=None, pseudo=None):
        self.items = scan_ce130(root, split)
        self.image_size = image_size
        if targets not in TARGETS:
            raise ValueError(f"targets {targets!r} không thuộc {TARGETS}")
        if targets == "point":
            if points is None or pseudo is None:
                raise ValueError("targets='point' cần points (PointTable) và pseudo (cfg data.pseudo_size)")
            miss = [it["image_id"] for it in self.items if it["image_id"] not in points]
            if miss:
                raise ValueError(f"{len(miss)} ảnh {split} không có điểm density (vd. {miss[:5]})")
        self.targets, self.points, self.pseudo = targets, points, pseudo
        if density is not None:
            if density not in MODES:
                raise ValueError(f"density {density!r} không thuộc {MODES}")
            if density_index is None:
                raise ValueError("density cần density_index")
            miss = [it["image_id"] for it in self.items if it["image_id"] not in density_index]
            if miss:
                raise ValueError(f"{len(miss)} ảnh {split} không có density (vd. {miss[:5]})")
        self.density, self.density_index, self.seed = density, density_index, seed
        self.epoch = 0

    def __len__(self):
        return len(self.items)

    def classes(self):
        return sorted({it["text"] for it in self.items})

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["img_path"]).convert("RGB")
        canvas, scale, nw, nh = letterbox(img, self.image_size)
        gt = scale_boxes(it["boxes_xyxy_px"], scale, nw, nh)
        boxes = gt if self.targets == "box" else self._point_targets(it["image_id"], scale, nw, nh)
        x = normalize(canvas)
        kind = None
        if self.density is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i]) if self.density == "mix" else None
            rel, kind = self.density_index.pick(it["image_id"], self.density, rng)
            den = (np.zeros((self.image_size, self.image_size), dtype=np.float32) if rel is None else
                   letterbox_density(load_density_levels(self.density_index.path(rel)), nw, nh, self.image_size))
            x = np.concatenate([x, den[None]], axis=0)
        return {
            "image": torch.from_numpy(x),
            "boxes": torch.from_numpy(boxes).float(),                    # ĐÍCH train, xyxy pixel canvas
            "gt_boxes": torch.from_numpy(gt).float(),                    # box GT, CHỈ để chấm
            "valid_hw": (nh, nw),
            "text": it["text"],
            "image_id": it["image_id"],
            "density_kind": kind,                                        # None | full | partial | empty
        }

    def _point_targets(self, iid, scale, nw, nh):
        """BETA: điểm density -> box giả xyxy pixel canvas (cỡ giả kNN kẹp [min_frac, max_frac]·nh)."""
        ps = self.pseudo
        pts = scale_points(self.points[iid], scale, nw, nh)
        s = pseudo_sizes(pts, ps["knn"], ps["beta"], ps["min_frac"] * nh, ps["max_frac"] * nh)
        return pseudo_boxes(pts, s)


def collate(batch):
    """Mọi ảnh cùng T×T nên stack thẳng; box giữ dạng list (số box mỗi ảnh khác nhau)."""
    nh = torch.tensor([b["valid_hw"][0] for b in batch], dtype=torch.float32)
    nw = torch.tensor([b["valid_hw"][1] for b in batch], dtype=torch.float32)
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "gt_boxes": [b["gt_boxes"] for b in batch],
        # (w, h, w, h) của VÙNG ẢNH THẬT = `images_whwh` của DiffusionDet
        "whwh": torch.stack([nw, nh, nw, nh], dim=1),
        "valid_hw": torch.stack([nh, nw], dim=1).long(),
        "text": [b["text"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
        "density_kind": [b["density_kind"] for b in batch],
    }


def to_device(batch, dev):
    nb = dev.type == "cuda"
    out = dict(batch)
    for k in ("images", "whwh", "valid_hw"):
        out[k] = batch[k].to(dev, non_blocking=nb)
    for k in ("boxes", "gt_boxes"):
        out[k] = [b.to(dev, non_blocking=nb) for b in batch[k]]
    return out
