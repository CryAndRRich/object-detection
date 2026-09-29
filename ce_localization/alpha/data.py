"""Dữ liệu ALPHA: ảnh gốc CE-130 (`ground_truth.jpg`) + mọi box của lớp đó.

Đầu vào y như CE-Loc gốc (`refs/repos/Count-Editing/CE-LocModel/data/dataset.py:27-46`):
  scale = min(T/W, T/H) ; nw, nh = int(W·scale), int(H·scale)
  resize (nw, nh) BILINEAR -> dán góc TRÊN-TRÁI lên canvas ĐEN T×T
KHÔNG augmentation (CE-Loc gốc không có). Khác CE-Loc gốc đúng một chỗ: chuẩn hoá mean/std
ImageNet, vì BatchNorm của R-50 bị đóng băng (docs/EXPERIMENT_ALPHA.md mục 2.1).

Box: `all_bboxes` xyxy pixel ẢNH GỐC (qua `CE130Detection._scan`: dedupe theo ảnh, KHÔNG trừ
`inpainted_bboxes`, bỏ box suy biến) -> nhân `scale` -> kẹp vào vùng ảnh thật [0,nw]×[0,nh]
-> bỏ box rỗng sau kẹp. Box ra là xyxy PIXEL CANVAS (tuyệt đối), đúng quy ước head DiffusionDet.

CE-130 cao 384, rộng >= 384 nên nw = T và phần đệm LUÔN ở đáy; nhưng code không giả định điều
đó — `valid_hw = (nh, nw)` đi kèm mỗi ảnh.

ALPHA3 (`density` khác None): nối density [0,1] (letterbox NEAREST, `alpha/density.py`) làm kênh
thứ 4 -> ảnh [4,T,T]. Kênh density KHÔNG chuẩn hoá mean/std (như CE-Loc gốc: chỉ /255). Chế độ
`mix` rút bản density theo RNG seed (seed, epoch, chỉ số ảnh): tái lập khi resume, không phụ thuộc
số worker (cạm bẫy 12). `epoch` do vòng train gán TRƯỚC khi tạo DataLoader của epoch đó.
"""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ce_localization.alpha.density import MODES, letterbox_density, load_density_levels
from ce_localization.data.ce130_dataset import CE130Detection

__all__ = ["IMAGENET_MEAN", "IMAGENET_STD", "letterbox", "scale_boxes", "AlphaCE130",
           "collate", "to_device"]

# Chuẩn ImageNet của torchvision (weight IMAGENET1K_V1 được học trên ảnh chuẩn hoá thế này).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def normalize(canvas_uint8):
    """uint8 HWC -> float32 CHW, [0,1] rồi chuẩn hoá ImageNet."""
    x = canvas_uint8.astype(np.float32) / 255.0
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


class AlphaCE130(Dataset):
    """Một phần tử = một ẢNH (đã dedupe) với mọi box của lớp trong ảnh."""

    def __init__(self, root, split, image_size=512, density=None, density_index=None, seed=0):
        # target của CE130Detection không dùng ở đây: chỉ lấy danh sách ảnh + box từ `_scan`.
        self.ds = CE130Detection(root, split, target=image_size)
        self.items = self.ds.items
        self.image_size = image_size
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
        boxes = scale_boxes(it["boxes_xyxy_px"], scale, nw, nh)
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
            "boxes": torch.from_numpy(boxes).float(),                    # xyxy pixel canvas
            "valid_hw": (nh, nw),
            "text": it["text"],
            "image_id": it["image_id"],
            "density_kind": kind,                                        # None | full | partial | empty
        }


def collate(batch):
    """Mọi ảnh cùng T×T nên stack thẳng; box giữ dạng list (số box mỗi ảnh khác nhau)."""
    nh = torch.tensor([b["valid_hw"][0] for b in batch], dtype=torch.float32)
    nw = torch.tensor([b["valid_hw"][1] for b in batch], dtype=torch.float32)
    return {
        "images": torch.stack([b["image"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
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
    out["boxes"] = [b.to(dev, non_blocking=nb) for b in batch["boxes"]]
    return out
