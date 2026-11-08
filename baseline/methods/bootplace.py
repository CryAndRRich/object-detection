"""BootPlace cho bài ADD: tiền xử lý (như `refs/BOOTPLACE/datasets`), dataset finetune CE-130 train, trainer, suy luận K box.

Ảnh nền: như bản gốc, mọi ảnh coi như đã resize méo về khung `FRAME` 960×540 (`Cityscapes._load_image`) rồi mới resize cạnh ngắn
(train: ngẫu nhiên `train_scales`, cạnh dài <= `train_max_size`; test: `test_size`, cạnh dài <= `test_max_size`) — gộp thành MỘT lần
resize từ ảnh gốc tới cỡ cuối, + chuẩn hoá mean / std của BootPlace, làm TRÊN GPU (`prep_images`, dùng chung train / test). Dataset trả
uint8 [3,H,W] GỐC (+ lật ngang 0,5 lúc train). Box đích chuẩn hoá theo ảnh (không đổi qua resize / khung).
Patch: <= 100 ảnh vật 64×64 xếp dọc, alpha 255 (vật có sẵn không có mask), ô thừa 0 trước chuẩn hoá 4 kênh. Train: patch = cắt
`ground_truth.jpg` ở từng lỗ; test: một vật cùng lớp có sẵn (`fg_select`). Box cảnh (`cMLP`): vật đang có, **xyxy PIXEL khung 960×540**
như bản gốc cả train lẫn test (transforms gốc không đụng `scene_boxes`; `location/*.txt` pixel) — `cMLP` nén chúng thành MỘT vô hướng nhân
vào đặc trưng mọi query, đầu vào chuẩn hoá [0, 1] làm vô hướng ≈ 0 ⇒ box hết phụ thuộc query (bản port đầu, ZETA mục 11.3). <= Q box (quá
thì ngẫu nhiên lúc train, Q box to nhất lúc test). Đích train = mọi lỗ tới lượt t, nhãn 0.
Suy luận: K box = 10 vùng xác suất lớp 0 cao nhất xếp theo điểm khớp patch TĂNG dần (luật `test.py` của tác giả bài ⇒ box[0] = box bản đó
chọn), rồi các vùng xác suất lớp 0 cao tiếp theo.
"""

import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from baseline.data import int_box, pick_objects, sample_rng
from baseline.models.bootplace import build_bootplace, build_criterion
from baseline.models.detr import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh, nested

__all__ = ["FRAME", "resize_size", "prep_images", "patch_canvas", "scene_boxes", "target_boxes", "BootPlaceTrain", "rank_queries", "sample_inputs", "predict", "Trainer"]

MEAN, STD = (0.440, 0.440, 0.440, 0.0), (0.320, 0.319, 0.318, 0.5)
PATCH, MAX_PATCHES, TOPK_CLIP = 64, 100, 10
FRAME = (960, 540)               # khung ảnh (W, H) của BootPlace gốc: mọi ảnh resize méo về đây; box cảnh pixel trong khung này


def resize_size(wh, size, max_size=None):
    """(W, H) -> (new_h, new_w): cạnh ngắn = size, cạnh dài <= max_size (như `transforms.resize` bản gốc)."""
    w, h = wh
    if max_size is not None and max(w, h) / min(w, h) * size > max_size:
        size = int(round(max_size * min(w, h) / max(w, h)))
    return (int(size * h / w), size) if w < h else (size, int(size * w / h))


def _stat(dev, n):
    return (torch.tensor(MEAN[:n], device=dev).view(-1, 1, 1), torch.tensor(STD[:n], device=dev).view(-1, 1, 1))


def prep_images(imgs_u8, sizes, dev):
    """list uint8 [3,H,W] + list (h, w) -> (tensor chuẩn hoá đệm 0 [B,3,h,w], mask). Resize bilinear antialias trên GPU."""
    m, s = _stat(dev, 3)
    out = [(F.interpolate(x.to(dev, non_blocking=True)[None].float(), size=hw, mode="bilinear", align_corners=False,
                          antialias=True)[0] / 255 - m) / s for x, hw in zip(imgs_u8, sizes)]
    return nested(out)


def patch_canvas(patches_u8, dev):
    """list uint8 [t,64,64,3] (t <= 100) -> [B, 4, 100·64, 64] chuẩn hoá."""
    m, s = _stat(dev, 4)
    out = torch.zeros(len(patches_u8), 4, MAX_PATCHES * PATCH, PATCH, device=dev)
    for i, p in enumerate(patches_u8):
        t = len(p)
        if t > MAX_PATCHES:
            raise ValueError(f"{t} patch > {MAX_PATCHES}")
        if t:
            out[i, :3, :t * PATCH] = p.to(dev, non_blocking=True).permute(3, 0, 1, 2).reshape(3, t * PATCH, PATCH).float() / 255
            out[i, 3, :t * PATCH] = 1.0
    return (out - m) / s


def _patches(img, boxes):
    """Cắt + resize 64×64 (PIL bilinear) -> uint8 [t,64,64,3]."""
    return torch.from_numpy(np.stack([np.asarray(img.crop(int_box(b, img.size)).resize((PATCH, PATCH), Image.BILINEAR))
                                      for b in boxes]) if len(boxes) else np.zeros((0, PATCH, PATCH, 3), np.uint8))


def scene_boxes(boxes_xyxy, wh, q, rng=None):
    """Box vật đang có (xyxy pixel ảnh gốc cỡ `wh`) -> xyxy pixel khung `FRAME` [<= q, 4]. Quá q: ngẫu nhiên q (`rng`) | q box diện tích
    lớn nhất."""
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if len(b) > q:
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        b = b[np.sort(rng.choice(len(b), q, replace=False) if rng is not None else np.argsort(-area, kind="stable")[:q])]
    sx, sy = FRAME[0] / wh[0], FRAME[1] / wh[1]
    return torch.as_tensor(b * np.array([sx, sy, sx, sy]), dtype=torch.float32)


def target_boxes(boxes_xyxy, wh):
    """Box đích (lỗ GT, xyxy pixel ảnh gốc cỡ `wh`) -> cxcywh chuẩn hoá [0, 1] như `pred_boxes` (KHÁC box cảnh `scene_boxes`)."""
    w, h = wh
    b = torch.as_tensor(np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4), dtype=torch.float32)
    return box_xyxy_to_cxcywh(b) / torch.tensor([w, h, w, h], dtype=torch.float32)


def _u8(img):
    return torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()


class BootPlaceTrain(Dataset):
    """Mẫu `data.train_items` -> dict uint8 / box chuẩn hoá (gom bằng `collate` = list, không đệm trên CPU)."""

    def __init__(self, items, num_queries, scales, max_size, flip=0.5):
        self.items, self.q, self.scales, self.max_size, self.flip = items, num_queries, list(scales), max_size, flip

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        W, H = img.size
        holes = np.asarray(it["holes"], dtype=np.float64).reshape(-1, 4)
        objs = np.asarray(it["objects"], dtype=np.float64).reshape(-1, 4)
        patches = _patches(Image.open(it["gt"]).convert("RGB"), holes)
        x = _u8(img)
        if random.random() < self.flip:
            x, patches = x.flip(-1), patches.flip(2)
            fl = lambda b: b[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([W, 0, W, 0])  # noqa: E731
            holes, objs = fl(holes), fl(objs)
        rng = np.random.default_rng(random.getrandbits(32))
        return {"image": x, "size": resize_size(FRAME, random.choice(self.scales), self.max_size), "patches": patches,
                "boxes": target_boxes(holes, (W, H)), "scene": scene_boxes(objs, (W, H), self.q, rng)}


def collate(batch):
    return batch


class Trainer:
    """Finetune BootPlace (bản tác giả bài: bỏ khoá `drop_keys` của checkpoint, hệ số loss theo config). `step` = một bước tối ưu trên
    batch / rank, cộng dồn `micro` ảnh một lượt, đồng bộ grad bằng `sync` trước bước. Chọn best theo loss train (có giám sát)."""

    select = "loss"

    def __init__(self, cfg, dev, model):
        f = cfg["finetune"]
        self.cfg, self.dev, self.model = cfg, dev, model
        self.crit = build_criterion(cfg["loss"]).to(dev)
        bb = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
        rest = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
        self.params = rest + bb
        fused = dev.type == "cuda"
        self.optimizers = [torch.optim.AdamW([{"params": rest}, {"params": bb, "lr": f["lr_backbone"]}], lr=f["lr"],
                                             weight_decay=f["weight_decay"], fused=fused)]
        self.clip = f.get("grad_clip") or 0.0

    def dataset(self, items, **_):
        f = self.cfg["finetune"]
        return BootPlaceTrain(items, self.model.num_queries, f["train_scales"], f["train_max_size"]), collate

    def train(self):
        self.model.train()

    def context(self, batch, world):
        """Số box TB / GPU của cả bước (chia loss box, như DETR all-reduce `num_boxes`)."""
        n = torch.tensor(float(sum(len(s["boxes"]) for s in batch)), device=self.dev)
        if world > 1:
            torch.distributed.all_reduce(n)
        return max(n.item() / world, 1.0)

    def backward(self, chunk, num_boxes, frac, amp, scaler):
        """Micro-batch: forward (autocast), loss fp32, backward có thang. `frac` = phần ảnh của micro trong batch / rank. -> thành phần."""
        x, mask = prep_images([s["image"] for s in chunk], [s["size"] for s in chunk], self.dev)
        if amp:
            x = x.contiguous(memory_format=torch.channels_last)
        patches = patch_canvas([s["patches"] for s in chunk], self.dev)
        tg = [{"boxes": s["boxes"].to(self.dev), "labels": torch.zeros(len(s["boxes"]), dtype=torch.int64, device=self.dev)} for s in chunk]
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=amp):
            out = self.model(x, mask, patches, [s["scene"].to(self.dev) for s in chunk])
        out = {k: [{kk: vv.float() for kk, vv in a.items()} for a in v] if k == "aux_outputs" else v.float() for k, v in out.items()}
        box, mean, parts = self.crit(out, tg, num_boxes)
        loss = box + mean * frac
        scaler.scale(loss).backward()
        return {"loss": loss.detach(), **{k: v.detach() * frac for k, v in parts.items()}}

    def state_dict(self):
        return {"model": self.model.state_dict()}

    def load_state_dict(self, sd):
        self.model.load_state_dict(sd["model"])

    def weights(self):
        return {"model": self.model.state_dict()}


def rank_queries(logits, clip_logits, k):
    """logits [Q, C+1], clip_logits [Q] (điểm khớp patch 0) -> K chỉ số vùng (docstring module)."""
    prob = logits.softmax(-1)[:, 0]
    order = torch.argsort(prob, descending=True, stable=True)
    top = order[:TOPK_CLIP]
    top = top[torch.argsort(clip_logits.softmax(0)[top], stable=True)]
    return torch.cat([top, order[TOPK_CLIP:]])[:k]


def sample_inputs(model, s, p, seed=0, dev="cpu"):
    """Mẫu test -> (đối số forward (x, mask, patches, [box cảnh]), (W, H) ảnh gốc, chỉ số vật làm patch `fg`)."""
    img = Image.open(s["image"]).convert("RGB")
    W, H = img.size
    objs = np.asarray(s["objects"], dtype=np.float64).reshape(-1, 4)
    fg = pick_objects(objs, 1, p["fg_select"], sample_rng(seed, s["image_id"]))[0]
    x, mask = prep_images([_u8(img)], [resize_size(FRAME, p["test_size"], p["test_max_size"])], dev)
    args = (x, mask, patch_canvas([_patches(img, objs[fg:fg + 1])], dev), [scene_boxes(objs, (W, H), model.num_queries).to(dev)])
    return args, (W, H), fg


@torch.no_grad()
def predict(model, samples, k, p, seed=0, dev="cpu", log=print, log_every=500):
    """-> list {"image_id", "boxes" [K,4] pixel ảnh gốc (thứ tự xếp hạng), "fg", "prob"}. `p` = khối `predict` của config."""
    model.eval()
    q = model.num_queries
    if k > q:
        raise ValueError(f"K {k} > num_queries {q}")
    out, t0 = [], time.time()
    for n, s in enumerate(samples, 1):
        args, (W, H), fg = sample_inputs(model, s, p, seed, dev)
        o = model(*args)
        idx = rank_queries(o["pred_logits"][0], o["pred_clip"][0, :, 0], k)
        b = box_cxcywh_to_xyxy(o["pred_boxes"][0, idx]) * torch.tensor([W, H, W, H], dtype=torch.float32, device=dev)
        out.append({"image_id": s["image_id"], "boxes": b.cpu().double().numpy().round(2).tolist(), "fg": fg,
                    "prob": o["pred_logits"][0, idx].softmax(-1)[:, 0].cpu().numpy().round(5).tolist()})
        if log_every and (n % log_every == 0 or n == len(samples)):
            el = time.time() - t0
            log(f"    [bootplace {n:5d}/{len(samples)}] {el / n * 1000:.0f} ms/ảnh | còn ~{el / n * (len(samples) - n):.0f} s")
    return out


def build(cfg, train_backbone=False):
    return build_bootplace({**cfg["model"], "train_backbone": train_backbone})
