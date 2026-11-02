"""HiddenObjects (ZETA3) cho bài ADD: ảnh letterbox 512 của CE-Loc, nhúng lớp từ CLIP text, dataset finetune (+ cache memmap), trainer,
suy luận top-K theo plausibility.

Ảnh: `ce_localization.data.turns.letterbox` (neo góc trên trái, đệm đen, s = min(S / W, S / H)) — cùng canvas lúc chấm; bản gốc center
crop vuông (mất hai mép ảnh). Chuẩn hoá ImageNet trên GPU. Box: xywh chuẩn hoá theo canvas S (box gốc × s / S); suy luận đổi ngược
(÷ s), không kẹp. Lớp: CLIP text `model.clip_text` (`pooler_output`, như CE-Loc) chuẩn hoá L2, tính một lần mỗi tên lớp.
Đích train = mọi lỗ tới lượt t (`holes[:t]`, 1–4 box), không augment (như bản gốc). Suy luận: K box xếp theo sigmoid(plausibility) giảm
dần ⇒ box[0] = box điểm cao nhất.

Cache (`--cache-dir`): ảnh letterbox uint8 [N, S, S, 3] trong một memmap trên đĩa (19.998 ảnh × 786 KB = 15,7 GB) — model nhẹ, giải mã PNG
mỗi bước thành nút thắt. Rank 0 dựng (nhiều tiến trình), rank khác chờ `meta.json`; thiếu đĩa ⇒ đọc PNG như thường.
"""

import hashlib
import json
import os
import shutil
import time
from multiprocessing import get_context

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from baseline.models.hiddenobjects import PlacementDETR, hungarian_loss, xywh_to_xyxy
from ce_localization.data.turns import letterbox
from ce_localization.models.text import encode_class_names

__all__ = ["build", "class_texts", "to_canvas", "to_orig", "ensure_cache", "HOTrain", "Trainer", "predict"]

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build(cfg):
    m = cfg["model"]
    model = PlacementDETR(**m)
    model.clip_text, model.image_size = m["clip_text"], m["image_size"]
    return model


def class_texts(names, model_name, dev):
    """list tên lớp -> ({tên: chỉ số}, tensor [n, 512] CLIP text chuẩn hoá L2 trên `dev`)."""
    names = sorted(set(names))
    emb = encode_class_names(names, model_name, device=str(dev))
    t = torch.stack([emb[n] for n in names]).float()
    return {n: i for i, n in enumerate(names)}, torch.nn.functional.normalize(t, dim=-1).to(dev)


def to_canvas(boxes_xyxy, wh, S):
    """Box xyxy pixel gốc [m,4] -> xywh chuẩn hoá theo canvas S (float32)."""
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4) * min(S / wh[0], S / wh[1]) / S
    return torch.from_numpy(np.concatenate([b[:, :2], b[:, 2:] - b[:, :2]], 1)).float()


def to_orig(xywh, wh, S):
    """xywh chuẩn hoá canvas [K,4] (tensor) -> xyxy pixel gốc (numpy float64)."""
    return xywh_to_xyxy(xywh.double()).cpu().numpy() * S / min(S / wh[0], S / wh[1])


def _canvas_u8(path, S):
    with Image.open(path) as im:
        return letterbox(im.convert("RGB"), S)[0]


def _ids_sha(items):
    return hashlib.sha1("\n".join(it["image_id"] for it in items).encode()).hexdigest()


_MM = {}


def _fill(args):
    """Tiến trình con: ghi ảnh [i0, i1) vào memmap (mở r+ một lần mỗi tiến trình)."""
    path, shape, paths, i0 = args
    if path not in _MM:
        _MM[path] = np.memmap(path, dtype=np.uint8, mode="r+", shape=shape)
    for j, p in enumerate(paths):
        _MM[path][i0 + j] = _canvas_u8(p, shape[1])
    return len(paths)


def ensure_cache(items, cache_dir, S, rank=0, log=print, procs=4, chunk=64, wait_s=5.0, timeout_h=3.0):
    """-> đường dẫn memmap [N,S,S,3] uint8 khớp `items` (đúng thứ tự), hoặc None (không cache: thiếu đĩa). Rank 0 dựng, rank khác chờ."""
    os.makedirs(cache_dir, exist_ok=True)
    path, meta_p = os.path.join(cache_dir, "images.u8"), os.path.join(cache_dir, "meta.json")
    want = {"n": len(items), "size": S, "ids": _ids_sha(items)}

    def meta():
        try:
            with open(meta_p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    m = meta()
    if m and {k: m.get(k) for k in want} == want:
        return path if m.get("status") == "ok" else None
    if rank != 0:
        t0 = time.time()
        while time.time() - t0 < timeout_h * 3600:
            m = meta()
            if m and {k: m.get(k) for k in want} == want:
                return path if m.get("status") == "ok" else None
            time.sleep(wait_s)
        raise TimeoutError(f"rank {rank}: chờ cache {cache_dir} quá {timeout_h} giờ")
    if os.path.exists(meta_p):
        os.remove(meta_p)
    need, free = len(items) * S * S * 3, shutil.disk_usage(cache_dir).free + (os.path.getsize(path) if os.path.exists(path) else 0)
    if free < need * 1.05:
        log(f"[cache] ⚠️ {cache_dir} trống {free / 2**30:.1f} GB < cần {need / 2**30:.1f} GB -> KHÔNG cache, đọc PNG mỗi bước")
        status = "skip"
    else:
        shape = (len(items), S, S, 3)
        np.memmap(path, dtype=np.uint8, mode="w+", shape=shape).flush()
        jobs = [(path, shape, [it["image"] for it in items[i:i + chunk]], i) for i in range(0, len(items), chunk)]
        log(f"[cache] dựng {len(items)} ảnh letterbox {S}² -> {path} ({need / 2**30:.1f} GB, {procs} tiến trình) …")
        t0, done, step = time.time(), 0, max(2000, chunk)
        if procs > 1 and len(jobs) > 1:
            with get_context("fork" if os.name == "posix" else "spawn").Pool(procs) as pool:
                for n in pool.imap_unordered(_fill, jobs):
                    done += n
                    if done // step != (done - n) // step or done == len(items):
                        el = time.time() - t0
                        log(f"[cache]   {done}/{len(items)} | {el:.0f}s | còn ~{el / done * (len(items) - done):.0f}s")
        else:
            for j in jobs:
                done += _fill(j)
        _MM.pop(path, None)
        log(f"[cache] xong {len(items)} ảnh trong {time.time() - t0:.0f}s")
        status = "ok"
    tmp = meta_p + ".part"
    with open(tmp, "w") as f:
        json.dump({**want, "status": status}, f)
    os.replace(tmp, meta_p)
    return path if status == "ok" else None


class HOTrain(Dataset):
    """Mẫu `data.train_items` -> {"image" uint8 [3,S,S], "cls" chỉ số lớp, "boxes" [m,4] xywh canvas chuẩn hoá}. `cache`: memmap
    [N,S,S,3] cùng thứ tự `items` (mở lười trong từng worker, chỉ đọc)."""

    def __init__(self, items, cls_idx, S, cache=None):
        self.items, self.cls_idx, self.S, self.cache, self._mm = items, cls_idx, S, cache, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        if self.cache:
            if self._mm is None:
                self._mm = np.memmap(self.cache, dtype=np.uint8, mode="r", shape=(len(self.items), self.S, self.S, 3))
            img = np.array(self._mm[i])
        else:
            img = _canvas_u8(it["image"], self.S)
        return {"image": torch.from_numpy(img).permute(2, 0, 1).contiguous(), "cls": self.cls_idx[it["class"]],
                "boxes": to_canvas(it["holes"], it["wh"], self.S)}


def collate(batch):
    return {"image": torch.stack([b["image"] for b in batch]), "cls": torch.tensor([b["cls"] for b in batch]),
            "boxes": [b["boxes"] for b in batch]}


def prep(x_u8, dev, channels_last=False):
    """uint8 [B,3,S,S] -> float chuẩn hoá ImageNet trên `dev`."""
    m = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    x = (x_u8.to(dev, non_blocking=True).float() / 255 - m) / s
    return x.contiguous(memory_format=torch.channels_last) if channels_last else x


def _strip_backbone(sd):
    return {k: v for k, v in sd.items() if not k.startswith("backbone.")}


class Trainer:
    """Finetune PlacementDETR: AdamW một nhóm (mọi tham số trừ backbone đóng băng), loss `hungarian_loss` TB theo mẫu (loss micro =
    Σ loss_i / batch-per-rank ⇒ sau `sync_grads` = TB batch toàn cục). Phần train không có BN ⇒ micro_batch không đổi kết quả. Chọn best
    theo loss train. Checkpoint không chứa backbone (dựng lại từ torchvision)."""

    select = "loss"
    fixed_size = True

    def __init__(self, cfg, dev, model):
        f = cfg["finetune"]
        self.cfg, self.dev, self.model, self.S = cfg, dev, model, cfg["model"]["image_size"]
        self.w = cfg["loss"]
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.optimizers = [torch.optim.AdamW(self.params, lr=f["lr"], weight_decay=f["weight_decay"], fused=dev.type == "cuda")]
        self.clip = f.get("grad_clip") or 0.0
        self.text = None

    def dataset(self, items, cache_dir=None, rank=0, log=print):
        idx, self.text = class_texts([it["class"] for it in items], self.model.clip_text, self.dev)
        log(f"[hiddenobjects] {len(idx)} lớp train (CLIP text {self.model.clip_text})")
        cache = ensure_cache(items, cache_dir, self.S, rank, log) if cache_dir else None
        return HOTrain(items, idx, self.S, cache), collate

    def train(self):
        self.model.train()

    def context(self, batch, world):
        return None

    def backward(self, chunk, ctx, frac, amp, scaler):
        x = prep(chunk["image"], self.dev, amp)
        text = self.text[chunk["cls"].to(self.dev)]
        with torch.autocast(device_type=self.dev.type, dtype=torch.float16, enabled=amp):
            boxes, logits = self.model(x, text)
        loss_i, parts = hungarian_loss(boxes.float(), logits.float(), chunk["boxes"], self.w["l1"], self.w["giou"], self.w["plaus"])
        loss = loss_i.mean() * frac
        scaler.scale(loss).backward()
        return {"loss": loss.detach(), **{k: v * frac for k, v in parts.items()}}

    def state_dict(self):
        return {"model": _strip_backbone(self.model.state_dict())}

    def load_state_dict(self, sd):
        missing, unexpected = self.model.load_state_dict(sd["model"], strict=False)
        bad = [k for k in missing if not k.startswith("backbone.")] + list(unexpected)
        if bad:
            raise RuntimeError(f"state dict HiddenObjects lệch: {bad[:5]}")

    def weights(self):
        return self.state_dict()


class _TestImages(Dataset):
    def __init__(self, samples, S):
        self.samples, self.S = samples, S

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return torch.from_numpy(_canvas_u8(self.samples[i]["image"], self.S)).permute(2, 0, 1).contiguous()


@torch.no_grad()
def predict(model, samples, k, p, seed=0, dev="cpu", log=print, log_every=500):
    """-> list {"image_id", "boxes" [K,4] pixel ảnh gốc (xếp theo điểm giảm dần), "score" [K]}. `p` = khối `predict` của config.
    Tất định (không ngẫu nhiên) ⇒ `seed` không dùng."""
    model.eval()
    dev = torch.device(dev)
    if k > model.num_queries:
        raise ValueError(f"K {k} > num_queries {model.num_queries}")
    S, n_img, nw = model.image_size, p.get("batch_images", 32), p.get("num_workers", 4)
    idx, text = class_texts([s["class"] for s in samples], model.clip_text, dev)
    amp = dev.type == "cuda"
    loader = DataLoader(_TestImages(samples, S), batch_size=n_img, num_workers=nw, pin_memory=amp)
    out, t0 = [], time.time()
    for x in loader:
        ss = samples[len(out):len(out) + len(x)]
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            boxes, logits = model(prep(x, dev, amp), text[torch.tensor([idx[s["class"]] for s in ss], device=dev)])
        score = torch.sigmoid(logits.float())
        order = torch.argsort(score, dim=1, descending=True, stable=True)[:, :k]
        for s, b, sc, o in zip(ss, boxes.float(), score, order):
            out.append({"image_id": s["image_id"], "boxes": to_orig(b[o], s["wh"], S).round(2).tolist(),
                        "score": sc[o].cpu().numpy().round(5).tolist()})
        if log_every and (len(out) // log_every != (len(out) - len(ss)) // log_every or len(out) == len(samples)):
            el = time.time() - t0
            log(f"    [hiddenobjects {len(out):5d}/{len(samples)}] {el / len(out) * 1000:.0f} ms/ảnh | còn ~{el / len(out) * (len(samples) - len(out)):.0f} s")
    return out
