"""DataLoader cho CE-130 — dùng chung cho train và eval."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ce_localization.data.ce130_dataset import PatchCache, normalize_for_clip

__all__ = ["TorchWrap", "collate", "seed_worker", "model_inputs", "make_loader"]


class TorchWrap(Dataset):
    """Bọc `CE130Detection` (numpy) thành torch Dataset.

    Có `cache` thì trả patch/text token CLIP đã tính sẵn và KHÔNG giải mã ảnh (tiết kiệm
    ~17 ms/ảnh; cache nhanh hơn chạy CLIP mỗi batch ~4,3 lần).
    """

    def __init__(self, ds, cache=None):
        self.ds = ds
        self.cache = cache

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        m = self.ds.__getitem__(i, need_image=self.cache is None)
        out = {"boxes": torch.from_numpy(m["boxes"]).float(), "text": m["text"],
               "valid_h": m["valid_h"], "image_id": m["image_id"]}
        if self.cache is None:
            out["pixel_values"] = torch.from_numpy(normalize_for_clip(m["image"]))
        else:
            patch, text = self.cache.get(m["image_id"], m["text"], m["flipped"])
            out["patch_raw"] = torch.from_numpy(patch)
            out["text_raw"] = torch.from_numpy(text)
        return out


def collate(batch):
    """Số box mỗi ảnh khác nhau -> giữ dạng list, không pad."""
    out = {k: [b[k] for b in batch] for k in ("boxes", "text", "valid_h", "image_id")}
    for k in ("pixel_values", "patch_raw", "text_raw"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def seed_worker(worker_id):
    """Seed lại `np.random.Generator` RIÊNG của dataset trong từng worker.

    Worker fork từ tiến trình chính nên mọi worker nhận CÙNG trạng thái RNG và tung cùng
    một dãy đồng xu lật ảnh — PyTorch chỉ seed lại RNG toàn cục, không đụng Generator
    riêng. `torch.initial_seed()` trong worker đã khác nhau theo worker_id.
    """
    info = torch.utils.data.get_worker_info()
    inner = getattr(info.dataset, "ds", None)
    if inner is not None and hasattr(inner, "rng"):
        inner.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)


def model_inputs(batch, dev):
    """kwargs cho model: token đã cache, hoặc ảnh thô + text.

    `non_blocking=True` cho copy host->device chồng lấn tính toán khi loader `pin_memory`.
    """
    nb = dev.type == "cuda"
    if "patch_raw" in batch:
        return {"patch_raw": batch["patch_raw"].to(dev, non_blocking=nb),
                "text_raw": batch["text_raw"].to(dev, non_blocking=nb)}
    return {"pixel_values": batch["pixel_values"].to(dev, non_blocking=nb),
            "texts": batch["text"]}


def make_loader(ds, cache_dir, split, batch_size, num_workers, dev, train):
    """Dataset + (tuỳ chọn) cache -> DataLoader. `train=True`: xáo + bỏ batch lẻ cuối."""
    cache = PatchCache(cache_dir, split) if cache_dir else None
    return DataLoader(TorchWrap(ds, cache), batch_size=batch_size, shuffle=train,
                      drop_last=train, num_workers=num_workers, collate_fn=collate,
                      pin_memory=dev.type == "cuda", persistent_workers=num_workers > 0,
                      worker_init_fn=seed_worker)
