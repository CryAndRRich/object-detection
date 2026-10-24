"""Cache token CLIP ViT frozen của ảnh train (refiner DELTA4+): CLIP không train, ảnh train không augment ⇒ token mỗi mẫu cố định,
tính MỘT lần thay cho mỗi iter (Kaggle T4: bỏ CLIP forward + bicubic 512 -> 224 khỏi vòng train).

`<dir>/feats.npy` [N, g*g, C] (`np.lib.format`, đọc memmap — mỗi tiến trình / worker mở riêng, KHÔNG nạp cả file vào RAM) +
`meta.json` (khoá -> hàng, CLIP, vân tay weight CLIP, lớp ViT, cỡ canvas, AMP). Token tính ĐÚNG đường train (`to_device_add` ->
`ClipBoxRefiner.clip_features`, autocast fp16 khi `training.amp`); AMP lưu fp16 (CLIP chạy fp16 sẵn: sai khác cỡ làm tròn fp16), không AMP
lưu fp32. Eval không dùng cache.
"""

import json
import os
import time

import numpy as np
import torch

__all__ = ["ClipCache", "build_clip_cache"]


class ClipCache:
    def __init__(self, cache_dir):
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.path = os.path.join(cache_dir, "feats.npy")
        self.rows = self.meta["rows"]
        self._arr = None

    def __getstate__(self):                      # gửi sang worker DataLoader: KHÔNG kèm memmap (pickle memmap = chép cả file)
        return {**self.__dict__, "_arr": None}

    def __contains__(self, key):
        return key in self.rows

    def get(self, key):
        """-> tensor [g*g, C] (bản sao ghi được, ~75 KB fp16 với B/32)."""
        if self._arr is None:
            self._arr = np.load(self.path, mmap_mode="r")
        return torch.from_numpy(np.array(self._arr[self.rows[key]]))

    def check(self, model, cfg, keys):
        """Cache phải đúng CLIP (vân tay weight `model.clip_sha[0]`), lớp ViT, canvas, AMP của lần train này và có mọi khoá `keys`."""
        r = model.refiner
        want = {"clip_sha": bytes(model.clip_sha[0].cpu().tolist()).hex(), "vision_layer": r.vision_layer,
                "image_size": cfg["data"]["image_size"], "amp": bool(cfg["training"].get("amp")) and next(model.parameters()).is_cuda}
        bad = {k: (self.meta.get(k), v) for k, v in want.items() if self.meta.get(k) != v}
        if bad:
            raise ValueError(f"clip cache {os.path.dirname(self.path)} không khớp lần train này (cache, cần): {bad} — dựng lại "
                             f"(tools/build_clip_cache.py)")
        miss = [k for k in keys if k not in self.rows]
        if miss:
            raise ValueError(f"clip cache thiếu {len(miss)} / {len(keys)} khoá train, vd {miss[:3]}")


@torch.no_grad()
def build_clip_cache(model, loader, out_dir, cfg, log=print):
    """Ghi token CLIP (`model.refiner.clip_features`) của mọi mẫu `loader` (KHÔNG xáo) vào `out_dir`. Ghi từng batch thẳng vào memmap,
    `meta.json` ghi CUỐI (có meta = cache đầy đủ). -> số mẫu."""
    from ce_localization.data.turns import to_device_add
    dev = next(model.parameters()).device
    amp = bool(cfg["training"].get("amp")) and dev.type == "cuda"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "feats.npy")
    keys, arr, i, t0 = loader.dataset.keys, None, 0, time.time()
    for bi, batch in enumerate(loader):
        images = to_device_add(batch, dev)["images"]
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            f = model.refiner.clip_features(images[:, :3])
        f = f.to(torch.float16 if amp else torch.float32).cpu().numpy()
        if arr is None:
            arr = np.lib.format.open_memmap(path, mode="w+", dtype=f.dtype, shape=(len(keys),) + f.shape[1:])
        arr[i:i + len(f)] = f
        i += len(f)
        if bi % 50 == 0 or i == len(keys):
            el = time.time() - t0
            log(f"  [clip cache] {i}/{len(keys)} | {el:.0f}s | còn ~{el / i * (len(keys) - i):.0f}s")
    arr.flush()
    del arr
    meta = {"clip_model_name": cfg["model"]["refiner"].get("clip_model_name"), "clip_sha": bytes(model.clip_sha[0].cpu().tolist()).hex(),
            "vision_layer": model.refiner.vision_layer, "image_size": cfg["data"]["image_size"], "amp": amp,
            "dtype": "float16" if amp else "float32", "rows": {k: j for j, k in enumerate(keys)}}
    with open(os.path.join(out_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    return i
