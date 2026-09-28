"""`ObjectPlacementDataset` của CE-Loc gốc (data/dataset.py) viết lại, thêm `use_density=False`
(không đọc file density) và cache `uint8` giải mã sẵn. Tiền xử lý dùng chung
`celoc_vision.resize_and_pad` (y hệt bản gốc).

Box: target_bbox cxcywh pixel ảnh gốc -> nhân scale -> chuẩn hoá [-1,1] theo canvas 512, KỂ CẢ
w, h (norm_w = w/512*2 - 1) như bản gốc.

Ảnh trả về là `uint8` [3,T,T] (density [1,T,T]); `to_model_input` đưa lên thiết bị rồi mới
`/255`. Bản gốc chia /255 trên CPU bằng float32; phép chia float32 là IEEE làm tròn đúng nên
kết quả trùng từng bit, còn DataLoader chuyển ít hơn 4×.

CACHE (`build_cache`): giải mã PNG + resize + độn MỘT lần, ghi đúng mảng uint8 mà dataset tạo
ra vào memmap `images.u8` [N,T,T,3] (+ `density.u8` [N,T,T]). Train đọc memmap thay vì giải mã
PNG — Kaggle 4 vCPU nghẽn ở giải mã (bản density giải mã 2 PNG/mẫu, chậm hơn 1,4×).
"""

import json
import os
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ce_localization.legacy.celoc_vision import TARGET, resize_and_pad

META = "meta.json"


def list_images(root):
    return sorted(f for f in os.listdir(os.path.join(root, "images")) if f.endswith((".jpg", ".png", ".jpeg")))


class ObjectPlacementDataset(Dataset):
    def __init__(self, root_dir, target=TARGET, use_density=True, cache_dir=None):
        self.root_dir = root_dir
        self.target = target
        self.use_density = use_density
        self.image_dir = os.path.join(root_dir, "images")
        self.density_dir = os.path.join(root_dir, "density")
        self.annot_dir = os.path.join(root_dir, "annotation")
        self.files = list_images(root_dir)
        self.cache_dir = cache_dir
        self._mm = None
        if cache_dir:
            meta = json.load(open(os.path.join(cache_dir, META)))
            if meta["files"] != self.files or meta["target"] != target:
                raise ValueError(f"cache {cache_dir} không khớp {root_dir} (danh sách file / target)")
            if use_density and not meta["density"]:
                raise ValueError(f"cache {cache_dir} không có density — build lại có density")
            self.scales = meta["scales"]

    def __len__(self):
        return len(self.files)

    def parse_annotation(self, filename):
        with open(os.path.join(self.annot_dir, os.path.splitext(filename)[0] + ".json")) as f:
            data = json.load(f)
        return data["class"], data.get("target_bbox", [0.0, 0.0, 0, 0])

    def load_canvas(self, idx, use_density=None):
        """Giải mã PNG -> (ảnh uint8 [T,T,3], density uint8 [T,T] hoặc None, scale), như bản gốc."""
        use_density = self.use_density if use_density is None else use_density
        filename = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        if use_density:
            den = Image.open(os.path.join(self.density_dir, os.path.splitext(filename)[0] + ".png")).convert("L")
        else:
            den = Image.new("L", img.size, 0)            # không dùng; chỉ để resize_and_pad chạy
        img_p, den_p, scale = resize_and_pad(img, den, self.target)
        return (np.array(img_p, dtype=np.uint8),                         # bản sao ghi được (asarray của PIL chỉ đọc)
                np.array(den_p, dtype=np.uint8) if use_density else None, scale)

    def _cached(self, idx):
        if self._mm is None:                             # mở trong từng worker DataLoader
            n, T = len(self.files), self.target
            self._mm = {"images": np.memmap(os.path.join(self.cache_dir, "images.u8"), np.uint8, "r",
                                            shape=(n, T, T, 3))}
            if self.use_density:
                self._mm["density"] = np.memmap(os.path.join(self.cache_dir, "density.u8"), np.uint8, "r",
                                                shape=(n, T, T))
        img = np.array(self._mm["images"][idx])                  # bản sao ghi được (không trỏ vào memmap)
        den = np.array(self._mm["density"][idx]) if self.use_density else None
        return img, den, self.scales[idx]

    def __getitem__(self, idx):
        img, den, scale = self._cached(idx) if self.cache_dir else self.load_canvas(idx)
        class_name, (cx, cy, w, h) = self.parse_annotation(self.files[idx])
        T = self.target
        box = torch.tensor([(cx * scale / T) * 2 - 1, (cy * scale / T) * 2 - 1,
                            (w * scale / T) * 2 - 1, (h * scale / T) * 2 - 1], dtype=torch.float32)
        item = {"pixel_values": torch.from_numpy(img).permute(2, 0, 1), "text": class_name, "bbox": box,
                "scale": scale, "index": idx}
        if self.use_density:
            item["density_map"] = torch.from_numpy(den)[None]
        return item


def to_model_input(batch, dev, use_density):
    """uint8 -> float32 /255 TRÊN thiết bị (trùng từng bit với `to_tensor` của bản gốc)."""
    rgb = batch["pixel_values"].to(dev, non_blocking=True).float().div_(255.0)
    den = batch["density_map"].to(dev, non_blocking=True).float().div_(255.0) if use_density else None
    return rgb, den


# ----------------------------------------------------------------------------- cache

def _fill(job):
    root, cache_dir, use_density, target, lo, hi = job
    ds = ObjectPlacementDataset(root, target=target, use_density=use_density)
    n = len(ds.files)
    imgs = np.memmap(os.path.join(cache_dir, "images.u8"), np.uint8, "r+", shape=(n, target, target, 3))
    dens = (np.memmap(os.path.join(cache_dir, "density.u8"), np.uint8, "r+", shape=(n, target, target))
            if use_density else None)
    scales = []
    for i in range(lo, hi):
        img, den, s = ds.load_canvas(i)
        imgs[i] = img
        if use_density:
            dens[i] = den
        scales.append(s)
    imgs.flush()
    if dens is not None:
        dens.flush()
    return lo, scales


def cache_bytes(n, use_density, target=TARGET):
    return n * target * target * (4 if use_density else 3)


def build_cache(root, cache_dir, use_density=True, target=TARGET, workers=4, log=print):
    """Ghi memmap uint8 cho mọi ảnh của `root`. Idempotent: meta đã khớp thì bỏ qua. Ghi meta
    SAU CÙNG nên cache dở dang không bao giờ bị coi là xong."""
    os.makedirs(cache_dir, exist_ok=True)
    files = list_images(root)
    mp = os.path.join(cache_dir, META)
    if os.path.exists(mp):
        m = json.load(open(mp))
        if m["files"] == files and m["target"] == target and (m["density"] or not use_density):
            log(f"cache đã có: {cache_dir} ({len(files)} ảnh, density {m['density']})")
            return
        os.remove(mp)
    n = len(files)
    np.memmap(os.path.join(cache_dir, "images.u8"), np.uint8, "w+", shape=(n, target, target, 3)).flush()
    if use_density:
        np.memmap(os.path.join(cache_dir, "density.u8"), np.uint8, "w+", shape=(n, target, target)).flush()
    chunk = 256
    jobs = [(root, cache_dir, use_density, target, lo, min(lo + chunk, n)) for lo in range(0, n, chunk)]
    scales, done = [None] * n, 0
    pool = Pool(workers) if workers > 0 else None
    for lo, sc in (pool.imap_unordered(_fill, jobs) if pool else map(_fill, jobs)):
        scales[lo:lo + len(sc)] = sc
        done += len(sc)
        if done % (chunk * 20) < chunk or done == n:
            log(f"  cache {done}/{n}")
    if pool:
        pool.close()
        pool.join()
    with open(mp, "w") as f:
        json.dump({"files": files, "target": target, "density": use_density, "scales": scales}, f)
    log(f"cache xong: {cache_dir} ({n} ảnh, {cache_bytes(n, use_density, target) / 2**30:.1f} GB)")
