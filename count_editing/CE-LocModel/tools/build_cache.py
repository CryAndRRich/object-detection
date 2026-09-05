#!/usr/bin/env python3
"""Cache patch token CLIP ra memmap fp16.

CHỈ CHẠY SAU KHI `profile_and_memory.py` cho thấy CLIP > 30 % thời gian.

Cache 2 BẢN (gốc + lật ngang) vì KHÔNG flip được trên token đã cache: ViT trộn
thông tin toàn cục qua 12 layer nên token (i,j) không còn là "feature của riêng ô
(i,j)". Dung lượng: 1.911 x 2 x 1024 x 768 fp16 ~ 6,0 GB.

Băng thông: 1,57 MB/ảnh -> 50 MB/batch(32) -> ~3 GB/epoch. Ổn trên SSD; chú ý nếu
là network FS.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.clip_encoder import CLIPConditionEncoder  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-flip", action="store_true", help="chỉ cache ảnh gốc")
    ap.add_argument("--limit", type=int, default=None, help="cắt nhỏ để thử")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(a.out, exist_ok=True)

    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    if a.limit:
        ds.items = ds.items[: a.limit]
    enc = CLIPConditionEncoder(cfg["model"]["clip_name"], cfg["model"]["d_model"],
                               cfg["data"]["image_size"], freeze=True).to(dev).eval()

    n_img = len(ds)
    n_ver = 1 if a.no_flip else 2
    n_tok = enc.num_patches
    d = enc.vision.config.hidden_size
    path = os.path.join(a.out, f"{a.split}_patch.f16")
    print(f"[cache] {n_img} ảnh x {n_ver} bản x {n_tok} token x {d} "
          f"= {n_img*n_ver*n_tok*d*2/1e9:.2f} GB -> {path}", flush=True)

    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(n_img, n_ver, n_tok, d))
    ids = []

    # Text embedding: mỗi CLASS chỉ một vector (input là 1 từ), nên cache theo class
    # thay vì theo ảnh — vài chục vector, bỏ luôn chi phí tokenize mỗi batch.
    lop = sorted({it["text"] for it in ds.items})
    txt = enc.encode_text_raw(lop, dev).cpu().numpy().astype(np.float16)  # [C,1,d_txt]
    np.save(os.path.join(a.out, f"{a.split}_text.npy"), txt)
    print(f"[cache] {len(lop)} class -> text embedding {txt.shape}", flush=True)

    for i0 in range(0, n_img, a.batch_size):
        idx = range(i0, min(i0 + a.batch_size, n_img))
        mau = [ds[i] for i in idx]
        ids += [m["image_id"] for m in mau]

        for v in range(n_ver):
            imgs = [m["image"][:, ::-1].copy() if v == 1 else m["image"] for m in mau]
            px = torch.stack([torch.from_numpy(normalize_for_clip(x)) for x in imgs]).to(dev)
            mm[list(idx), v] = enc.encode_image_raw(px).cpu().numpy().astype(np.float16)

        if i0 % (a.batch_size * 20) == 0:
            print(f"  {i0}/{n_img}", flush=True)

    mm.flush()
    with open(os.path.join(a.out, f"{a.split}_meta.json"), "w") as f:
        json.dump({"image_ids": ids, "shape": [n_img, n_ver, n_tok, d],
                   "dtype": "float16", "image_size": cfg["data"]["image_size"],
                   "clip": cfg["model"]["clip_name"], "classes": lop,
                   "n_ver": n_ver}, f)
    print(f"[cache] xong: {path}", flush=True)


if __name__ == "__main__":
    main()
