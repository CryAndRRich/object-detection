#!/usr/bin/env python3
"""Cache patch token CLIP (fp16 memmap) + embedding text theo lớp — bỏ ViT khỏi vòng train.

Lưu HAI bản (gốc + lật ngang): KHÔNG lật được token đã cache, vì ViT trộn thông tin toàn cục
qua 12 tầng nên token (i,j) không còn là "đặc trưng riêng ô (i,j)". Eval không lật => dùng
`--no-flip` cho split eval, tốn nửa dung lượng.

Dung lượng @1024px (4096 token x 768 x fp16): train 24 GB, val 11,4 GB, test (no-flip) 4,9 GB.
~5 phút / 1.000 ảnh-bản trên A30 => chạy nền:

  cd object-detection/ce_localization
  export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
  LOG=/mnt/disk1/aiotlab/haitn/log/cache_train_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- tools/build_cache.py --split train \
      --image-size 1024 --batch-size 2 --out ../data/cache_clip_1024 > $LOG 2>&1 &
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from ce_localization.models.clip_encoder import CLIPConditionEncoder  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--no-flip", action="store_true", help="cache the original image only")
    ap.add_argument("--limit", type=int, default=None, help="shrink for a smoke test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--image-size", type=int, default=None,
                    help="ghi đè `data.image_size` của config. 1024 -> lưới 64x64 "
                         "(box CE-130 trung vị 1,96 -> 3,92 ô). Cache to gấp 4 và ViT "
                         "attention tốn 16x, nên GIẢM --batch-size theo.")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.image_size:
        # Ghi vào cfg chứ không truyền riêng: `image_size` đi vào CẢ dataset, encoder
        # lẫn meta.json. Đặt một chỗ thì ba nơi không thể lệch nhau.
        cfg["data"]["image_size"] = a.image_size
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
    print(f"[cache] {n_img} images x {n_ver} versions x {n_tok} tokens x {d} "
          f"= {n_img*n_ver*n_tok*d*2/1e9:.2f} GB -> {path}", flush=True)

    mm = np.memmap(path, dtype=np.float16, mode="w+", shape=(n_img, n_ver, n_tok, d))
    ids = []

    # Text embeddings: one vector per CLASS (the input is a single word), so cache
    # per class rather than per image — a few dozen vectors, and it also removes
    # the per-batch tokenisation cost.
    classes = sorted({it["text"] for it in ds.items})
    txt = enc.encode_text_raw(classes, dev).cpu().numpy().astype(np.float16)  # [C,1,d_txt]
    np.save(os.path.join(a.out, f"{a.split}_text.npy"), txt)
    print(f"[cache] {len(classes)} classes -> text embeddings {txt.shape}", flush=True)

    t0 = time.time()
    for i0 in range(0, n_img, a.batch_size):
        idx = range(i0, min(i0 + a.batch_size, n_img))
        samples = [ds[i] for i in idx]
        ids += [m["image_id"] for m in samples]

        for v in range(n_ver):
            imgs = [m["image"][:, ::-1].copy() if v == 1 else m["image"] for m in samples]
            px = torch.stack([torch.from_numpy(normalize_for_clip(x)) for x in imgs]).to(dev)
            mm[list(idx), v] = enc.encode_image_raw(px).cpu().numpy().astype(np.float16)

        if i0 % (a.batch_size * 20) == 0:
            el = time.time() - t0
            eta = el / max(i0 + len(list(idx)), 1) * (n_img - i0 - len(list(idx)))
            print(f"  {i0}/{n_img}  ({el/60:.1f} phút, còn ~{eta/60:.1f} phút)",
                  flush=True)

    mm.flush()
    with open(os.path.join(a.out, f"{a.split}_meta.json"), "w") as f:
        json.dump({"image_ids": ids, "shape": [n_img, n_ver, n_tok, d],
                   "dtype": "float16", "image_size": cfg["data"]["image_size"],
                   "clip": cfg["model"]["clip_name"], "classes": classes,
                   "n_ver": n_ver}, f)
    print(f"[cache] xong sau {(time.time()-t0)/60:.1f} phút: {path}", flush=True)


if __name__ == "__main__":
    main()
