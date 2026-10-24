#!/usr/bin/env python3
"""Cache token CLIP ViT frozen của ảnh train (`data/clip_cache.py`) cho config `--config`: ~75 KB / mẫu fp16 (B/32) ⇒ train 19.998 mẫu
≈ 1,5 GB. Ảnh đọc như train (cache `build_add_cache` nếu có). Chạy MỘT GPU, trước train; train nhận `--clip-cache <out>`.
  python tools/build_clip_cache.py --config config/delta/delta6.yaml --turn-index ../data/turn_index.json --samples ../data/samples \
      --data-root ../data/all_phase2_V2 --add-cache /kaggle/temp/add_cache_raw --out /kaggle/temp/clip_cache
"""

import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.clip_cache import build_clip_cache  # noqa: E402
from ce_localization.data.turns import AddCache, CE130AddDataset, TurnIndex, collate_add  # noqa: E402
from ce_localization.models.box_policy import build_model  # noqa: E402
from ce_localization.utils.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--samples", default="../data/samples")
    ap.add_argument("--data-root", default="../data/all_phase2_V2")
    ap.add_argument("--add-cache", default=None, help="cache ảnh của build_add_cache (raw đọc nhanh nhất)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="chỉ N mẫu train đầu (test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = AddCache(a.add_cache) if a.add_cache else None
    ds = CE130AddDataset(TurnIndex(a.turn_index), a.data_root, a.samples, "train", cfg["data"]["image_size"], cache=cache)
    if a.limit:
        ds.keys = ds.keys[:a.limit]
    model = build_model(cfg, pretrained_backbone=False).to(dev).eval()
    if bool(cfg["training"].get("amp")) and dev.type == "cuda":
        model = model.to(memory_format=torch.channels_last)              # như train (CLIP kéo theo)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.workers, collate_fn=collate_add,
                        pin_memory=dev.type == "cuda")
    print(f"[clip cache] {len(ds)} mẫu train | CLIP {cfg['model']['refiner']['clip_model_name']} | {dev} -> {a.out}", flush=True)
    n = build_clip_cache(model, loader, a.out, cfg, log=lambda m: print(m, flush=True))
    gb = os.path.getsize(os.path.join(a.out, "feats.npy")) / 2 ** 30
    print(f"[clip cache] xong {n} mẫu ({gb:.2f} GB) | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
