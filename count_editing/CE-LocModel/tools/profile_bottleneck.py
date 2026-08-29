"""Đo tách bạch chi phí mỗi phần của vòng train, để biết 41 phút/epoch nằm ở đâu.
Không train thật, không lưu gì. Chạy vài trăm batch rồi in bảng.
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import ObjectPlacementDataset
from models.diffusion_module import ObjectPlacementPolicy
from train import load_config


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="resnet18_cnn")
    p.add_argument("--batches", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_cache", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_cfg = load_config("config/default.yaml")
    model_cfg = load_config(os.path.join("config", "variants", f"{args.variant}.yaml"))

    ds = ObjectPlacementDataset(train_cfg["training"]["data"]["train_path"],
                                use_cache=args.use_cache)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)
    print(f"dataset={len(ds)} batch_size={args.batch_size} "
          f"num_workers={args.num_workers} cache={'on' if args.use_cache else 'off'} "
          f"batches/epoch={len(loader)}")

    model = ObjectPlacementPolicy(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    t_data = t_h2d = t_vis = t_text = t_unet = t_bwd = 0.0
    n = 0
    it = iter(loader)

    # warmup
    for _ in range(3):
        b = next(it)
        loss = model.compute_loss(b["pixel_values"].to(device), b["density_map"].to(device),
                                  b["text"], b["bbox"].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    sync()

    wall0 = time.perf_counter()
    while n < args.batches:
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter(); t_data += t1 - t0

        rgb = b["pixel_values"].to(device, non_blocking=True)
        density = b["density_map"].to(device, non_blocking=True)
        gt = b["bbox"].to(device, non_blocking=True)
        sync(); t2 = time.perf_counter(); t_h2d += t2 - t1

        opt.zero_grad()
        vis = model.vision_encoder(rgb, density)
        sync(); t3 = time.perf_counter(); t_vis += t3 - t2

        txt = model.text_encoder(b["text"])
        sync(); t4 = time.perf_counter(); t_text += t4 - t3

        cond = torch.cat([vis, txt], dim=-1)
        B = rgb.shape[0]
        t = torch.randint(0, model.num_timesteps, (B,), device=device)
        noise = torch.randn_like(gt)
        ab = model.get_alpha_bar(t)
        noisy = torch.sqrt(ab) * gt + torch.sqrt(1 - ab) * noise
        pred = model.predict_noise(noisy, t, cond)
        loss = torch.nn.functional.mse_loss(pred, noise)
        sync(); t5 = time.perf_counter(); t_unet += t5 - t4

        loss.backward(); opt.step()
        sync(); t6 = time.perf_counter(); t_bwd += t6 - t5
        n += 1

    wall = time.perf_counter() - wall0
    steps = len(loader)
    print(f"\nĐo trên {n} batch, tổng {wall:.1f}s -> {wall/n*1000:.0f} ms/batch")
    print(f"Ước tính 1 epoch ({steps} batch): {wall/n*steps/60:.1f} phút\n")
    rows = [("DataLoader (chờ I/O+decode)", t_data), ("Chuyển CPU->GPU", t_h2d),
            ("Vision encoder fwd", t_vis), ("CLIP text fwd", t_text),
            ("UNet1D fwd + loss", t_unet), ("Backward + optimizer", t_bwd)]
    tot = sum(r[1] for r in rows)
    print(f"{'phần':<30}{'ms/batch':>10}{'%':>8}{'phút/epoch':>13}")
    print("-" * 61)
    for name, v in sorted(rows, key=lambda r: -r[1]):
        print(f"{name:<30}{v/n*1000:>10.1f}{v/tot*100:>7.1f}%{v/n*steps/60:>13.1f}")


if __name__ == "__main__":
    main()
