#!/usr/bin/env python3
"""Train lại CE-Loc GỐC (bài add) — cùng công thức với checkpoint `weights/celoc/best_model.pth`,
có tuỳ chọn BỎ density (`--no-density`: vision 3 kênh, không đọc file density).

Công thức suy từ checkpoint gốc (train_w_args.py không có trong repo):
  AdamW(model.parameters(), lr 5e-5, wd 0,01 mặc định), CosineAnnealingLR(T_max=300) bước theo
  EPOCH (lr ở epoch 113 khớp 3,4203e-5), batch 32, không drop_last (71.250 bước / 114 epoch =
  625 = ceil(19.998/32)), 300 epoch, không AMP / clip grad / EMA, best = loss train TB nhỏ nhất.
Thêm so với bản gốc (không đổi phép toán): last.pt mỗi epoch + --resume, tiến độ + ETA, eval
định kỳ trên tập con cố định của test (log hội tụ; chọn checkpoint vẫn theo loss như gốc).

  export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache TORCH_HOME=/mnt/disk1/aiotlab/haitn/torch_cache
  python ../tools/run_on_free_gpu.py -- legacy/train.py --save-dir checkpoints/celoc_nodensity --no-density
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.legacy.celoc_data import ObjectPlacementDataset  # noqa: E402
from ce_localization.legacy.celoc_model import ObjectPlacementPolicy  # noqa: E402
from ce_localization.legacy.eval import print_table, run_eval, summarize  # noqa: E402
from ce_localization.utils.checkpoint import CheckpointManager, rng_state, set_rng_state  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--no-density", action="store_true", help="bỏ density map (vision 3 kênh)")
    ap.add_argument("--data", default="../data/samples/train")
    ap.add_argument("--eval-data", default="../data/samples/test")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=10, help="0 = tắt")
    ap.add_argument("--eval-n", type=int, default=500, help="số ảnh test cố định cho eval định kỳ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = {k: v for k, v in vars(args).items() if k not in ("resume", "num_workers", "save_dir")}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_density = not args.no_density
    ckm = CheckpointManager(args.save_dir)
    if ckm.has_last() and not args.resume:
        sys.exit(f"{ckm.last_path} đã có — thêm --resume để chạy tiếp, hoặc đổi --save-dir")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    ds = ObjectPlacementDataset(args.data, use_density=use_density)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    eval_ds = ObjectPlacementDataset(args.eval_data, use_density=use_density)
    idx = np.random.default_rng(12345).permutation(len(eval_ds))[: args.eval_n]
    eval_loader = DataLoader(Subset(eval_ds, sorted(idx.tolist())), batch_size=64, shuffle=False,
                             num_workers=args.num_workers)

    model = ObjectPlacementPolicy(use_density=use_density).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{fmt_time(time.time() - t0)}] {dev} | density {use_density} | train {len(ds)} ảnh, "
          f"{len(loader)} bước/epoch | eval {len(idx)} ảnh test | {n_train / 1e6:.2f}M tham số train | cfg {cfg}",
          flush=True)

    start, best_loss, history = 0, float("inf"), []
    if args.resume and ckm.has_last():
        ck = ckm.load_last()
        if ck["config"] != cfg:
            sys.exit(f"config lệch checkpoint:\n  ckpt {ck['config']}\n  now  {cfg}")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start, best_loss, history = ck["epoch"] + 1, ck["best_loss"], ck["history"]
        set_rng_state(ck["rng"], gen)
        print(f"resume từ epoch {ck['epoch']} (best loss {best_loss:.5f})", flush=True)

    t_train = time.time()
    for epoch in range(start, args.epochs):
        model.train()
        te, total = time.time(), 0.0
        for k, batch in enumerate(loader):
            den = batch["density_map"].to(dev, non_blocking=True) if use_density else None
            loss = model.compute_loss(batch["pixel_values"].to(dev, non_blocking=True), den, list(batch["text"]),
                                      batch["bbox"].to(dev, non_blocking=True), generator=gen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            if epoch == start and (k + 1) % 100 == 0:                   # epoch đầu: báo tốc độ sớm
                el = time.time() - te
                print(f"  epoch {epoch} bước {k + 1}/{len(loader)} loss {total / (k + 1):.5f} | "
                      f"{el / (k + 1):.3f} s/bước | ETA epoch {fmt_time(el / (k + 1) * (len(loader) - k - 1))}",
                      flush=True)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg = total / len(loader)
        rec = {"epoch": epoch, "loss": avg, "lr": lr, "epoch_s": time.time() - te}

        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            model.eval()
            out, _ = run_eval(model, eval_loader, 30, seed=0)
            rec["eval"] = summarize(out)
        history.append(rec)
        is_best = avg < best_loss
        best_loss = min(best_loss, avg)
        ckm.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                  "loss": avg, "best_loss": best_loss, "config": cfg, "history": history, "rng": rng_state(gen)},
                 is_best)
        with open(os.path.join(args.save_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

        done = epoch + 1 - start
        el = time.time() - t_train
        ev = ""
        if "eval" in rec:
            e = rec["eval"]
            ev = (f" | eval mock best {e['mock']['best_iou']:.4f} hit50 {e['mock']['hit50']:.3f} "
                  f"| ddpm best {e['ddpm']['best_iou']:.4f} hit50 {e['ddpm']['hit50']:.3f} mean {e['ddpm']['mean_iou']:.4f}")
        print(f"[{fmt_time(time.time() - t0)} | ETA {fmt_time(el / done * (args.epochs - epoch - 1))}] "
              f"epoch {epoch + 1}/{args.epochs} loss {avg:.5f}{' *best' if is_best else ''} lr {lr:.3e} "
              f"({fmt_time(rec['epoch_s'])}){ev}", flush=True)

    if history and "eval" in history[-1]:
        print_table(history[-1]["eval"], f"eval cuối ({len(idx)} ảnh test)")
    print(f"[{fmt_time(time.time() - t0)}] xong -> {args.save_dir}")


if __name__ == "__main__":
    main()
