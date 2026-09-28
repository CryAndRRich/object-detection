#!/usr/bin/env python3
"""Train lại CE-Loc GỐC (bài add) — cùng công thức với checkpoint `weights/celoc/best_model.pth`,
có tuỳ chọn BỎ density (`--no-density`: vision 3 kênh, không đọc file density).

Công thức suy từ checkpoint gốc (train_w_args.py không có trong repo):
  AdamW(model.parameters(), lr 5e-5, wd 0,01 mặc định), CosineAnnealingLR(T_max=300) bước theo
  EPOCH (lr ở epoch 113 khớp 3,4203e-5), batch 32, không drop_last (71.250 bước / 114 epoch =
  625 = ceil(19.998/32)), không AMP / clip grad / EMA, best = loss train TB nhỏ nhất.
  `--stop-epoch N`: giữ lịch 300 epoch nhưng dừng sau N epoch (best gốc ở epoch 113 cho thấy
  bản gốc nhiều khả năng cũng dừng sớm).

Hai cách chạy, CÙNG phép toán (batch TOÀN CỤC 32):
  1 GPU (server, qua run_on_free_gpu):  python legacy/train.py --save-dir ...
  nhiều GPU (Kaggle T4×2, DDP):          torchrun --standalone --nproc_per_node=2 legacy/train.py ...
  DDP chia 32 = 16/GPU; BatchNorm đổi thành SyncBatchNorm để thống kê vẫn trên đủ 32 ảnh; gradient
  trung bình 2 GPU = gradient của trung bình 32 (hai nửa bằng nhau, kể cả batch cuối 30 = 15+15).

Ngẫu nhiên tái lập theo epoch (resume khớp đúng): thứ tự dữ liệu seed theo (seed, epoch); t / nhiễu
diffusion seed theo (seed, epoch, rank).

Checkpoint: `last.pth` mỗi epoch + `best.pth`, định dạng pickle KHÔNG zip (Kaggle không bung ra
như với .pth dạng zip). Eval định kỳ trên tập con test cố định (chỉ để xem hội tụ).

  export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache TORCH_HOME=/mnt/disk1/aiotlab/haitn/torch_cache
  python ../tools/run_on_free_gpu.py -- legacy/train.py --save-dir checkpoints/celoc_nodensity --no-density
"""

import argparse
import datetime
import json
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.legacy.celoc_data import ObjectPlacementDataset, to_model_input  # noqa: E402
from ce_localization.legacy.celoc_model import ObjectPlacementPolicy  # noqa: E402
from ce_localization.legacy.eval import print_table, run_eval, summarize  # noqa: E402
from ce_localization.utils.checkpoint import CheckpointManager  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402

# không vào config của checkpoint: đổi được giữa các lần --resume mà không đổi phép toán
RUNTIME_ARGS = ("resume", "num_workers", "save_dir", "bench", "max_hours", "cudnn_benchmark", "cache_dir",
                "stop_epoch")


class PthCheckpoints(CheckpointManager):
    """last.pth / best.pth, pickle không zip."""
    LAST, BEST = "last.pth", "best.pth"

    @staticmethod
    def _atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp, _use_new_zipfile_serialization=False)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)


def setup_dist():
    """-> (rank, world, dev). torchrun đặt WORLD_SIZE/RANK/LOCAL_RANK; không có thì 1 tiến trình."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        dev = torch.device("cpu")
    dist.init_process_group("nccl" if dev.type == "cuda" else "gloo", timeout=datetime.timedelta(hours=1))
    return dist.get_rank(), world, dev


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bench(model, loader, optimizer, gen, dev, use_density, n):
    """Tách nút thắt: (1) CHỈ đọc dữ liệu, (2) CHỈ tính toán trên một batch cố định đã nằm trên
    GPU, với cudnn.benchmark tắt rồi bật."""
    model.train()
    t, it = time.time(), iter(loader)
    for k in range(n + 3):
        try:
            batch = next(it)
        except StopIteration:                                            # tập nhỏ: quay vòng
            it = iter(loader)
            batch = next(it)
        if k == 2:                                                       # bỏ 3 batch khởi động worker
            t = time.time()
    print(f"(1) CHỈ đọc dữ liệu: {(time.time() - t) / n:.3f} s/bước ({n} bước, {loader.num_workers} worker)",
          flush=True)
    rgb, den = to_model_input(batch, dev, use_density)
    box, text = batch["bbox"].to(dev), list(batch["text"])
    for flag in (False, True):
        torch.backends.cudnn.benchmark = flag
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        for k in range(n + 3):
            if k == 3:
                _sync(dev)
                t = time.time()
            loss = model(rgb, den, text, box, generator=gen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        _sync(dev)
        mem = torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0
        print(f"(2) CHỈ tính toán, cudnn.benchmark={flag}: {(time.time() - t) / n:.3f} s/bước, "
              f"đỉnh bộ nhớ {mem:.1f} GB", flush=True)
    print("-> bước train thật ≈ max(1), (2) khi worker đọc song song với GPU", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--no-density", action="store_true", help="bỏ density map (vision 3 kênh)")
    ap.add_argument("--data", default="../data/samples/train")
    ap.add_argument("--eval-data", default="../data/samples/test")
    ap.add_argument("--cache-dir", default=None, help="cache uint8 của --data (legacy/build_cache.py)")
    ap.add_argument("--epochs", type=int, default=300, help="độ dài lịch cosine (T_max)")
    ap.add_argument("--stop-epoch", type=int, default=0, help="dừng sau N epoch (0 = chạy hết --epochs)")
    ap.add_argument("--batch-size", type=int, default=32, help="batch TOÀN CỤC (chia đều cho các GPU)")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=8, help="worker DataLoader MỖI tiến trình")
    ap.add_argument("--eval-every", type=int, default=10, help="0 = tắt")
    ap.add_argument("--eval-n", type=int, default=500, help="số ảnh test cố định cho eval định kỳ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-hours", type=float, default=0,
                    help="dừng sạch sau epoch nếu epoch kế tiếp có thể vượt N giờ; 0 = tắt")
    ap.add_argument("--cudnn-benchmark", action="store_true",
                    help="cuDNN tự chọn thuật toán conv nhanh nhất (phép toán không đổi)")
    ap.add_argument("--bench", type=int, default=0,
                    help="đo riêng đọc dữ liệu / tính toán GPU trong N bước rồi thoát (không ghi gì)")
    args = ap.parse_args()
    cfg = {k: v for k, v in vars(args).items() if k not in RUNTIME_ARGS}
    rank, world, dev = setup_dist()
    main_proc = rank == 0
    log = (lambda *a: print(*a, flush=True)) if main_proc else (lambda *a: None)
    if args.batch_size % world:
        sys.exit(f"--batch-size {args.batch_size} không chia hết cho {world} GPU")
    if args.bench and world > 1:
        sys.exit("--bench chỉ chạy 1 tiến trình")
    use_density = not args.no_density
    stop = min(args.stop_epoch or args.epochs, args.epochs)
    ckm = PthCheckpoints(args.save_dir) if not args.bench else None
    if ckm is not None and ckm.has_last() and not args.resume:
        sys.exit(f"{ckm.last_path} đã có — thêm --resume để chạy tiếp, hoặc đổi --save-dir")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    t0 = time.time()

    ds = ObjectPlacementDataset(args.data, use_density=use_density, cache_dir=args.cache_dir)
    shuffle_gen = torch.Generator()
    sampler = (DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=False)
               if world > 1 else RandomSampler(ds, generator=shuffle_gen))
    loader = DataLoader(ds, batch_size=args.batch_size // world, sampler=sampler, num_workers=args.num_workers,
                        pin_memory=dev.type == "cuda", persistent_workers=args.num_workers > 0)
    eval_ds = ObjectPlacementDataset(args.eval_data, use_density=use_density)
    idx = np.random.default_rng(12345).permutation(len(eval_ds))[: args.eval_n]
    eval_loader = DataLoader(Subset(eval_ds, sorted(idx.tolist())), batch_size=64, shuffle=False,
                             num_workers=args.num_workers)

    model = ObjectPlacementPolicy(use_density=use_density).to(dev)
    if world > 1 and dev.type == "cuda":
        model.vision_encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.vision_encoder)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None) \
        if world > 1 else model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    gen = torch.Generator(device=dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[{fmt_time(time.time() - t0)}] {dev} × {world} tiến trình | density {use_density} | train {len(ds)} ảnh, "
        f"{len(loader)} bước/epoch, batch {args.batch_size // world}/GPU | cache {args.cache_dir} | "
        f"eval {len(idx)} ảnh test | {n_train / 1e6:.2f}M tham số train | dừng sau epoch {stop} | cfg {cfg}")

    if args.bench:
        gen.manual_seed(args.seed)
        return bench(model, loader, optimizer, gen, dev, use_density, args.bench)

    start, best_loss, history = 0, float("inf"), []
    if args.resume and ckm.has_last():
        ck = ckm.load_last()
        if ck["config"] != cfg:
            sys.exit(f"config lệch checkpoint:\n  ckpt {ck['config']}\n  now  {cfg}")
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        start, best_loss, history = ck["epoch"] + 1, ck["best_loss"], ck["history"]
        log(f"resume từ epoch {ck['epoch']} (best loss {best_loss:.5f})")

    t_train, slowest = time.time(), 0.0
    for epoch in range(start, stop):
        model.train()
        te, total = time.time(), torch.zeros((), device=dev, dtype=torch.float64)
        if world > 1:
            sampler.set_epoch(epoch)
        else:
            shuffle_gen.manual_seed(args.seed * 100003 + epoch)
        gen.manual_seed(args.seed * 100003 + epoch * 101 + rank)
        for k, batch in enumerate(loader):
            rgb, den = to_model_input(batch, dev, use_density)
            loss = net(rgb, den, list(batch["text"]), batch["bbox"].to(dev, non_blocking=True), generator=gen)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.detach()
            if epoch == start and (k + 1) % 100 == 0:                   # epoch đầu: báo tốc độ sớm
                el = time.time() - te
                log(f"  epoch {epoch} bước {k + 1}/{len(loader)} loss {total.item() / (k + 1):.5f} | "
                    f"{el / (k + 1):.3f} s/bước | ETA epoch {fmt_time(el / (k + 1) * (len(loader) - k - 1))}")
        if world > 1:
            dist.all_reduce(total)
            total /= world
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg = total.item() / len(loader)
        rec = {"epoch": epoch, "loss": avg, "lr": lr, "epoch_s": time.time() - te}
        is_best = avg < best_loss
        best_loss = min(best_loss, avg)

        halt = torch.zeros((), device=dev)
        if main_proc:
            if args.eval_every and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == stop):
                model.eval()
                out, _ = run_eval(model, eval_loader, 30, seed=0)
                rec["eval"] = summarize(out)
            history.append(rec)
            ckm.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                      "loss": avg, "best_loss": best_loss, "config": cfg, "history": history}, is_best)
            with open(os.path.join(args.save_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)

            done = epoch + 1 - start
            el = time.time() - t_train
            ev = ""
            if "eval" in rec:
                e = rec["eval"]
                ev = (f" | eval mock best {e['mock']['best_iou']:.4f} hit50 {e['mock']['hit50']:.3f} "
                      f"| ddpm best {e['ddpm']['best_iou']:.4f} hit50 {e['ddpm']['hit50']:.3f} "
                      f"mean {e['ddpm']['mean_iou']:.4f}")
            log(f"[{fmt_time(time.time() - t0)} | ETA {fmt_time(el / done * (stop - epoch - 1))}] "
                f"epoch {epoch + 1}/{stop} (lịch {args.epochs}) loss {avg:.5f}{' *best' if is_best else ''} "
                f"lr {lr:.3e} ({fmt_time(rec['epoch_s'])}){ev}")
            slowest = max(slowest, time.time() - te)                     # kể cả eval + ghi checkpoint
            if args.max_hours and epoch + 1 < stop and time.time() - t0 + 1.15 * slowest > args.max_hours * 3600:
                log(f"DỪNG trước epoch {epoch + 2}: epoch kế tiếp (~{fmt_time(slowest)}) có thể vượt "
                    f"--max-hours {args.max_hours}. Chạy lại đúng lệnh + --resume.")
                halt.fill_(1)
        if world > 1:
            dist.broadcast(halt, src=0)                                  # mọi tiến trình dừng cùng lúc
        if halt.item():
            break
    else:
        if main_proc and history and "eval" in history[-1]:
            print_table(history[-1]["eval"], f"eval cuối ({len(idx)} ảnh test)")
        log(f"[{fmt_time(time.time() - t0)}] xong {stop} epoch -> {args.save_dir}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
