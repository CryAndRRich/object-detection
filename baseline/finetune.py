#!/usr/bin/env python3
"""Finetune một baseline ZETA trên CE-130 TRAIN (`samples/train`, 19.998 mẫu, như CE-Loc) từ weight công bố — 1 GPU hoặc `torchrun`.

Batch toàn cục `finetune.batch_size` (64 như CE-Loc) chia đều các GPU; mỗi GPU cộng dồn grad qua micro-batch `finetune.micro_batch`
(nút chỉnh bộ nhớ: không đổi kết quả, trừ thống kê BN của CSANet); grad TB qua các GPU MỘT lần mỗi bước (`dist.sync_grads`). AMP fp16
+ channels_last + GradScaler (`finetune.amp`, như CE-Loc trên T4), loss tính ở fp32. Như giao thức CE-Loc gốc: không val; `best.pth` =
epoch có chỉ số `Trainer.select` TB thấp nhất (cộng qua mọi GPU; BootPlace = loss train, CSANet = `g_rec`); `last.pth` mỗi epoch + khi dừng (`--max-hours`, đủ để `--resume` giữa epoch);
`snapshots/epNNN.pth` chỉ weight mỗi epoch + `ep000` (weight khởi tạo); `history.json`.

  cd object-detection/baseline
  python ../tools/run_on_free_gpu.py -- finetune.py --config config/zeta2.yaml --save-dir ../weights/add/zeta2
  torchrun --standalone --nproc_per_node=2 finetune.py --config config/zeta1.yaml --save-dir /kaggle/working --init <weight> \\
      --max-hours 9.5 [--resume]
"""

import argparse
import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from baseline.data import train_items  # noqa: E402
from baseline.dist import all_sum, sync_grads  # noqa: E402
from baseline.methods import batch_len, load_model, make_trainer, split_batch  # noqa: E402
from ce_localization.data.turns import TurnIndex  # noqa: E402
from ce_localization.engine.train_utils import EpochBatches, PthCheckpoints, fmt_gpu_peak, gpu_peak, setup_dist  # noqa: E402
from ce_localization.utils.config import load_config  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--init", default=None, help="weight công bố (mặc định finetune.init); resume vẫn cần để dựng kiến trúc")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--samples-root", default="../data/samples")
    ap.add_argument("--gt-root", default="../data/all_phase2_V2", help="thư mục <split>/<nhánh>/ground_truth.jpg")
    ap.add_argument("--micro-batch", type=int, default=None, help="ghi đè finetune.micro_batch (OOM thì hạ)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-hours", type=float, default=None, help="dừng sạch (ghi last.pth) sau N giờ")
    ap.add_argument("--max-steps", type=int, default=None, help="dừng sạch sau N bước tối ưu (test)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ N mẫu train đầu (test)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    t0 = time.time()
    cfg = load_config(a.config)
    f = cfg["finetune"]
    if a.micro_batch:
        f["micro_batch"] = a.micro_batch
    rank, world, dev = setup_dist(a.device)
    main_proc = rank == 0
    log = (lambda *x: print(*x, flush=True)) if main_proc else (lambda *x: None)
    if f["batch_size"] % world:
        sys.exit(f"batch_size {f['batch_size']} không chia hết cho {world} GPU")
    bpr = f["batch_size"] // world
    micro = min(f["micro_batch"], bpr)
    amp = bool(f.get("amp")) and dev.type == "cuda"
    torch.backends.cudnn.benchmark = dev.type == "cuda" and cfg["method"] == "csanet"       # CSANet: mọi ảnh 256²
    random.seed(f["seed"] * 1000 + rank)
    np.random.seed((f["seed"] * 1000 + rank) % 2 ** 32)
    torch.manual_seed(f["seed"] * 1000 + rank)

    items = train_items(TurnIndex(a.turn_index), a.samples_root, a.gt_root)[: a.limit]
    ck = PthCheckpoints(a.save_dir)
    state = ck.load_last() if a.resume and ck.has_last() else None
    init = a.init or f["init"]
    model = load_model(cfg, init, dev, drop_keys=f.get("drop_keys", ()), train=True)
    if amp:
        for m in model if isinstance(model, tuple) else (model,):
            m.to(memory_format=torch.channels_last)
    tr = make_trainer(cfg, dev, model)
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler") else torch.cuda.amp.GradScaler(enabled=amp))
    epochs = f["epochs"]
    epoch = step = it = 0
    ep_acc = torch.zeros(3, dtype=torch.float64, device=dev)          # [tổng chỉ số chọn best, số bước, tổng loss] epoch đang chạy
    history, best = [], float("inf")
    if state:
        free = ("micro_batch", "num_workers", "init", "log_every")              # khoá đổi được giữa các phiên (không đổi kết quả)
        key = lambda c: (c["method"], c["model"], c.get("loss"), {k: v for k, v in c["finetune"].items() if k not in free})  # noqa: E731
        if key(state["config"]) != key(cfg):
            sys.exit(f"KHÔNG resume được: config của {ck.last_path} khác {a.config} (ngoài {free}) — dùng --save-dir khác")
        tr.load_state_dict(state["model"])
        for o, s in zip(tr.optimizers, state["optimizers"]):
            o.load_state_dict(s)
        scaler.load_state_dict(state["scaler"])
        epoch, step, it, history, best = state["epoch"], state["step"], state["iter"], state["history"], state["best"]
        if main_proc:                                                  # tổng dở dang đã cộng qua mọi GPU, giữ ở rank 0
            ep_acc += torch.tensor(state["ep_acc"], dtype=torch.float64, device=dev)
    elif a.resume:
        log(f"[finetune] --resume nhưng chưa có {ck.last_path}: train từ đầu")
    ds, collate = tr.dataset(items)
    ipe = len(ds) // f["batch_size"]
    if ipe == 0:
        sys.exit(f"{len(ds)} mẫu < batch_size {f['batch_size']}")
    nw = f["num_workers"] if a.num_workers is None else a.num_workers
    sampler = EpochBatches(len(ds), bpr, rank, world, f["seed"])
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=nw, collate_fn=collate, pin_memory=dev.type == "cuda",
                        persistent_workers=nw > 0, generator=torch.Generator().manual_seed(f["seed"] * 100003 + rank))
    n_iter = ipe * epochs
    n_learn = sum(p.numel() for p in tr.params)
    log(f"[finetune] {cfg['method']} | {a.config} | init {init} | {len(ds)} mẫu | batch {f['batch_size']} = {world} GPU × {bpr} "
        f"(micro {micro}) | {ipe} bước / epoch × {epochs} = {n_iter} | {n_learn / 1e6:.1f}M tham số train | AMP {amp} | {dev} | "
        + (f"resume epoch {epoch} bước {step} (iter {it}) | " if state else "") + fmt_time(time.time() - t0))

    def save(is_best=False):
        acc = all_sum(ep_acc, world)
        if main_proc:
            ck.save({"model": tr.state_dict(), "optimizers": [o.state_dict() for o in tr.optimizers], "scaler": scaler.state_dict(),
                     "epoch": epoch, "step": step, "iter": it, "ep_acc": acc.tolist(), "history": history, "best": best, "config": cfg},
                    is_best)

    if main_proc and not state:
        ck.save_snapshot({**tr.weights(), "config": cfg, "epoch": 0}, "ep000.pth")
    win = {"loss": torch.zeros((), dtype=torch.float64, device=dev), "n": 0, "data": 0.0, "t": time.time()}
    it0, t_run, stop = it, time.time(), False
    gpu_peak(dev)
    while epoch < epochs and not stop:
        sampler.set(epoch, step)
        t_data = time.time()
        for batch in loader:
            win["data"] += time.time() - t_data
            tr.train()
            ctx = tr.context(batch, world)
            for o in tr.optimizers:
                o.zero_grad(set_to_none=True)
            n, loss, sel = batch_len(batch), 0.0, 0.0
            for chunk in split_batch(batch, micro):
                out = tr.backward(chunk, ctx, batch_len(chunk) / n, amp, scaler)
                loss, sel = loss + out["loss"], sel + out[tr.select]
            sync_grads(tr.params, world)
            for o in tr.optimizers:
                scaler.unscale_(o)
            if tr.clip:
                torch.nn.utils.clip_grad_norm_(tr.params, tr.clip)
            for o in tr.optimizers:
                scaler.step(o)
            scaler.update()
            loss, sel = loss.double(), sel.double()
            ok = torch.isfinite(loss) & torch.isfinite(sel)
            ep_acc += torch.stack([torch.where(ok, sel, 0.0), ok.double(), torch.where(ok, loss, 0.0)])
            win["loss"] += torch.where(ok, loss, 0.0)
            win["n"] += 1
            step, it = step + 1, it + 1
            if it % f["log_every"] == 0 or it == it0 + 1:
                el, dt = time.time() - t_run, time.time() - win["t"]
                log(f"  [ep {epoch + 1}/{epochs} | bước {it}/{n_iter}] loss {win['loss'].item() / win['n']:.4f} | {dt / win['n']:.2f} s/bước "
                    f"(chờ dữ liệu {win['data'] / win['n']:.2f}) | {fmt_time(time.time() - t0)} | còn ~{fmt_time(el / (it - it0) * (n_iter - it))}"
                    + fmt_gpu_peak(gpu_peak(dev, reset=False)) + (f" | AMP scale {scaler.get_scale():.0f}" if amp else ""))
                win = {"loss": torch.zeros((), dtype=torch.float64, device=dev), "n": 0, "data": 0.0, "t": time.time()}
            over = bool(a.max_hours and time.time() - t0 > a.max_hours * 3600) or bool(a.max_steps and it >= a.max_steps)
            flag = torch.tensor(float(over), device=dev)
            stop = bool(all_sum(flag, world).item())
            if stop or step == ipe:
                break
            t_data = time.time()
        if step == ipe:                                                # hết epoch
            acc = all_sum(ep_acc, world)
            mean, loss_mean = (acc[0] / acc[1].clamp(min=1)).item(), (acc[2] / acc[1].clamp(min=1)).item()
            is_best = mean < best
            best = min(best, mean)
            history.append({"epoch": epoch + 1, "iter": it, "loss": loss_mean, "select": mean, "select_by": tr.select,
                            "steps_ok": int(acc[1].item()), "sec": time.time() - t0})
            epoch, step = epoch + 1, 0
            ep_acc.zero_()
            save(is_best)
            if main_proc:
                ck.save_snapshot({**tr.weights(), "config": cfg, "epoch": epoch}, f"ep{epoch:03d}.pth")
                with open(os.path.join(a.save_dir, "history.json"), "w") as fh:
                    json.dump(history, fh, indent=1)
            log(f"[epoch {epoch}/{epochs}] loss TB {loss_mean:.4f} | {tr.select} TB {mean:.4f}{' (best)' if is_best else ''} | "
                f"{fmt_time(time.time() - t0)}"
                + fmt_gpu_peak(gpu_peak(dev)))
        elif stop:
            save()
            log(f"[finetune] DỪNG ở epoch {epoch + 1} bước {step}/{ipe} (iter {it}) -> {ck.last_path} — chạy lại với --resume")
    log(f"[done] {it}/{n_iter} bước | {fmt_time(time.time() - t0)} -> {a.save_dir}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
