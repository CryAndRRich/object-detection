#!/usr/bin/env python3
"""SFT + LoRA một VLM kiểu Qwen2.5-VL trên CE-130 train (ZETA4.1 = planner RePlan; docs/EXPERIMENT_ZETA.md mục 14.1) — 1 GPU.

Dữ liệu / câu đích / gom batch: `vlm/sft.py`; LoRA tự viết: `vlm/lora.py` (chỉ LLM, tháp ảnh + merger + embedding + lm_head đóng băng).
Batch toàn cục `finetune.batch_size` (64 như CE-Loc) = `micro_batch` × cộng dồn; loss = TB theo TOKEN câu trả lời trên cả batch (mỗi
micro nhân tỉ lệ token của nó). AdamW, warmup tuyến tính rồi cosine về 0, clip grad; weight bf16, LoRA fp32, autocast bf16, gradient
checkpointing. Mỗi epoch: `snapshots/epNNN.pth` (chỉ LoRA) + `last.pth` (LoRA + optimizer, `--resume` ở biên epoch) + `history.json`;
`--max-steps` dừng giữa chừng thì ghi `snapshots/itNNNNN.pth` (pilot). `--merge epNNN --out DIR`: gộp LoRA vào weight -> thư mục HF
đầy đủ (model + processor) cho vLLM (`vlm/run.py --model DIR`).

  cd /mnt/disk1/aiotlab/haitn/object-detection/vlm      # venv vLLM (torch + transformers Qwen2.5-VL), HF_HUB_OFFLINE=1
  python ../tools/run_on_free_gpu.py -- finetune.py --config config/server2/replan-7b-sft.yaml --save-dir ../weights/add/zeta4_1
  python finetune.py --config config/server2/replan-7b-sft.yaml --save-dir ../weights/add/zeta4_1 --merge ep001 \\
      --out ../weights/add/zeta4_1/merged_ep001
"""

import argparse
import json
import math
import os
import sys
import time
from functools import partial

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from baseline.data import train_items  # noqa: E402
from ce_localization.data.turns import TurnIndex  # noqa: E402
from ce_localization.engine.train_utils import PthCheckpoints, fmt_gpu_peak, gpu_peak  # noqa: E402
from ce_localization.utils.config import load_config  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402
from vlm.lora import add_lora, load_lora, lora_parameters, lora_state_dict, merge_lora  # noqa: E402
from vlm.parse import qwen_resized_wh  # noqa: E402
from vlm.sft import SFTDataset, collate  # noqa: E402


def log(*a):
    print(*a, flush=True)


def load_hf(cfg, dev):
    """-> (model bf16 trên `dev`, processor) từ `cfg["model"]` (id HF trong cache / thư mục). Test thay hàm này bằng model giả."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(cfg["model"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["model"], torch_dtype=torch.bfloat16,
                                                               attn_implementation=cfg["finetune"].get("attn", "sdpa"))
    return model.to(dev), proc


def lora_model(cfg, dev):
    model, proc = load_hf(cfg, dev)
    lo = cfg["finetune"]["lora"]
    n = add_lora(model, lo["targets"], lo["r"], lo["alpha"], lo.get("dropout", 0.0), skip=tuple(lo.get("skip", ("visual",))))
    return model, proc, n


def lr_at(it, n_iter, warmup):
    """Hệ số lr của bước `it` (0-based): tăng tuyến tính trong `warmup` bước đầu, rồi cosine về 0 tới bước cuối."""
    if it < warmup:
        return (it + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(1, n_iter - warmup)))


def check_resize(ds, proc, cfg):
    """Ảnh model thấy (lưới patch của processor × 14) phải trùng `qwen_resized_wh` dùng để đổi toạ độ đích — lệch là box sai thang."""
    s = ds[0]
    enc = proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[s["image"]], return_tensors="pt")
    t, h, w = (int(v) for v in enc["image_grid_thw"][0])
    want = qwen_resized_wh(s["image"].size, **(cfg.get("smart_resize") or {}))
    if (w * 14, h * 14) != tuple(want):
        raise RuntimeError(f"processor resize ảnh {s['image'].size} -> {(w * 14, h * 14)} khác qwen_resized_wh {want} — sai smart_resize")


def merge(a, cfg):
    dev = torch.device(a.device or "cpu")
    model, proc, n = lora_model(cfg, dev)
    path = os.path.join(a.save_dir, "snapshots", f"{a.merge}.pth")
    load_lora(model, torch.load(path, map_location="cpu", weights_only=False)["lora"])
    k = merge_lora(model)
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    proc.save_pretrained(a.out)
    log(f"[merge] {path}: gộp {k}/{n} lớp LoRA -> {a.out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--resume", action="store_true", help="tiếp từ last.pth (biên epoch)")
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--samples-root", default="../data/samples")
    ap.add_argument("--gt-root", default="../data/all_phase2_V2")
    ap.add_argument("--micro-batch", type=int, default=None, help="ghi đè finetune.micro_batch (OOM thì hạ; không đổi kết quả)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="dừng sau N bước tối ưu, ghi snapshots/itNNNNN.pth (pilot)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ N mẫu train đầu (pilot / test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--merge", default=None, help="tên snapshot (ep001 | it00020) -> gộp LoRA, ghi --out")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.merge:
        if not a.out:
            ap.error("--merge cần --out")
        return merge(a, cfg)
    t0 = time.time()
    f = cfg["finetune"]
    micro = a.micro_batch or f["micro_batch"]
    if f["batch_size"] % micro:
        raise ValueError(f"batch_size {f['batch_size']} không chia hết cho micro_batch {micro}")
    accum, epochs = f["batch_size"] // micro, f["epochs"]
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(f.get("seed", 0))
    items = train_items(TurnIndex(a.turn_index), a.samples_root, a.gt_root)[: a.limit]
    ds = SFTDataset(items, cfg)
    ipe = len(ds) // f["batch_size"]
    if ipe == 0:
        raise ValueError(f"{len(ds)} mẫu < batch {f['batch_size']}")
    n_iter, warmup = ipe * epochs, max(1, round(f.get("warmup", 0.03) * ipe * epochs))
    log(f"[boot] {cfg['name']} = {cfg['model']} | {len(ds)} mẫu | {dev} | {fmt_time(time.time() - t0)}")
    model, proc, n_lora = lora_model(cfg, dev)
    check_resize(ds, proc, cfg)
    if f.get("grad_ckpt", True) and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    params = lora_parameters(model)
    opt = torch.optim.AdamW(params, lr=f["lr"], weight_decay=f.get("weight_decay", 0.0), betas=(0.9, 0.999), fused=dev.type == "cuda")
    ck = PthCheckpoints(a.save_dir)
    epoch, it, history = 0, 0, []
    if a.resume and ck.has_last():
        st = ck.load_last()
        load_lora(model, st["lora"])
        opt.load_state_dict(st["optimizer"])
        epoch, it, history = st["epoch"], st["iter"], st["history"]
    amp = dev.type == "cuda"
    nw = f.get("num_workers", 4) if a.num_workers is None else a.num_workers
    log(f"[finetune] {a.config} | LoRA {n_lora} lớp, {sum(p.numel() for p in params) / 1e6:.1f}M tham số train | batch {f['batch_size']} = "
        f"micro {micro} × {accum} | {ipe} bước / epoch × {epochs} = {n_iter} | lr {f['lr']:g} (warmup {warmup}) | {nw} worker"
        + (f" | resume epoch {epoch} (iter {it})" if it else "") + f" | {fmt_time(time.time() - t0)}")
    gpu_peak(dev)
    it0, t_run, stop = it, time.time(), False
    while epoch < epochs and not stop:
        ds.set_epoch(epoch)
        order = np.random.default_rng(f.get("seed", 0) + epoch).permutation(len(ds))[: ipe * f["batch_size"]]
        loader = DataLoader(Subset(ds, order.tolist()), batch_size=micro, shuffle=False, num_workers=nw,
                            collate_fn=partial(collate, processor=proc), persistent_workers=False)
        model.train()
        ep_loss, ep_tok, win, t_win, buf = 0.0, 0, [0.0, 0, 0], time.time(), []
        for mb in loader:
            buf.append(mb)
            if len(buf) < accum:
                continue
            total = sum(b["n_label"] for b in buf)
            loss_sum = 0.0
            for g in opt.param_groups:
                g["lr"] = f["lr"] * lr_at(it, n_iter, warmup)
            for b in buf:
                inp = {k: v.to(dev, non_blocking=True) for k, v in b.items() if k != "n_label"}
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                    out = model(**inp)
                (out.loss.float() * (b["n_label"] / total)).backward()
                loss_sum += out.loss.item() * b["n_label"]
            torch.nn.utils.clip_grad_norm_(params, f.get("clip", 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            buf, it = [], it + 1
            ep_loss, ep_tok = ep_loss + loss_sum, ep_tok + total
            win = [win[0] + loss_sum, win[1] + total, win[2] + 1]
            if it % f.get("log_every", 10) == 0 or it == it0 + 1 or time.time() - t_win > 60:
                el = time.time() - t_run
                log(f"  [ep {epoch + 1}/{epochs} | bước {it}/{n_iter}] loss {win[0] / max(win[1], 1):.4f} | lr {opt.param_groups[0]['lr']:.2e} | "
                    f"{(time.time() - t_win) / win[2]:.2f} s/bước | {fmt_time(time.time() - t0)} | còn ~{fmt_time(el / (it - it0) * (n_iter - it))}"
                    + fmt_gpu_peak(gpu_peak(dev, reset=False)))
                win, t_win = [0.0, 0, 0], time.time()
            if a.max_steps and it >= a.max_steps:
                stop = True
                break
        if stop:
            p = ck.save_snapshot({"lora": lora_state_dict(model), "config": cfg, "epoch": epoch, "iter": it}, f"it{it:05d}.pth")
            log(f"[finetune] DỪNG (--max-steps) ở iter {it} -> {p}")
            break
        epoch += 1
        history.append({"epoch": epoch, "iter": it, "loss": ep_loss / max(ep_tok, 1), "tokens": ep_tok, "sec": time.time() - t0})
        ck.save_snapshot({"lora": lora_state_dict(model), "config": cfg, "epoch": epoch, "iter": it}, f"ep{epoch:03d}.pth")
        ck.save({"lora": lora_state_dict(model), "optimizer": opt.state_dict(), "epoch": epoch, "iter": it, "history": history,
                 "config": cfg}, False)
        with open(os.path.join(a.save_dir, "history.json"), "w") as fh:
            json.dump(history, fh, indent=1)
        log(f"[epoch {epoch}/{epochs}] loss TB {history[-1]['loss']:.4f} ({ep_tok} token) | {fmt_time(time.time() - t0)}"
            + fmt_gpu_peak(gpu_peak(dev)))
    log(f"[done] {it}/{n_iter} bước | {fmt_time(time.time() - t0)} -> {a.save_dir}")


if __name__ == "__main__":
    main()
