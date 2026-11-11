#!/usr/bin/env python3
"""SFT + LoRA một VLM trên CE-130 train — 1 GPU (ZETA4.1 planner RePlan, ZETA8.x Qwen3-VL / InternVL / LLaVA; docs/EXPERIMENT_ZETA.md
mục 14.1, 16).

Phụ thuộc kiến trúc: `vlm/adapters.py` (`finetune.adapter`, mặc định `qwen`); dữ liệu / câu đích / gom batch: `vlm/sft.py`; LoRA tự
viết: `vlm/lora.py` (chỉ LLM; tháp ảnh + projector + embedding + lm_head đóng băng). Batch toàn cục `finetune.batch_size` (64 như
CE-Loc) = `micro_batch` × cộng dồn; loss = TB theo TOKEN câu trả lời trên cả batch (mỗi micro nhân tỉ lệ token của nó) và chỉ chiếu
`lm_head` ở token có nhãn (`answer_loss`). Trong mỗi batch các mẫu được sắp theo độ dài ước (token ảnh) trước khi chia micro
(`finetune.group_by_length`, mặc định bật) — đệm ít hơn, tập mẫu / loss của bước KHÔNG đổi. AdamW, warmup tuyến tính rồi cosine về 0,
clip grad; weight bf16, LoRA fp32, autocast bf16, gradient checkpointing. Mỗi epoch: `snapshots/epNNN.pth` (chỉ LoRA) + `last.pth`
(LoRA + optimizer, `--resume` ở biên epoch; config lệch thì dừng) + `history.json`; `--max-steps` dừng giữa chừng thì ghi
`snapshots/itNNNNN.pth` (pilot). `--merge epNNN --out DIR`: gộp LoRA vào weight -> thư mục HF đầy đủ (model + processor gốc) cho vLLM
(`vlm/run.py --model DIR`).

  cd /mnt/disk1/aiotlab/haitn/object-detection/vlm      # venv vLLM (torch + transformers), HF_HUB_OFFLINE=1
  python ../tools/run_on_free_gpu.py --reserve-gib 30 -- finetune.py --config config/server2/qwen3-vl-4b-sft.yaml \\
      --save-dir ../weights/add/zeta8_1 --resume
  python finetune.py --config config/server2/qwen3-vl-4b-sft.yaml --save-dir ../weights/add/zeta8_1 --merge ep003 \\
      --out ../weights/add/zeta8_1/merged_ep003
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
from ce_localization.engine.train_utils import (PthCheckpoints, config_mismatch, fmt_gpu_peak, gpu_peak,  # noqa: E402
                                                write_json_atomic)
from ce_localization.utils.config import load_config  # noqa: E402
from ce_localization.utils.log import fmt_time, log  # noqa: E402
from vlm.adapters import make_adapter  # noqa: E402
from vlm.lora import add_lora, load_lora, lora_parameters, lora_state_dict, merge_lora  # noqa: E402
from vlm.parse import qwen_resized_wh  # noqa: E402
from vlm.sft import SFTDataset, collate  # noqa: E402

# Khoá config quyết định dữ liệu / câu đích / model ⇒ resume phải trùng; `finetune` trừ các khoá chỉ đổi tốc độ / bộ nhớ.
RESUME_KEYS = ("model", "prompt", "instruction", "coord", "smart_resize", "k", "ask_k", "icl", "chat_template_kwargs", "finetune")
RESUME_FREE = {"finetune": ("micro_batch", "num_workers", "log_every", "group_by_length", "epochs")}   # epochs: kéo dài (cảnh báo)


def load_hf(cfg, dev):
    """-> (model bf16 trên `dev`, processor) qua adapter của config. Test thay hàm này bằng model giả."""
    return make_adapter(cfg).load(dev, cfg["finetune"].get("attn", "sdpa"))


def lora_model(cfg, dev, adapter):
    model, proc = load_hf(cfg, dev)
    lo = cfg["finetune"]["lora"]
    n = add_lora(model, lo["targets"], lo["r"], lo["alpha"], lo.get("dropout", 0.0), skip=tuple(lo.get("skip", adapter.skip)))
    return model, proc, n


def answer_loss(model, inp, adapter):
    """CE TB trên token câu trả lời — bằng hệt `model(**inp).loss` của HF nhưng chỉ chiếu `lm_head` ở vị trí có nhãn (~40–100 token /
    mẫu) thay vì cả chuỗi (1–2k token): logits ~152k từ × fp32 × cả chuỗi là phần lớn bộ nhớ đỉnh (ZETA4.1 2026-11-08: 44,75 -> 26 GiB
    ở micro 8)."""
    inp = dict(inp)
    tgt = inp.pop("labels")[:, 1:]
    keep = tgt != -100
    h = adapter.hidden(model, inp)[:, :-1][keep]
    return torch.nn.functional.cross_entropy(adapter.head(model)(h).float(), tgt[keep])


def lr_at(it, n_iter, warmup):
    """Hệ số lr của bước `it` (0-based): tăng tuyến tính trong `warmup` bước đầu, rồi cosine về 0 tới bước cuối."""
    if it < warmup:
        return (it + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * (it - warmup) / max(1, n_iter - warmup)))


def check_inputs(ds, proc, cfg, adapter):
    """Ảnh model thấy lúc train == lúc vLLM sinh (adapter); toạ độ `qwen_resized` (RePlan) đổi theo đúng cỡ ảnh đó."""
    s = ds[0]
    got = adapter.check(proc, s["image"])
    if cfg.get("coord") == "qwen_resized" and tuple(got) != tuple(qwen_resized_wh(s["image"].size, **(cfg.get("smart_resize") or {}))):
        raise RuntimeError(f"ảnh model thấy {got} khác smart_resize của config — toạ độ đích `qwen_resized` sẽ sai thang")


def sample_lengths(ds, proc, adapter, cache_path):
    """Độ dài ước của từng mẫu = số token ảnh (phần thay đổi giữa các mẫu; text gần như cố định) — đọc header ảnh một lần, cache JSON."""
    key = [len(ds), ds.items[0]["image"], ds.items[-1]["image"]]
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            c = json.load(f)
        if c.get("key") == key:
            return c["lengths"]
    lengths = [adapter.n_image_tokens(proc, ds.image_size(i)) for i in range(len(ds))]
    write_json_atomic(cache_path, {"key": key, "lengths": lengths})
    return lengths


def micro_order(order, lengths, batch_size, group):
    """Thứ tự mẫu cho DataLoader (batch_size = micro, không xáo): mỗi khối `batch_size` mẫu liên tiếp của `order` = MỘT bước tối ưu;
    `group` ⇒ trong khối sắp theo độ dài giảm dần trước khi chia micro (micro dài nhất chạy trước ⇒ lộ OOM sớm)."""
    out = []
    for i in range(0, len(order), batch_size):
        blk = [int(j) for j in order[i:i + batch_size]]
        out += sorted(blk, key=lambda j: -lengths[j]) if group else blk
    return out


AUX_SKIP = (".safetensors", ".bin", ".pt", ".pth", ".msgpack", ".h5", ".gitattributes", ".md")


def copy_aux_files(model, out):
    """Chép mọi file KHÔNG phải weight của checkpoint gốc (code remote *.py, tokenizer, chat template, preprocessor, generation config)
    mà `save_pretrained` chưa ghi ⇒ thư mục merge nạp được bằng vLLM / transformers y như gốc (InternVL / LLaVA-OV-2 dùng code remote).
    -> danh sách file đã chép."""
    import shutil
    if os.path.isdir(model):
        src = model
    else:
        from huggingface_hub import snapshot_download
        src = snapshot_download(model, ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.pth"])
    done = []
    for f in sorted(os.listdir(src)):
        p = os.path.join(src, f)
        if os.path.isfile(p) and not f.endswith(AUX_SKIP) and f != "model.safetensors.index.json" and not os.path.exists(os.path.join(out, f)):
            shutil.copy(p, os.path.join(out, f))
            done.append(f)
    return done


def merge(a, cfg):
    dev = torch.device(a.device or "cpu")
    adapter = make_adapter(cfg)
    model, proc, n = lora_model(cfg, dev, adapter)
    path = os.path.join(a.save_dir, "snapshots", f"{a.merge}.pth")
    load_lora(model, torch.load(path, map_location="cpu", weights_only=False)["lora"])
    k = merge_lora(model)
    model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
    if hasattr(proc, "save_pretrained"):              # processor code remote của LLaVA-OV-2 là lớp thường: file gốc do copy_aux_files chép
        proc.save_pretrained(a.out)
    extra = copy_aux_files(cfg["model"], a.out)
    log(f"[merge] {path}: gộp {k}/{n} lớp LoRA -> {a.out} (+ {len(extra)} file gốc: {extra[:8]})")


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
    adapter = make_adapter(cfg)
    items = train_items(TurnIndex(a.turn_index), a.samples_root, a.gt_root)[: a.limit]
    ds = SFTDataset(items, cfg, adapter.end)
    ipe = len(ds) // f["batch_size"]
    if ipe == 0:
        raise ValueError(f"{len(ds)} mẫu < batch {f['batch_size']}")
    n_iter, warmup = ipe * epochs, max(1, round(f.get("warmup", 0.03) * ipe * epochs))
    os.makedirs(a.save_dir, exist_ok=True)
    ck = PthCheckpoints(a.save_dir)
    st = ck.load_last() if a.resume and ck.has_last() else None
    if st is not None:
        bad = config_mismatch(st["config"], cfg, RESUME_KEYS, RESUME_FREE)
        if bad:
            sys.exit(f"KHÔNG resume được: config của {ck.last_path} khác {a.config} ở {bad} — dùng --save-dir khác")
        if st["config"]["finetune"]["epochs"] != epochs:
            log(f"[CẢNH BÁO] resume với epochs {st['config']['finetune']['epochs']} -> {epochs}: lịch lr cosine kéo theo số bước mới")
    log(f"[boot] {cfg['name']} = {cfg['model']} | {len(ds)} mẫu | {dev} | {fmt_time(time.time() - t0)}")
    model, proc, n_lora = lora_model(cfg, dev, adapter)
    check_inputs(ds, proc, cfg, adapter)
    group = bool(f.get("group_by_length", True))
    lengths = sample_lengths(ds, proc, adapter, os.path.join(a.save_dir, "lengths.json")) if group else None
    if f.get("grad_ckpt", True):
        adapter.grad_ckpt(model)
    params = lora_parameters(model)
    opt = torch.optim.AdamW(params, lr=f["lr"], weight_decay=f.get("weight_decay", 0.0), betas=(0.9, 0.999), fused=dev.type == "cuda")
    epoch, it, history = 0, 0, []
    if st is not None:
        load_lora(model, st["lora"])
        opt.load_state_dict(st["optimizer"])
        epoch, it, history = st["epoch"], st["iter"], st["history"]
    elif a.resume:
        log(f"[finetune] --resume nhưng chưa có {ck.last_path}: train từ đầu")
    amp = dev.type == "cuda"
    nw = f.get("num_workers", 4) if a.num_workers is None else a.num_workers
    log(f"[finetune] {a.config} | adapter {type(adapter).__name__} | LoRA {n_lora} lớp, {sum(p.numel() for p in params) / 1e6:.1f}M "
        f"tham số train | batch {f['batch_size']} = micro {micro} × {accum}{' (gom theo độ dài)' if group else ''} | {ipe} bước / epoch "
        f"× {epochs} = {n_iter} | lr {f['lr']:g} (warmup {warmup}) | {nw} worker" + (f" | resume epoch {epoch} (iter {it})" if it else "")
        + f" | {fmt_time(time.time() - t0)}")
    gpu_peak(dev)
    it0, t_run, stop = it, time.time(), False
    loader_kw = dict(num_workers=nw, pin_memory=dev.type == "cuda", collate_fn=partial(collate, processor=proc, adapter=adapter),
                     **({"persistent_workers": False, "prefetch_factor": 4} if nw else {}))
    while epoch < epochs and not stop:
        ds.set_epoch(epoch)
        order = np.random.default_rng(f.get("seed", 0) + epoch).permutation(len(ds))[: ipe * f["batch_size"]]
        loader = DataLoader(Subset(ds, micro_order(order, lengths, f["batch_size"], group)), batch_size=micro, shuffle=False, **loader_kw)
        model.train()
        zero = lambda: torch.zeros((), dtype=torch.float64, device=dev)  # noqa: E731
        ep_loss, ep_tok, win, t_win, buf = zero(), 0, [zero(), 0, 0], time.time(), []
        for mb in loader:
            buf.append(mb)
            if len(buf) < accum:
                continue
            total = sum(b["n_label"] for b in buf)
            loss_sum = zero()
            for g in opt.param_groups:
                g["lr"] = f["lr"] * lr_at(it, n_iter, warmup)
            for b in buf:
                inp = {k: v.to(dev, non_blocking=True) for k, v in b.items() if k != "n_label"}
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                    loss = answer_loss(model, inp, adapter)
                (loss * (b["n_label"] / total)).backward()
                loss_sum += loss.detach().double() * b["n_label"]                 # cộng trên thiết bị: không đồng bộ GPU mỗi micro
            torch.nn.utils.clip_grad_norm_(params, f.get("clip", 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            buf, it = [], it + 1
            ep_loss, ep_tok = ep_loss + loss_sum, ep_tok + total
            win = [win[0] + loss_sum, win[1] + total, win[2] + 1]
            if it % f.get("log_every", 10) == 0 or it == it0 + 1 or time.time() - t_win > 60:
                el = time.time() - t_run
                left = (min(n_iter, a.max_steps) if a.max_steps else n_iter) - it
                log(f"  [ep {epoch + 1}/{epochs} | bước {it}/{n_iter}] loss {win[0].item() / max(win[1], 1):.4f} | lr "
                    f"{opt.param_groups[0]['lr']:.2e} | {(time.time() - t_win) / win[2]:.2f} s/bước | {fmt_time(time.time() - t0)} | còn "
                    f"~{fmt_time(el / (it - it0) * left)}" + fmt_gpu_peak(gpu_peak(dev, reset=False)))
                win, t_win = [zero(), 0, 0], time.time()
            if a.max_steps and it >= a.max_steps:
                stop = True
                break
        if stop:
            p = ck.save_snapshot({"lora": lora_state_dict(model), "config": cfg, "epoch": epoch, "iter": it}, f"it{it:05d}.pth")
            log(f"[finetune] DỪNG (--max-steps) ở iter {it} -> {p}")
            break
        epoch += 1
        history.append({"epoch": epoch, "iter": it, "loss": ep_loss.item() / max(ep_tok, 1), "tokens": ep_tok, "sec": time.time() - t0})
        ck.save_snapshot({"lora": lora_state_dict(model), "config": cfg, "epoch": epoch, "iter": it}, f"ep{epoch:03d}.pth")
        ck.save({"lora": lora_state_dict(model), "optimizer": opt.state_dict(), "epoch": epoch, "iter": it, "history": history,
                 "config": cfg}, False)
        write_json_atomic(os.path.join(a.save_dir, "history.json"), history)
        log(f"[epoch {epoch}/{epochs}] loss TB {history[-1]['loss']:.4f} ({ep_tok} token) | {fmt_time(time.time() - t0)}"
            + fmt_gpu_peak(gpu_peak(dev)))
    log(f"[done] {it}/{n_iter} bước | {fmt_time(time.time() - t0)} -> {a.save_dir}")


if __name__ == "__main__":
    main()
