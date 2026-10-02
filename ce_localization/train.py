#!/usr/bin/env python3
"""Train CE-Loc detector: R-50 + FPN, RoI -> token, stage decoder kiểu Diffusion Policy, khuếch tán /
loss kiểu DiffusionDet. MỘT thí nghiệm = MỘT config (`config/<nhóm>/<tên>.yaml`): ALPHA0/1/2 khác
`model.memory`, ALPHA3.1/3.2 thêm density (`data.density`), BETA đổi đích train (`data.targets`) —
docs/EXPERIMENT_ALPHA.md, docs/EXPERIMENT_BETA.md.

Train theo ITERATION (12k iter, batch TOÀN CỤC 2), lịch lr WarmupMultiStep, clip grad toàn mô hình.
Mỗi `eval_every` iter: eval THẬT trên val (DDIM từ nhiễu thuần, N=200, 1 bước) -> `oracle_recall`
chọn `best.pth`. Mỗi `ckpt_every` iter: `last.pth` (ghi nguyên tử), đủ để `--resume` tái lập:
thứ tự dữ liệu seed theo (seed, epoch), t / nhiễu seed theo (seed, iter, rank).

Hai cách chạy, CÙNG phép toán:
  1 GPU (server):  python ../tools/run_on_free_gpu.py -- train.py --config config/alpha/alpha0.yaml \\
                       --save-dir ../weights/detection/alpha0
  2 GPU (Kaggle):  torchrun --standalone --nproc_per_node=2 train.py --config ... --save-dir ...
Chạy > 5 phút => nohup nền, xem docs/EXPERIMENT_ALPHA.md mục 9.

Cửa chặn:
  G3 overfit: --limit 16 --max-iter 1500 --eval-split train --eval-every 500 [--lr 1e-4]
  G4 bench  : --bench 50  (không ghi gì, in s/iter tách đọc dữ liệu / tính toán + bộ nhớ đỉnh)

BETA (`data.targets: point`, docs/EXPERIMENT_BETA.md): đích train = box giả từ điểm density
(`data.points`, `data.pseudo_size`), loss chế độ `point`; eval định kỳ vẫn chấm trên box GT, chọn
`best.pth` theo `eval.select_metric` (BETA: `oracle_recall_pt`).

GAMMA (`task: add`, docs/EXPERIMENT_GAMMA.md): bài ADD — mẫu = (nhánh, lượt) qua `data.turn_index`, đích = lỗ
mới nhất, `noise_per_image` bộ (t, ε) mỗi ảnh. `model.arch: box_policy` (GAMMA0: R-50 -> SpatialSoftmax C5 -> FiLM ->
U-Net 1D, ε-MSE, eval DDPM) | `box_refiner` (GAMMA1: 6 tầng RoI + cross-attn [t ; text ; vis], L1 + GIoU ở mọi tầng,
eval DDIM `eval.sampling_steps` bước). Eval định kỳ = `eval.n_samples` mẫu / ảnh trên `eval.limit` mẫu val cố định,
chọn `best.pth` theo `eval.select_metric` (`mean_iou_any`).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.data.dataset import CE130Dataset, collate, to_device  # noqa: E402
from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.points import PointTable  # noqa: E402
from ce_localization.data.turns import CE130AddDataset, TurnIndex, collate_add, to_device_add  # noqa: E402
from ce_localization.engine.add_eval import add_metrics, predict_add  # noqa: E402
from ce_localization.engine import nan_debug  # noqa: E402
from ce_localization.engine.criterion import Criterion, build_targets  # noqa: E402
from ce_localization.engine.diffusion import prepare_train_boxes  # noqa: E402
from ce_localization.engine.evaluate import predict, score  # noqa: E402
from ce_localization.engine.train_utils import (PthCheckpoints, epoch_batches,  # noqa: E402
                                               lr_factor, noise_seed, setup_dist)
from ce_localization.models.backbone import density_weight_ratio  # noqa: E402
from ce_localization.models.box_policy import boxes_to_unit, norm_whwh  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.models.text import TextTable, encode_class_names  # noqa: E402
from ce_localization.utils.grad_monitor import GradMonitor  # noqa: E402
from ce_localization.utils.log import fmt_time, print_banner, run_env  # noqa: E402

# Nhánh config quyết định kiến trúc / bài toán: khác checkpoint thì KHÔNG resume. `training` /
# `eval` được phép khác (chỉ cảnh báo); `data.num_workers` / đường dẫn dữ liệu không ảnh hưởng phép
# toán (Kaggle gắn dataset ở đường dẫn khác nhau giữa các phiên).
MUST_MATCH = ("model", "diffusion", "matcher", "data", "loss")
DATA_FREE = ("num_workers", "root", "density_root", "density_index", "points", "samples_root", "turn_index")
TASKS = ("detect", "add")
# nhãn số hạng loss trên dòng log (giữ đúng nhãn cũ của ALPHA: ce / l1 / giou / iou)
LOG_LABEL = {"loss_ce": "ce", "loss_bbox": "l1", "loss_giou": "giou", "iou_matched": "iou",
             "loss_center": "center", "loss_size": "size", "center_px": "center_px"}


def config_diff(saved, cfg):
    """-> (lỗi, cảnh báo): nhánh config khác nhau giữa checkpoint và lần chạy này."""
    strip = lambda b, d: {k: v for k, v in (d or {}).items()  # noqa: E731
                          if not (b == "data" and k in DATA_FREE)}
    errors = [k for k in MUST_MATCH if strip(k, saved.get(k)) != strip(k, cfg.get(k))]
    if saved.get("task", "detect") != cfg.get("task", "detect"):
        errors.append("task")
    warns = [k for k in ("training", "eval") if saved.get(k) != cfg.get(k)]
    return errors, warns


def build_text_table(names, cfg, dev, state_dict=None):
    """Tách ra hàm riêng để test thay bằng embedding giả (không tải CLIP). `state_dict`: CLIP text lưu trong checkpoint CE-Loc
    gốc của bài (eval.py)."""
    return TextTable(encode_class_names(names, cfg["model"]["clip_text"], device=str(dev), state_dict=state_dict))


def density_setup(cfg):
    """-> DensityIndex hoặc None. Kiểm `data.density` khớp `model.in_channels` (4 <=> có density)."""
    mode, ch = cfg["data"].get("density"), cfg["model"].get("in_channels", 3)
    if (mode is not None) != (ch == 4):
        sys.exit(f"config mâu thuẫn: data.density={mode} nhưng model.in_channels={ch} (density <=> 4 kênh)")
    if mode is None:
        return None
    try:
        return DensityIndex(cfg["data"]["density_index"], cfg["data"]["density_root"])
    except FileNotFoundError as e:
        sys.exit(str(e))


def add_density_setup(cfg):
    """GAMMA: `data.density` (None | sample) phải khớp `model.in_channels` (4 <=> có density)."""
    mode, ch = cfg["data"].get("density"), cfg["model"].get("in_channels", 3)
    if (mode is not None) != (ch == 4):
        sys.exit(f"config mâu thuẫn: data.density={mode} nhưng model.in_channels={ch} (density <=> 4 kênh)")
    return mode


def add_datasets(cfg, eval_split, limit=None, eval_limit=None):
    """GAMMA: -> (ds_tr, ds_ev, TurnIndex). Eval định kỳ trên `eval.limit` mẫu CỐ ĐỊNH của split (hoán vị seed
    12345, như eval định kỳ của CE-Loc gốc) — chỉ để chọn checkpoint; báo cáo bằng eval.py trên cả split."""
    d = cfg["data"]
    try:
        index = TurnIndex(d["turn_index"])
    except FileNotFoundError as e:
        sys.exit(str(e))
    dens = add_density_setup(cfg)
    mk = lambda split: CE130AddDataset(index, d["root"], d["samples_root"], split, d["image_size"],  # noqa: E731
                                       density=dens, style=d.get("input_style", "ours"))
    ds_tr = mk("train")
    if limit:
        ds_tr.keys = ds_tr.keys[:limit]
    ds_ev = mk("train" if eval_split == "train" else eval_split)
    if eval_split == "train":
        ds_ev.keys = ds_tr.keys[:]                      # G3: eval trên CHÍNH các mẫu đã train
    elif cfg["eval"].get("limit"):
        perm = np.random.default_rng(12345).permutation(len(ds_ev.keys))[: cfg["eval"]["limit"]]
        ds_ev.keys = [ds_ev.keys[i] for i in sorted(perm.tolist())]
    if eval_limit:
        ds_ev.keys = ds_ev.keys[:eval_limit]
    return ds_tr, ds_ev, index


def targets_setup(cfg):
    """-> (targets 'box'|'point', PointTable | None, pseudo_size dict | None) cho tập TRAIN."""
    d = cfg["data"]
    targets = d.get("targets", "box")
    if targets == "box":
        return targets, None, None
    try:
        return targets, PointTable(d["points"]), d["pseudo_size"]
    except FileNotFoundError as e:
        sys.exit(str(e))


def make_eval_loader(ds, cfg, num_workers):
    if cfg.get("task", "detect") == "add":
        return DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=num_workers,
                          collate_fn=collate_add)
    bs = cfg["eval"]["batch_size"] if cfg["eval"]["sampling_steps"] == 1 else 1
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate)


def run_val(model, loader, text_table, cfg, log):
    t0 = time.time()
    ev = cfg["eval"]
    if cfg.get("task", "detect") == "add":
        rec = predict_add(model, loader, text_table, ev["n_samples"], seed=0,
                          log_every=max(len(loader) // 4, 1), log=log, steps=ev.get("sampling_steps"))
        res = add_metrics(rec)
        res["eval_sec"] = time.time() - t0
        if hasattr(model, "pop_attn"):                  # GAMMA1: attention của query lên [t ; text ; vis]
            res["attn"] = model.pop_attn()
        return res
    rec, stage = predict(model, loader, text_table, cfg["diffusion"]["num_proposals"],
                         steps=ev["sampling_steps"], top_k=ev["top_k"], nms_thr=ev["nms_thr"],
                         seed=0, log_every=max(len(loader) // 4, 1), log=log)
    res = score(rec, stage, ev["top_k"], ev["nms_thr"], oracle=True)
    res["eval_sec"] = time.time() - t0
    return res


def bench(model, net, crit, loader_fn, text_table, cfg, opt, dev, n, log, todev=to_device):
    """G4: đo riêng đọc dữ liệu và tính toán (forward + backward + step) trên n iter."""
    model.train()
    batches = loader_fn(0, 0)
    it = iter(batches)
    t_data, t_comp = 0.0, 0.0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device=dev.type)
    for k in range(n + 3):
        t = time.time()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader_fn(1, 0))
            batch = next(it)
        batch = todev(batch, dev)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        td = time.time() - t
        t = time.time()
        gen.manual_seed(k)
        loss = _step_loss(net, crit, batch, text_table, cfg, model, gen, dev)[0]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if k >= 3:                                            # bỏ 3 iter khởi động
            t_data += td
            t_comp += time.time() - t
    mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == "cuda" else 0.0
    per = (t_data + t_comp) / n
    log(f"[bench] {n} iter: đọc dữ liệu {t_data / n:.3f} s/iter | tính toán {t_comp / n:.3f} s/iter | "
        f"tổng {per:.3f} s/iter | bộ nhớ đỉnh {mem:.2f} GB")
    log(f"[bench] ETA thô cho {cfg['training']['max_iter']} iter (chưa tính eval): "
        f"{fmt_time(per * cfg['training']['max_iter'])} (đọc dữ liệu chạy song song với worker nên "
        f"thời gian thật gần max(đọc, tính) hơn là tổng)")


def _step_loss(net, crit, batch, text_table, cfg, model, gen, dev):
    if crit is None:                                    # GAMMA (bài add)
        k = cfg["diffusion"]["noise_per_image"]
        text = text_table(batch["text"], dev)
        if cfg["model"].get("arch") == "box_refiner":   # GAMMA1: L1 + GIoU ở mọi tầng, đích = lỗ mới nhất
            return net(batch["images"], text, batch["valid_hw"], batch["target"], batch["whwh"], k=k, generator=gen)
        x0 = boxes_to_unit(batch["target"], norm_whwh(model, batch["whwh"], batch["images"].shape[-1]))  # GAMMA0: ε-MSE
        loss = net(batch["images"], text, batch["valid_hw"], x0, k=k, generator=gen)
        return loss, {"loss": loss.detach()}
    boxes, t = prepare_train_boxes(batch["boxes"], batch["whwh"], cfg["diffusion"]["num_proposals"],
                                   model.alphas_cumprod, model.snr_scale, gen)
    text = text_table(batch["text"], dev)
    logits, pred = net(batch["images"], text, batch["valid_hw"], boxes, t)
    return crit(logits, pred, build_targets(batch["boxes"], batch["whwh"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-dir", required=True, help="vd ../weights/detection/alpha0 (weights/ đã .gitignore)")
    ap.add_argument("--resume", action="store_true",
                    help="train tiếp từ <save-dir>/last.pth. Không có cờ này mà last.pth đã có thì DỪNG")
    ap.add_argument("--max-iter", type=int, default=None, help="ghi đè training.max_iter (vd. G3)")
    ap.add_argument("--lr", type=float, default=None,
                    help="ghi đè training.lr (vd. G3 kiểm 'code sai hay lr chậm'); ghi vào config của checkpoint")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N ảnh train đầu (G3 overfit)")
    ap.add_argument("--eval-split", default=None, help="mặc định eval.split; 'train' = eval trên chính ảnh train")
    ap.add_argument("--eval-limit", type=int, default=None, help="chỉ eval N ảnh đầu (chạy thử)")
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--data-root", default=None, help="ghi đè data.root (vd. Kaggle)")
    ap.add_argument("--density-root", default=None, help="ALPHA3: ghi đè data.density_root (thư mục samples/)")
    ap.add_argument("--density-index", default=None, help="ALPHA3: ghi đè data.density_index (.json)")
    ap.add_argument("--samples-root", default=None, help="GAMMA: ghi đè data.samples_root (thư mục samples/)")
    ap.add_argument("--turn-index", default=None, help="GAMMA: ghi đè data.turn_index (.json, cửa G0)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="dừng sạch (ghi last.pth) nếu đoạn kế tiếp có thể vượt N giờ; 0 = tắt")
    ap.add_argument("--bench", type=int, default=0, help="G4: đo N iter rồi thoát, không ghi gì")
    ap.add_argument("--nan-debug", action="store_true",
                    help="in loss/grad MỖI bước, kiểm weight sau mỗi step; lần NaN đầu tiên: báo đầu vào / "
                         "tham số / buffer / module đầu tiên ra NaN rồi DỪNG (1 tiến trình)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    tr = cfg["training"]
    if a.max_iter:
        tr["max_iter"] = a.max_iter
        tr["steps"] = [s for s in tr["steps"] if s < a.max_iter]
    if a.lr:
        tr["lr"] = a.lr
    if a.eval_every:
        tr["eval_every"] = a.eval_every
    if a.ckpt_every:
        tr["ckpt_every"] = a.ckpt_every
    if a.limit:
        cfg["data"]["limit"] = a.limit
    if a.eval_split:
        cfg["eval"]["split"] = a.eval_split
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    if a.density_root:
        cfg["data"]["density_root"] = a.density_root
    if a.density_index:
        cfg["data"]["density_index"] = a.density_index
    if a.samples_root:
        cfg["data"]["samples_root"] = a.samples_root
    if a.turn_index:
        cfg["data"]["turn_index"] = a.turn_index
    nw = cfg["data"]["num_workers"] if a.num_workers is None else a.num_workers
    task = cfg.get("task", "detect")
    if task not in TASKS:
        sys.exit(f"task {task!r} không thuộc {TASKS}")
    todev, collate_fn = (to_device_add, collate_add) if task == "add" else (to_device, collate)

    t_boot = time.time()
    rank, world, dev = setup_dist(a.device)
    main_proc = rank == 0
    log = (lambda *s: print(*s, flush=True)) if main_proc else (lambda *s: None)
    if tr["batch_size"] % world:
        sys.exit(f"batch_size {tr['batch_size']} không chia hết cho {world} GPU")
    if (a.bench or a.nan_debug) and world > 1:
        sys.exit("--bench / --nan-debug chỉ chạy 1 tiến trình")
    bpr = tr["batch_size"] // world

    ckm = PthCheckpoints(a.save_dir) if not a.bench else None
    if ckm is not None and ckm.has_last() and not a.resume:
        sys.exit(f"\n{ckm.last_path} ĐÃ TỒN TẠI.\n  train tiếp: thêm --resume\n  train mới : đổi --save-dir\n")
    if a.resume and (ckm is None or not ckm.has_last()):
        sys.exit(f"\n--resume nhưng không thấy {os.path.join(a.save_dir, 'last.pth')}. Sai --save-dir?\n")

    torch.manual_seed(tr["seed"] + rank)
    np.random.seed(tr["seed"] + rank)
    env = run_env(cfg, dev)
    env["world_size"] = world
    if main_proc:
        print_banner(f"TRAIN — {cfg['experiment']}", env)

    # ------------------------------------------------------------------ dữ liệu
    root, size = cfg["data"]["root"], cfg["data"]["image_size"]
    ev_split = cfg["eval"]["split"]
    if task == "add":                                  # GAMMA: (nhánh, lượt), đích = lỗ mới nhất
        ds_tr, ds_ev, _ = add_datasets(cfg, ev_split, cfg["data"].get("limit"), a.eval_limit)
        d_tr = d_ev = cfg["data"].get("density")
        targets, ptable, pseudo = "lỗ mới nhất", None, None
    else:
        dindex = density_setup(cfg)
        d_tr = cfg["data"].get("density")
        # eval định kỳ (chọn best.pth) luôn với density ĐẦY ĐỦ, kể cả ALPHA3.2 (train mix)
        d_ev = cfg["eval"].get("density", "full") if d_tr else None
        # BETA: chỉ tập TRAIN đổi đích sang box giả; tập eval luôn chế độ box (chấm trên box GT)
        targets, ptable, pseudo = targets_setup(cfg)
        ds_tr = CE130Dataset(root, "train", size, density=d_tr, density_index=dindex, seed=tr["seed"],
                           targets=targets, points=ptable, pseudo=pseudo)
        if cfg["data"].get("limit"):
            ds_tr.items = ds_tr.items[: cfg["data"]["limit"]]
        if ev_split == "train":
            ds_ev = CE130Dataset(root, "train", size, density=d_ev, density_index=dindex)
            ds_ev.items = ds_tr.items[:]                   # G3: eval trên CHÍNH các ảnh đã train
        else:
            ds_ev = CE130Dataset(root, ev_split, size, density=d_ev, density_index=dindex)
        if a.eval_limit:
            ds_ev.items = ds_ev.items[: a.eval_limit]
    ipe = len(ds_tr) // tr["batch_size"]
    if ipe == 0:
        sys.exit(f"train chỉ có {len(ds_tr)} mẫu < batch_size {tr['batch_size']}")
    log(f"[data] task {task} | train {len(ds_tr)} mẫu ({ipe} iter/epoch, batch {bpr}/GPU × {world}) | "
        f"eval {ev_split} {len(ds_ev)} mẫu | density train {d_tr} / eval {d_ev} | đích train {targets}"
        + ("" if ptable is None else f" (điểm {cfg['data']['points']}, tham số tách đỉnh {ptable.params}, "
                                     f"cỡ giả {pseudo})")
        + f" | {fmt_time(time.time() - t_boot)}")

    def loader_fn(epoch, start_batch):
        ds_tr.epoch = epoch                            # density `mix`: RNG theo (seed, epoch, ảnh)
        bl = epoch_batches(len(ds_tr), bpr, rank, world, tr["seed"], epoch)[start_batch:]
        # generator riêng: tạo iterator DataLoader rút base_seed của worker từ đây, KHÔNG từ RNG toàn cục — nếu không,
        # --resume giữa epoch (tạo iterator mới) làm lệch RNG toàn cục => mặt nạ dropout khác bản train liền (GAMMA1)
        g = torch.Generator().manual_seed(tr["seed"] * 100003 + epoch * 101 + rank)
        return DataLoader(ds_tr, batch_sampler=bl, num_workers=nw, collate_fn=collate_fn,
                          pin_memory=dev.type == "cuda", generator=g)

    # ------------------------------------------------------------------ text
    t = time.time()
    names = sorted(set(ds_tr.classes()) | set(ds_ev.classes()))
    if main_proc:
        text_table = build_text_table(names, cfg, dev)          # rank 0 tải CLIP trước
    if world > 1:
        dist.barrier()
    if not main_proc:
        text_table = build_text_table(names, cfg, dev)
    log(f"[text] {len(names)} lớp -> CLIP text {cfg['model']['clip_text']} | {fmt_time(time.time() - t)}")

    # ------------------------------------------------------------------ model
    t = time.time()
    model = build_model(cfg).to(dev)
    if world > 1 and any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)   # GAMMA (BN train như bài): thống kê trên cả batch
    # find_unused_parameters=True BẮT BUỘC: conv 3x3 đầu ra của một tầng FPN chỉ có gradient khi có
    # RoI rơi vào tầng đó. Ở canvas 512, P5 cần sqrt(diện tích) >= 448 px nên ở ALPHA0 (P5 chỉ đi
    # qua RoI) gần như không bao giờ được dùng; tầng khác cũng có thể trống ở một iteration.
    net = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dev.index] if dev.type == "cuda" else None,
        find_unused_parameters=True) if world > 1 else model
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] arch={cfg['model'].get('arch', 'detector')} | memory={cfg['model'].get('memory')} | "
        f"in_channels {model.backbone.in_channels} | {n_learn / 1e6:.2f}M tham số train | "
        f"N={cfg['diffusion'].get('num_proposals')} | {fmt_time(time.time() - t)}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, tr))
    crit = None if task == "add" else Criterion(cfg["loss"], cfg["matcher"], mode=targets)

    if a.bench:
        return bench(model, net, crit, loader_fn, text_table, cfg, opt, dev, a.bench, log, todev)

    start, best, history = 0, None, []
    if a.resume:
        st = ckm.load_last(map_location="cpu")
        bad, warns = config_diff(st["config"], cfg)
        if bad:
            sys.exit(f"\nKHÔNG resume được: config khác checkpoint ở {bad}. Dùng --save-dir khác.\n")
        if warns:
            log(f"[resume] ⚠️  nhánh {warns} khác lần trước — vẫn tiếp tục")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        start, best, history = st["iter"], st["best"], st["history"]
        if main_proc:                                   # RNG toàn cục (dropout) của rank 0
            torch.set_rng_state(st["rng"]["torch"])
        log(f"[resume] {ckm.last_path}: đã xong {start} iter, best {best}")
    if start >= tr["max_iter"]:
        sys.exit(f"\nĐã train đủ {tr['max_iter']} iter; muốn thêm thì tăng --max-iter.\n")

    ev_loader = make_eval_loader(ds_ev, cfg, nw) if main_proc else None
    gmon = GradMonitor(model, every=tr["log_every"])
    gen = torch.Generator(device=dev.type)
    sm = cfg["eval"]["select_metric"]

    def save(it_next, is_best):
        if not main_proc:
            return
        rng = torch.get_rng_state()
        ckm.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                  "scheduler": sched.state_dict(), "iter": it_next, "best": best,
                  "history": history, "config": cfg, "env": env, "rng": {"torch": rng}}, is_best)
        with open(os.path.join(a.save_dir, "history.json"), "w") as f:
            json.dump({"config": cfg, "environment": env, "best": best, "history": history}, f,
                      indent=1, ensure_ascii=False, default=float)

    log(f"[boot ] khởi động xong {fmt_time(time.time() - t_boot)} — bắt đầu iter {start}/{tr['max_iter']}")

    # ------------------------------------------------------------------ vòng train
    t_train, t_last, it = time.time(), time.time(), start
    win = {"loss": 0.0, "n": 0, "gn": [], "data": 0.0, "skip": 0}
    slowest_chunk, chunk_t0 = 0.0, time.time()
    model.train()
    done = False
    while not done:
        epoch, offset = divmod(it, ipe)
        t_data = time.time()
        for batch in loader_fn(epoch, offset):
            win["data"] += time.time() - t_data
            batch = todev(batch, dev)
            gen.manual_seed(noise_seed(tr["seed"], it, rank))
            loss, st = _step_loss(net, crit, batch, text_table, cfg, model, gen, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gmon.maybe_record(it)
            # grad_clip null = KHÔNG clip (CE-Loc gốc), vẫn đo norm để log
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"] or float("inf"))
            finite = bool(torch.isfinite(gn)) and bool(torch.isfinite(loss))
            if a.nan_debug:
                log(f"  [nan-debug] it {it} | loss {float(loss):.4f} | grad trước clip {float(gn):.4g} | "
                    f"lr {sched.get_last_lr()[0]:.3e} | density {batch.get('density_kind')} | ảnh {batch['image_id']}")
                if not finite and task == "add":
                    sys.exit(f"[nan-debug] dừng ở NaN đầu tiên, it {it} (bài add: chưa có báo cáo module)")
                if not finite:
                    gen.manual_seed(noise_seed(tr["seed"], it, rank))     # tái tạo đúng box / t của bước lỗi
                    boxes, t = prepare_train_boxes(batch["boxes"], batch["whwh"],
                                                   cfg["diffusion"]["num_proposals"], model.alphas_cumprod,
                                                   model.snr_scale, gen)
                    nan_debug.report(model, batch, boxes, t, text_table(batch["text"], dev), log)
                    sys.exit(f"[nan-debug] dừng ở NaN đầu tiên, it {it}")
            if not finite:
                win["skip"] += 1                        # bước NaN sẽ ghi NaN vào mọi weight
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()
                if a.nan_debug:
                    bp = nan_debug.bad_params(model)
                    if bp:
                        for n, p in list(model.named_parameters()):
                            if n in bp[:10]:
                                log(f"  [nan-debug] {n}: weight {nan_debug.tensor_stats(p)} | grad "
                                    f"{nan_debug.tensor_stats(p.grad) if p.grad is not None else None}")
                        sys.exit(f"[nan-debug] opt.step() với grad hữu hạn làm {len(bp)} tham số thành "
                                 f"không hữu hạn ở it {it}: {bp[:10]}")
            sched.step()
            win["loss"] += float(st["loss"])
            win["n"] += 1
            win["gn"].append(float(gn))
            it += 1

            if it % tr["log_every"] == 0 or it == tr["max_iter"]:
                el = time.time() - t_train
                spi = (time.time() - t_last) / max(win["n"], 1)
                eta = spi * (tr["max_iter"] - it)
                if crit is None:
                    lps = st.get("loss_per_stage")
                    detail = "" if lps is None else " (tầng " + " ".join(f"{float(x):.3f}" for x in lps) + ")"
                else:
                    lps = " ".join(f"{float(x):.2f}" for x in st["loss_per_stage"])
                    terms = " ".join(f"{LOG_LABEL.get(k, k)} {float(st[k + '_final']):.3f}" for k in crit.log_keys)
                    detail = f" (stage {lps}) | {terms} n_match {st['n_matched_final']}"
                log(f"  it {it:6d}/{tr['max_iter']} | loss {win['loss'] / win['n']:8.4f}{detail} | "
                    f"lr {sched.get_last_lr()[0]:.3e} | "
                    f"grad {np.median(win['gn']):.1f} | {spi:.3f} s/iter (đọc {win['data'] / win['n']:.3f}) | "
                    f"đã chạy {fmt_time(el)} | ETA {fmt_time(eta)}"
                    + (f" | ⚠️ bỏ {win['skip']} bước NaN" if win["skip"] else ""))
                gsum = gmon.summary()
                if gsum:
                    share = GradMonitor.share(gsum)
                    log("          grad theo nhóm: " + ", ".join(
                        f"{g} {v:.2f} ({share[g] * 100:.0f}%)" for g, v in list(gsum.items())[:5]))
                history.append({"iter": it, "loss": win["loss"] / win["n"],
                                "loss_per_stage": [float(x) for x in st.get("loss_per_stage", [])],
                                "lr": sched.get_last_lr()[0], "grad_norm_p50": float(np.median(win["gn"])),
                                "s_per_iter": spi, "skipped": win["skip"], "elapsed_sec": el})
                win = {"loss": 0.0, "n": 0, "gn": [], "data": 0.0, "skip": 0}
                t_last = time.time()

            is_eval = it % tr["eval_every"] == 0 or it == tr["max_iter"]
            is_ckpt = is_eval or it % tr["ckpt_every"] == 0
            if is_eval:
                is_best = False
                if main_proc:
                    res = run_val(model, ev_loader, text_table, cfg, log)
                    model.train()
                    val = res[sm]
                    is_best = best is None or val > best[sm]
                    dw = density_weight_ratio(model.backbone)
                    if task == "add":
                        K = cfg["eval"]["n_samples"]
                        keys = ("mean_iou_any", "box_hit50_any", f"best_iou@{K}_any", f"hit50@{K}_any",
                                f"best_iou@{K}_latest", "hole_cover", "on_object", "cnll_F1_n1_median")
                        if is_best:
                            best = {"iter": it, sm: val, **{k: res[k] for k in keys}}
                        log(f"[eval it {it}] {ev_split} ({res['n']} mẫu, {K} mẫu/ảnh): "
                            + " | ".join(f"{k} {res[k]:.4f}" for k in keys) + f" | {fmt_time(res['eval_sec'])}"
                            + ("" if dw is None else f" | ‖W density‖/‖W RGB‖ conv1 {dw:.4f}")
                            + (" | *best" if is_best else ""))
                        if res.get("attn"):
                            log("          attention query -> [t ; text ; vis] theo tầng: " + " | ".join(
                                "/".join(f"{a[k]:.2f}" for k in ("t", "text", "vis")) for a in res["attn"]))
                        history.append({"iter": it, "eval": res, "density_weight_ratio": dw})
                    else:
                        if is_best:
                            best = {"iter": it, sm: val, "oracle_recall": res["oracle_recall"],
                                    "AP50": res["AP50"], "score_AUC": res["score_AUC"],
                                    "oracle_recall_pt": res["oracle_recall_pt"], "AP_pt": res["AP_pt"]}
                        stage = " ".join(f"{v:.3f}" for v in res["oracle_recall_per_stage"])
                        log(f"[eval it {it}] {ev_split}: oracle_recall {res['oracle_recall']:.4f} | "
                            f"score_AUC {res['score_AUC']:.4f} | AP50 {res['AP50']:.4f} | "
                            f"AP75 {res['AP75']:.4f} | trần AP50 {res['oracle_score']['AP50']:.4f} | "
                            f"recall/stage {stage} | ĐIỂM: oracle_recall_pt {res['oracle_recall_pt']:.4f} "
                            f"AP_pt {res['AP_pt']:.4f} AUC_pt {res['score_AUC_pt']:.4f} | {fmt_time(res['eval_sec'])}"
                            + ("" if dw is None else f" | ‖W density‖/‖W RGB‖ conv1 {dw:.4f}")
                            + (" | *best" if is_best else ""))
                        history.append({"iter": it, "eval": {k: v for k, v in res.items()
                                                             if k not in ("oracle_score",)},
                                        "eval_oracle_AP50": res["oracle_score"]["AP50"],
                                        "density_weight_ratio": dw})
                if world > 1:
                    dist.barrier()
            if is_ckpt:
                t = time.time()
                save(it, is_eval and main_proc and is_best)
                log(f"          💾 last.pth{' + best.pth' if is_eval and is_best else ''} ({fmt_time(time.time() - t)})")
                slowest_chunk = max(slowest_chunk, time.time() - chunk_t0)
                chunk_t0 = time.time()
                halt = torch.zeros((), device=dev)
                if main_proc and a.max_hours and it < tr["max_iter"] and \
                        time.time() - t_boot + 1.15 * slowest_chunk > a.max_hours * 3600:
                    log(f"DỪNG ở iter {it}: đoạn kế tiếp (~{fmt_time(slowest_chunk)}) có thể vượt "
                        f"--max-hours {a.max_hours}. Chạy lại đúng lệnh + --resume.")
                    halt.fill_(1)
                if world > 1:
                    dist.broadcast(halt, src=0)
                if halt.item():
                    done = True
                    break
            if it >= tr["max_iter"]:
                done = True
                break
            t_data = time.time()

    if it >= tr["max_iter"]:
        log(f"[done] {tr['max_iter']} iter | {fmt_time(time.time() - t_train)} | best {best} -> {a.save_dir}")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
