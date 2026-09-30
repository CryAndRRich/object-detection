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
                       --save-dir checkpoints/alpha0
  2 GPU (Kaggle):  torchrun --standalone --nproc_per_node=2 train.py --config ... --save-dir ...
Chạy > 5 phút => nohup nền, xem docs/EXPERIMENT_ALPHA.md mục 9.

Cửa chặn:
  G3 overfit: --limit 16 --max-iter 1500 --eval-split train --eval-every 500 [--lr 1e-4]
  G4 bench  : --bench 50  (không ghi gì, in s/iter tách đọc dữ liệu / tính toán + bộ nhớ đỉnh)

BETA (`data.targets: point`, docs/EXPERIMENT_BETA.md): đích train = box giả từ điểm density
(`data.points`, `data.pseudo_size`), loss chế độ `point`; eval định kỳ vẫn chấm trên box GT, chọn
`best.pth` theo `eval.select_metric` (BETA: `oracle_recall_pt`).
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
from ce_localization.engine import nan_debug  # noqa: E402
from ce_localization.engine.criterion import Criterion, build_targets  # noqa: E402
from ce_localization.engine.diffusion import prepare_train_boxes  # noqa: E402
from ce_localization.engine.evaluate import predict, score  # noqa: E402
from ce_localization.engine.train_utils import (PthCheckpoints, epoch_batches,  # noqa: E402
                                               noise_seed, setup_dist, warmup_multistep)
from ce_localization.models.backbone import density_weight_ratio  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.models.text import TextTable, encode_class_names  # noqa: E402
from ce_localization.utils.grad_monitor import GradMonitor  # noqa: E402
from ce_localization.utils.log import fmt_time, print_banner, run_env  # noqa: E402

# Nhánh config quyết định kiến trúc / bài toán: khác checkpoint thì KHÔNG resume. `training` /
# `eval` được phép khác (chỉ cảnh báo); `data.num_workers` / đường dẫn dữ liệu không ảnh hưởng phép
# toán (Kaggle gắn dataset ở đường dẫn khác nhau giữa các phiên).
MUST_MATCH = ("model", "diffusion", "matcher", "data", "loss")
DATA_FREE = ("num_workers", "root", "density_root", "density_index", "points")
# nhãn số hạng loss trên dòng log (giữ đúng nhãn cũ của ALPHA: ce / l1 / giou / iou)
LOG_LABEL = {"loss_ce": "ce", "loss_bbox": "l1", "loss_giou": "giou", "iou_matched": "iou",
             "loss_center": "center", "loss_size": "size", "center_px": "center_px"}


def config_diff(saved, cfg):
    """-> (lỗi, cảnh báo): nhánh config khác nhau giữa checkpoint và lần chạy này."""
    strip = lambda b, d: {k: v for k, v in (d or {}).items()  # noqa: E731
                          if not (b == "data" and k in DATA_FREE)}
    errors = [k for k in MUST_MATCH if strip(k, saved.get(k)) != strip(k, cfg.get(k))]
    warns = [k for k in ("training", "eval") if saved.get(k) != cfg.get(k)]
    return errors, warns


def build_text_table(names, cfg, dev):
    """Tách ra hàm riêng để test thay bằng embedding giả (không tải CLIP)."""
    return TextTable(encode_class_names(names, cfg["model"]["clip_text"], device=str(dev)))


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
    bs = cfg["eval"]["batch_size"] if cfg["eval"]["sampling_steps"] == 1 else 1
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=collate)


def run_val(model, loader, text_table, cfg, log):
    t0 = time.time()
    ev = cfg["eval"]
    rec, stage = predict(model, loader, text_table, cfg["diffusion"]["num_proposals"],
                         steps=ev["sampling_steps"], top_k=ev["top_k"], nms_thr=ev["nms_thr"],
                         seed=0, log_every=max(len(loader) // 4, 1), log=log)
    res = score(rec, stage, ev["top_k"], ev["nms_thr"], oracle=True)
    res["eval_sec"] = time.time() - t0
    return res


def bench(model, net, crit, loader_fn, text_table, cfg, opt, dev, n, log):
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
        batch = to_device(batch, dev)
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
    boxes, t = prepare_train_boxes(batch["boxes"], batch["whwh"], cfg["diffusion"]["num_proposals"],
                                   model.alphas_cumprod, model.snr_scale, gen)
    text = text_table(batch["text"], dev)
    logits, pred = net(batch["images"], text, batch["valid_hw"], boxes, t)
    return crit(logits, pred, build_targets(batch["boxes"], batch["whwh"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-dir", required=True, help="vd checkpoints/alpha0 (đã .gitignore)")
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
    nw = cfg["data"]["num_workers"] if a.num_workers is None else a.num_workers

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
        print_banner(f"TRAIN ALPHA — {cfg['experiment']}", env)

    # ------------------------------------------------------------------ dữ liệu
    root, size = cfg["data"]["root"], cfg["data"]["image_size"]
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
    ev_split = cfg["eval"]["split"]
    if ev_split == "train":
        ds_ev = CE130Dataset(root, "train", size, density=d_ev, density_index=dindex)
        ds_ev.items = ds_tr.items[:]                   # G3: eval trên CHÍNH các ảnh đã train
    else:
        ds_ev = CE130Dataset(root, ev_split, size, density=d_ev, density_index=dindex)
    if a.eval_limit:
        ds_ev.items = ds_ev.items[: a.eval_limit]
    ipe = len(ds_tr) // tr["batch_size"]
    if ipe == 0:
        sys.exit(f"train chỉ có {len(ds_tr)} ảnh < batch_size {tr['batch_size']}")
    log(f"[data] train {len(ds_tr)} ảnh ({ipe} iter/epoch, batch {bpr}/GPU × {world}) | "
        f"eval {ev_split} {len(ds_ev)} ảnh | density train {d_tr} / eval {d_ev} | đích train {targets}"
        + ("" if ptable is None else f" (điểm {cfg['data']['points']}, tham số tách đỉnh {ptable.params}, "
                                     f"cỡ giả {pseudo})")
        + f" | {fmt_time(time.time() - t_boot)}")

    def loader_fn(epoch, start_batch):
        ds_tr.epoch = epoch                            # density `mix`: RNG theo (seed, epoch, ảnh)
        bl = epoch_batches(len(ds_tr), bpr, rank, world, tr["seed"], epoch)[start_batch:]
        return DataLoader(ds_tr, batch_sampler=bl, num_workers=nw, collate_fn=collate,
                          pin_memory=dev.type == "cuda")

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
    # find_unused_parameters=True BẮT BUỘC: conv 3x3 đầu ra của một tầng FPN chỉ có gradient khi có
    # RoI rơi vào tầng đó. Ở canvas 512, P5 cần sqrt(diện tích) >= 448 px nên ở ALPHA0 (P5 chỉ đi
    # qua RoI) gần như không bao giờ được dùng; tầng khác cũng có thể trống ở một iteration.
    net = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[dev.index] if dev.type == "cuda" else None,
        find_unused_parameters=True) if world > 1 else model
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] memory={cfg['model']['memory']} | in_channels {model.backbone.in_channels} | {n_learn / 1e6:.2f}M tham số train | "
        f"N={cfg['diffusion']['num_proposals']} | {fmt_time(time.time() - t)}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: warmup_multistep(it, tr["steps"], tr["gamma"], tr["warmup_iters"], tr["warmup_factor"]))
    crit = Criterion(cfg["loss"], cfg["matcher"], mode=targets)

    if a.bench:
        return bench(model, net, crit, loader_fn, text_table, cfg, opt, dev, a.bench, log)

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
            batch = to_device(batch, dev)
            gen.manual_seed(noise_seed(tr["seed"], it, rank))
            loss, st = _step_loss(net, crit, batch, text_table, cfg, model, gen, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gmon.maybe_record(it)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            finite = bool(torch.isfinite(gn)) and bool(torch.isfinite(loss))
            if a.nan_debug:
                log(f"  [nan-debug] it {it} | loss {float(loss):.4f} | grad trước clip {float(gn):.4g} | "
                    f"lr {sched.get_last_lr()[0]:.3e} | density {batch.get('density_kind')} | ảnh {batch['image_id']}")
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
                lps = " ".join(f"{float(x):.2f}" for x in st["loss_per_stage"])
                terms = " ".join(f"{LOG_LABEL.get(k, k)} {float(st[k + '_final']):.3f}" for k in crit.log_keys)
                log(f"  it {it:6d}/{tr['max_iter']} | loss {win['loss'] / win['n']:8.3f} (stage {lps}) | "
                    f"{terms} n_match {st['n_matched_final']} | lr {sched.get_last_lr()[0]:.3e} | "
                    f"grad {np.median(win['gn']):.1f} | {spi:.3f} s/iter (đọc {win['data'] / win['n']:.3f}) | "
                    f"đã chạy {fmt_time(el)} | ETA {fmt_time(eta)}"
                    + (f" | ⚠️ bỏ {win['skip']} bước NaN" if win["skip"] else ""))
                gsum = gmon.summary()
                if gsum:
                    share = GradMonitor.share(gsum)
                    log("          grad theo nhóm: " + ", ".join(
                        f"{g} {v:.2f} ({share[g] * 100:.0f}%)" for g, v in list(gsum.items())[:5]))
                history.append({"iter": it, "loss": win["loss"] / win["n"],
                                "loss_per_stage": [float(x) for x in st["loss_per_stage"]],
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
                    if is_best:
                        best = {"iter": it, sm: val, "oracle_recall": res["oracle_recall"],
                                "AP50": res["AP50"], "score_AUC": res["score_AUC"],
                                "oracle_recall_pt": res["oracle_recall_pt"], "AP_pt": res["AP_pt"]}
                    stage = " ".join(f"{v:.3f}" for v in res["oracle_recall_per_stage"])
                    dw = density_weight_ratio(model.backbone)
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
