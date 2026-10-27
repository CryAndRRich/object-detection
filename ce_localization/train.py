#!/usr/bin/env python3
"""Train CE-Loc (DELTA4+, `models/box_policy.py`) cho bài ADD: mẫu = (nhánh, lượt) CE-130, đích = lỗ mới nhất, `noise_per_image` bộ
(t, ε) mỗi ảnh. MỘT thí nghiệm = MỘT config (`config/delta/<run>.yaml`, kế thừa qua `base` — `utils/config.py`).

Chọn `best.pth` theo `eval.select_metric`:
  mean_iou_any (DELTA4 / 5.x) — mỗi `eval_every` iter sinh box thật từ nhiễu thuần trên `eval.limit` mẫu val cố định
  train_loss   (DELTA6+, như CE-Loc gốc; `eval.split: null`) — không val; cuối mỗi epoch TRỌN lấy loss train TB của epoch (mọi bước
               hữu hạn, cộng qua mọi GPU), thấp nhất = best; tổng dở dang lưu trong `last.pth` ⇒ `--resume` giữa epoch khớp từng số
`training.snapshot_every_epochs` N: cuối mỗi N epoch ghi weight (không optimizer) `snapshots/ep<E>.pth` — eval sau trên server
(`eval.py --ckpt`, `tools/plot_snapshots.py`) để xem độ ổn định theo epoch; không chọn best. `training.snapshot_initial`: thêm
`ep000.pth` = weight khởi tạo, ghi trước iter 0 (run mới, không ghi lại khi `--resume`).
`data.clip_cache` / `--clip-cache`: token CLIP ảnh train tính sẵn (`tools/build_clip_cache.py`) thay CLIP forward mỗi iter.
`data.train_target`: `latest_hole` (mặc định, như CE-Loc gốc) | `positive` (DELTA7+) — tập đích của mẫu = {mọi lỗ tới lượt t} ∪ {box
positive gán tay của `data.positive_file` / `--positive-file` (`box_review/ce130_train.jsonl`) có IoU với mọi lỗ <=
`data.positive_hole_iou`} (box trùng lỗ = chính lỗ đó, bỏ). `data.positive_sampling`: `random` (DELTA7, vắng = random) — mỗi iter MỘT box
ngẫu nhiên đều / ảnh, rút bằng generator seed theo (seed, iter, rank) như t / nhiễu; `all` (DELTA7.0.1) — mọi box của ảnh trong cùng
iter, mỗi box bộ (t, ε) riêng, chung phần mã hoá ảnh, loss TB trong ảnh rồi TB trên ảnh (`BoxPolicy.eps_loss` x0_index); `pair`
(DELTA7.0.2) — mỗi cặp (mẫu, box) là một phần tử dataset (epoch = mọi cặp).
Vòng lặp không ép CPU chờ GPU (loss / thống kê cộng dồn trên GPU, đọc về mỗi `log_every`; AMP: GradScaler tự bỏ bước grad inf / NaN).
Mỗi `ckpt_every` iter: `last.pth` ghi nguyên tử, đủ để `--resume` tái lập (thứ tự dữ liệu seed theo (seed, epoch), t / nhiễu theo
(seed, iter, rank)). Lịch lr cosine theo iter (+ warmup `cosine_warmup_iters`); AMP fp16 (`training.amp`, chỉ CUDA; refiner + cross-attn
box vật tự giữ fp32); bước NaN / inf bị bỏ (GradScaler dò thang ở vài bước đầu là bình thường).

  1 GPU:  python ../tools/run_on_free_gpu.py -- train.py --config config/delta/delta6.yaml --save-dir ../weights/add/delta6
  2 GPU (Kaggle): torchrun --standalone --nproc_per_node=2 train.py --config ... --save-dir ...
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

# allocator CUDA theo đoạn mở rộng được (bớt phân mảnh) — phải đặt TRƯỚC khi CUDA khởi tạo
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.data.clip_cache import ClipCache  # noqa: E402
from ce_localization.data.turns import (AddCache, CE130AddDataset, TurnIndex, collate_add, load_positives,  # noqa: E402
                                        to_device_add)
from ce_localization.engine.add_eval import add_metrics, predict_add  # noqa: E402
from ce_localization.engine.train_utils import (EpochBatches, PthCheckpoints, fmt_gpu_peak, gpu_peak, lr_factor,  # noqa: E402
                                               noise_seed, setup_dist)
from ce_localization.models.box_policy import boxes_to_unit, build_model  # noqa: E402
from ce_localization.models.clip_refiner import refiner_config, text_model_name  # noqa: E402
from ce_localization.models.text import TextTable, encode_class_names, encode_class_tokens  # noqa: E402
from ce_localization.utils.config import load_config  # noqa: E402
from ce_localization.utils.grad_monitor import GradMonitor  # noqa: E402
from ce_localization.utils.log import fmt_time, print_banner, run_env  # noqa: E402

# nhánh config quyết định kiến trúc: khác checkpoint thì KHÔNG resume (`training` / `eval` chỉ cảnh báo); khoá đường dẫn / số worker
# của `data` không ảnh hưởng phép toán (Kaggle gắn dataset ở đường dẫn khác nhau giữa các phiên)
MUST_MATCH = ("model", "diffusion", "data")
DATA_FREE = ("num_workers", "root", "density_root", "density_index", "samples_root", "turn_index", "cache_dir", "clip_cache",
             "positive_file")
# thống kê refiner trên dòng log (TB cửa sổ) và trong history — `BoxPolicy.pop_stats`
EXTRA_LOG = {"loss_eps": "ε", "loss_aux": "aux", "delta_abs": "|Δ|", "aux_ratio": "aux/x_t", "loss_giou": "1−GIoU"}


def config_diff(saved, cfg):
    """-> (lỗi, cảnh báo): nhánh config khác nhau giữa checkpoint và lần chạy này."""
    strip = lambda b, d: {k: v for k, v in (d or {}).items() if not (b == "data" and k in DATA_FREE)}  # noqa: E731
    errors = [k for k in MUST_MATCH if strip(k, saved.get(k)) != strip(k, cfg.get(k))]
    warns = [k for k in ("training", "eval") if saved.get(k) != cfg.get(k)]
    return errors, warns


def build_text_table(names, cfg, dev):
    """Embedding CLIP pooler (cond) + token chữ CLIP text của refiner (+ vân tay) cho `names`. Tách hàm để test thay bằng bản giả."""
    m = cfg["model"]
    tokens, sha = encode_class_tokens(names, text_model_name(refiner_config(m["refiner"])), device=str(dev))
    return TextTable(encode_class_names(names, m["clip_text"], device=str(dev)), tokens, sha)


def add_datasets(cfg, limit=None):
    """-> (ds_tr, ds_ev | None). ds_ev = `eval.limit` mẫu CỐ ĐỊNH của `eval.split` (hoán vị seed 12345) — chỉ để chọn checkpoint.
    ds_tr kèm cache token CLIP nếu có `data.clip_cache`, tập box positive nếu `data.train_target: positive`."""
    d, ev = cfg["data"], cfg["eval"]
    try:
        index = TurnIndex(d["turn_index"])
    except FileNotFoundError as e:
        sys.exit(str(e))
    cache = AddCache(d["cache_dir"]) if d.get("cache_dir") else None          # Kaggle: ảnh train đã letterbox

    def mk(split, n=None):
        ds = CE130AddDataset(index, d["root"], d["samples_root"], split, d["image_size"], cache=cache)
        if n:
            perm = np.random.default_rng(12345).permutation(len(ds.keys))[:n]
            ds.keys = [ds.keys[i] for i in sorted(perm.tolist())]
        return ds

    ds_tr = mk("train")
    if limit:
        ds_tr.keys = ds_tr.keys[:limit]
    if d.get("clip_cache"):
        ds_tr.clip_cache = ClipCache(d["clip_cache"])
    target = d.get("train_target", "latest_hole")
    if target not in ("latest_hole", "positive"):
        sys.exit(f"data.train_target {target!r} không thuộc ('latest_hole', 'positive')")
    if target == "positive":
        ds_tr.set_positives(load_positives(d["positive_file"]), d["positive_hole_iou"], d.get("positive_sampling", "random"))
        train_stems = {os.path.splitext(os.path.basename(v["sample"]))[0] for k, v in index.turns.items() if v["sample"].startswith("train/")}
        unknown = sorted(set(ds_tr.positives) - train_stems)
        if unknown:
            sys.exit(f"{d['positive_file']}: {len(unknown)} id không phải mẫu samples/train, vd {unknown[:3]}")
    return ds_tr, mk(ev["split"], ev.get("limit")) if ev["split"] else None


def eval_loader(ds, cfg, rank, world, nw):
    """DataLoader phần `rank::world` của `ds` (mọi GPU cùng sinh box); generator riêng ⇒ không rút RNG toàn cục."""
    sh = copy.copy(ds)
    sh.keys = ds.keys[rank::world]
    return DataLoader(sh, batch_size=cfg["eval"]["batch_size"], shuffle=False, num_workers=nw, collate_fn=collate_add,
                      pin_memory=torch.cuda.is_available(), generator=torch.Generator())


def run_eval(model, loader, text_table, cfg, log, rank=0, world=1):
    """Mỗi GPU sinh `n_samples` box / ảnh cho phần của mình (seed = rank), gom record về rank 0 chấm -> dict chỉ số (rank 0) | None."""
    t0 = time.time()
    ev = cfg["eval"]
    rec = predict_add(model, loader, text_table, ev["n_samples"], seed=rank, variants=(("", ev["sample_kw"]),),
                      amp=bool(cfg["training"].get("amp")), log_every=max(len(loader) // 4, 1), log=log)[""]
    if world > 1:
        parts = [None] * world if rank == 0 else None
        dist.gather_object(rec, parts, dst=0)
        rec = [r for p in parts for r in p] if rank == 0 else None
    if rank:
        return None
    res = add_metrics(rec)
    res["eval_sec"] = time.time() - t0
    return res


def advance_lr(sched):
    """= `LambdaLR.step()` mà không gọi step(): lịch lr theo SỐ ITER nên bước bị bỏ (NaN / GradScaler dò thang) vẫn phải tiến."""
    sched.last_epoch += 1
    lrs = [base * f(sched.last_epoch) for base, f in zip(sched.base_lrs, sched.lr_lambdas)]
    for g, lr in zip(sched.optimizer.param_groups, lrs):
        g["lr"] = lr
    sched._last_lr = lrs


def step_loss(net, batch, text_table, cfg, model, gen, dev, obj_gen=None):
    target, x0_index = batch["target"], None
    if "positives" in batch:                    # data.train_target positive (`pair`: dataset đã trả đúng box trong `target`)
        if cfg["data"].get("positive_sampling", "random") == "all":     # mọi box của ảnh, hàng r thuộc ảnh x0_index[r]
            target = torch.cat(batch["positives"]).to(target.device)
            x0_index = torch.repeat_interleave(torch.arange(len(batch["positives"])),        # CPU (`eps_loss` tự đưa lên)
                                               torch.tensor([len(p) for p in batch["positives"]]))
        else:                                   # random: MỘT box ngẫu nhiên / ảnh
            target = torch.stack([p[torch.randint(len(p), (), generator=obj_gen)] for p in batch["positives"]]).to(target.device)
    x0 = boxes_to_unit(target, torch.full_like(target, float(batch["images"].shape[-1])), model.box_coder)
    loss = net(batch["images"], text_table(batch["text"], dev), batch["valid_hw"], x0, k=cfg["diffusion"]["noise_per_image"],
               generator=gen, objects=batch["objects"], text_tokens=text_table.tokens(batch["text"], dev), clip_feats=batch.get("clip"),
               obj_generator=obj_gen, **({"x0_index": x0_index} if x0_index is not None else {}))
    return loss, {"loss": loss.detach(), **model.pop_stats()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--save-dir", required=True, help="vd ../weights/add/delta6 (weights/ đã .gitignore)")
    ap.add_argument("--resume", action="store_true", help="train tiếp từ <save-dir>/last.pth (không có cờ này mà last.pth đã có thì DỪNG)")
    ap.add_argument("--max-iter", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="batch TOÀN CỤC")
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N mẫu train đầu")
    ap.add_argument("--num-workers", type=int, default=None)
    for k in ("data-root", "samples-root", "turn-index", "cache-dir", "clip-cache", "positive-file"):
        ap.add_argument(f"--{k}", default=None, help=f"ghi đè data.{k.replace('-', '_')}")
    ap.add_argument("--max-hours", type=float, default=0.0, help="dừng sạch (ghi last.pth) nếu đoạn kế tiếp có thể vượt N giờ; 0 = tắt")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    tr, d = cfg["training"], cfg["data"]
    for k, v in (("max_iter", a.max_iter), ("lr", a.lr), ("batch_size", a.batch_size), ("eval_every", a.eval_every)):
        if v:
            tr[k] = v
    for k, v in (("limit", a.limit), ("root", a.data_root), ("samples_root", a.samples_root), ("turn_index", a.turn_index),
                 ("cache_dir", a.cache_dir), ("clip_cache", a.clip_cache), ("positive_file", a.positive_file)):
        if v:
            d[k] = v
    nw = d["num_workers"] if a.num_workers is None else a.num_workers

    t_boot = time.time()
    rank, world, dev = setup_dist(a.device)
    main_proc = rank == 0
    log = (lambda *s: print(*s, flush=True)) if main_proc else (lambda *s: None)
    if tr["batch_size"] % world:
        sys.exit(f"batch_size {tr['batch_size']} không chia hết cho {world} GPU")
    bpr = tr["batch_size"] // world
    ckm = PthCheckpoints(a.save_dir)
    if ckm.has_last() and not a.resume:
        sys.exit(f"\n{ckm.last_path} ĐÃ TỒN TẠI.\n  train tiếp: thêm --resume\n  train mới : đổi --save-dir\n")
    if a.resume and not ckm.has_last():
        sys.exit(f"\n--resume nhưng không thấy {ckm.last_path}. Sai --save-dir?\n")

    torch.manual_seed(tr["seed"] + rank)
    np.random.seed(tr["seed"] + rank)
    torch.backends.cudnn.benchmark = dev.type == "cuda"         # mọi ảnh cùng cỡ: cudnn dò thuật toán conv MỘT lần
    env = run_env(cfg, dev)
    env["world_size"] = world
    if main_proc:
        print_banner(f"TRAIN — {cfg['experiment']}", env)

    # ------------------------------------------------------------------ dữ liệu
    ev_split, snap_every = cfg["eval"]["split"], tr.get("snapshot_every_epochs")
    train_select = cfg["eval"]["select_metric"] == "train_loss"
    if train_select != (ev_split is None):
        sys.exit("eval.select_metric 'train_loss' đi cùng eval.split null (không val) — và ngược lại")
    ds_tr, ds_ev = add_datasets(cfg, d.get("limit"))
    ipe = len(ds_tr) // tr["batch_size"]
    if ipe == 0:
        sys.exit(f"train chỉ có {len(ds_tr)} mẫu < batch_size {tr['batch_size']}")
    log(f"[data] train {len(ds_tr)} mẫu ({ipe} iter/epoch, batch {bpr}/GPU × {world})"
        + (" | KHÔNG val: best = loss train TB / epoch thấp nhất" if train_select else f" | eval {ev_split} {len(ds_ev)} mẫu")
        + (f" | snapshot weight mỗi {snap_every} epoch" if snap_every else "")
        + f" | cache ảnh {d.get('cache_dir')} | cache CLIP {d.get('clip_cache')} | {fmt_time(time.time() - t_boot)}")
    if ds_tr.positives is not None:
        n_hole = np.asarray([ds_tr.index.turns[k]["t"] for k in ds_tr.keys])
        n_hand = np.asarray([len(ds_tr.positive_boxes(k)) for k in ds_tr.keys]) - n_hole
        mode = {"random": "1 box ngẫu nhiên / ảnh / iter", "all": "mọi box của ảnh trong cùng iter (loss TB trong ảnh)",
                "pair": f"mỗi cặp (mẫu, box) một phần tử: {len(ds_tr)} cặp"}[ds_tr.positive_sampling]
        p_hole = n_hole.sum() / (n_hole + n_hand).sum() if ds_tr.positive_sampling == "pair" else np.mean(n_hole / (n_hole + n_hand))
        log(f"[data] đích trong lỗ ∪ box tay không trùng lỗ (IoU <= {ds_tr.positive_hole_iou}, {d['positive_file']}) — {mode}: "
            f"TB {n_hole.mean():.2f} lỗ + {n_hand.mean():.2f} box tay, {(n_hand > 0).mean():.1%} mẫu có box tay | P(đích là lỗ) {p_hole:.3f}")
    sampler = EpochBatches(len(ds_tr), bpr, rank, world, tr["seed"])
    # worker BỀN (không dựng lại mỗi epoch); generator riêng: base_seed của worker không rút từ RNG toàn cục
    loader = DataLoader(ds_tr, batch_sampler=sampler, num_workers=nw, collate_fn=collate_add, pin_memory=dev.type == "cuda",
                        persistent_workers=nw > 0, generator=torch.Generator().manual_seed(tr["seed"] * 100003 + rank))

    # ------------------------------------------------------------------ text
    t = time.time()
    names = sorted(set().union(*(ds.classes() for ds in (ds_tr, ds_ev) if ds is not None)))
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
    model.set_text_fingerprint(text_table.token_sha)           # vân tay CLIP text vào clip_sha TRƯỚC khi resume (nạp tự kiểm)
    amp = bool(tr.get("amp")) and dev.type == "cuda"            # Kaggle T4: tensor core chỉ chạy fp16
    if ds_tr.clip_cache is not None:
        ds_tr.clip_cache.check(model, cfg, ds_tr.keys)
    if amp:
        model = model.to(memory_format=torch.channels_last)
    # BN thường (`sync_bn: false`): batch / GPU 32 = batch BN của bài; SyncBN chậm hơn
    if world > 1 and tr.get("sync_bn", True):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if world > 1:
        # mọi tham số luôn có grad ⇒ grad là view của bucket DDP (khỏi chép, không lệch stride); tạo sẵn để iter 0 cũng cộng thẳng vào
        # bucket; zero_grad giữ tensor (set_to_none=False)
        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.zeros_like(p)
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index] if dev.type == "cuda" else None,
                                                    find_unused_parameters=False, gradient_as_bucket_view=True) if world > 1 else model
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"[model] {n_learn / 1e6:.2f}M tham số train | box_coder {model.box_coder['type']} | "
        f"AMP {'fp16 + channels_last' if amp else 'tắt'} | {fmt_time(time.time() - t)}")
    opt = torch.optim.AdamW(model.parameters(), lr=float(tr["lr"]), weight_decay=float(tr["weight_decay"]), fused=dev.type == "cuda")
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))           # torch < 2.3 (venv test local)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, tr))

    start, best, history = 0, None, []
    ep_acc = torch.zeros(2, dtype=torch.float64, device=dev)          # train_loss: [tổng loss, số bước] epoch đang chạy (rank này)
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
        scaler.load_state_dict(st["scaler"])
        start, best, history = st["iter"], st["best"], st["history"]
        if main_proc:                                   # RNG toàn cục của rank 0
            torch.set_rng_state(st["rng"]["torch"])
            if train_select:                            # tổng dở dang của epoch (đã cộng qua mọi GPU, giữ ở rank 0)
                ep_acc += torch.tensor(st["epoch_loss"], dtype=torch.float64, device=dev)
        log(f"[resume] {ckm.last_path}: đã xong {start} iter, best {best}")
    if start >= tr["max_iter"]:
        sys.exit(f"\nĐã train đủ {tr['max_iter']} iter; muốn thêm thì tăng --max-iter.\n")

    ev_loader = eval_loader(ds_ev, cfg, rank, world, nw) if ds_ev is not None else None
    gmon = GradMonitor(model, every=tr["log_every"])
    gen = torch.Generator(device=dev.type)
    obj_gen = torch.Generator()                                 # CPU: chọn ngẫu nhiên `model.obj_max` box vật (ảnh nhiều vật hơn)
    sm = cfg["eval"]["select_metric"]

    def reduce_epoch_acc():
        """train_loss: cộng `ep_acc` của mọi GPU về rank 0 (rank khác về 0) ⇒ tổng toàn cục = tổng các rank, lưu được trong last.pth."""
        if world > 1:
            dist.all_reduce(ep_acc)
            if not main_proc:
                ep_acc.zero_()

    def save(it_next, is_best):
        if not main_proc:
            return
        ckm.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                  "scaler": scaler.state_dict(), "iter": it_next, "best": best, "history": history, "config": cfg, "env": env,
                  "rng": {"torch": torch.get_rng_state()}, **({"epoch_loss": ep_acc.tolist()} if train_select else {})}, is_best)
        with open(os.path.join(a.save_dir, "history.json"), "w") as f:
            json.dump({"config": cfg, "environment": env, "best": best, "history": history}, f, indent=1, ensure_ascii=False,
                      default=float)

    def new_window():
        """Cửa sổ log: tổng loss / thống kê / số bước bỏ là tensor TRÊN GPU (cộng float64 như tổng float Python trước đây) — chỉ đọc về
        CPU khi in log."""
        z = lambda dt: torch.zeros((), dtype=dt, device=dev)  # noqa: E731
        return {"loss": z(torch.float64), "n": 0, "gn": [], "data": 0.0, "skip": z(torch.long), "extra": {}}

    log(f"[boot ] khởi động xong {fmt_time(time.time() - t_boot)} — bắt đầu iter {start}/{tr['max_iter']}")
    if tr.get("snapshot_initial") and start == 0 and main_proc:
        path = ckm.save_snapshot({"model": model.state_dict(), "iter": 0, "epoch": 0, "config": cfg, "env": env}, "ep000.pth")
        log(f"          📸 {os.path.relpath(path, a.save_dir)} (weight khởi tạo)")

    # ------------------------------------------------------------------ vòng train
    t_train, t_last, it = time.time(), time.time(), start
    win = new_window()
    slowest_chunk, chunk_t0 = 0.0, time.time()
    model.train()
    done = False
    while not done:
        sampler.set(*divmod(it, ipe))
        t_data = time.time()
        for batch in loader:
            win["data"] += time.time() - t_data
            batch = to_device_add(batch, dev)
            if amp:
                batch["images"] = batch["images"].contiguous(memory_format=torch.channels_last)
            gen.manual_seed(noise_seed(tr["seed"], it, rank))
            obj_gen.manual_seed(noise_seed(tr["seed"], it, rank))
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                loss, st = step_loss(net, batch, text_table, cfg, model, gen, dev, obj_gen)
            opt.zero_grad(set_to_none=world == 1)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                        # grad về thang thật trước khi đo / clip
            gmon.maybe_record(it)
            # grad_clip null = KHÔNG clip (CE-Loc gốc), vẫn đo norm để log
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"] or float("inf"))
            ok = torch.isfinite(gn) & torch.isfinite(st["loss"])        # trên GPU, không đồng bộ
            # bước NaN / inf sẽ ghi NaN vào weight: bỏ. AMP: GradScaler tự bỏ (found_inf của unscale_ đưa thẳng vào AdamW fused, CPU
            # không chờ; loss NaN ⇒ grad NaN ⇒ cũng bị bỏ) rồi giảm thang ở update. Không AMP (CPU / test): kiểm trên CPU.
            if amp or bool(ok):
                scaler.step(opt)
            else:
                opt.zero_grad(set_to_none=world == 1)
            scaler.update()
            advance_lr(sched)
            loss_d = st["loss"].double()
            if train_select:
                ep_acc[0] += torch.where(ok, loss_d, 0.0)
                ep_acc[1] += ok.double()
            win["skip"] += (~ok).long()
            win["loss"] += loss_d
            for k_ in EXTRA_LOG:
                if k_ in st:
                    win["extra"][k_] = win["extra"].get(k_, 0.0) + st[k_].double()
            win["n"] += 1
            win["gn"].append(gn.detach().double())
            it += 1

            if it % tr["log_every"] == 0 or it == tr["max_iter"]:
                # đọc cửa sổ về CPU (đồng bộ duy nhất của cửa sổ)
                win.update(loss=float(win["loss"]), skip=int(win["skip"]), gn=torch.stack(win["gn"]).cpu().numpy(),
                           extra={k_: float(v) for k_, v in win["extra"].items()})
                el = time.time() - t_train
                spi = (time.time() - t_last) / max(win["n"], 1)
                peak = gpu_peak(dev)                    # đỉnh bộ nhớ GPU của cửa sổ log này (rank 0)
                detail = " | " + " ".join(f"{EXTRA_LOG[k_]} {v / win['n']:.4f}" for k_, v in win["extra"].items())
                log(f"  it {it:6d}/{tr['max_iter']} | loss {win['loss'] / win['n']:8.4f}{detail} | lr {sched.get_last_lr()[0]:.3e} | "
                    f"grad {np.median(win['gn']):.1f} | {spi:.3f} s/iter (đọc {win['data'] / win['n']:.3f}) | đã chạy {fmt_time(el)} | "
                    f"ETA {fmt_time(spi * (tr['max_iter'] - it))}" + (f" | AMP scale {scaler.get_scale():.0f}" if amp else "")
                    + fmt_gpu_peak(peak) + (f" | ⚠️ bỏ {win['skip']} bước NaN" if win["skip"] else ""))
                gsum = gmon.summary()
                if gsum:
                    share = GradMonitor.share(gsum)
                    log("          grad theo nhóm: " + ", ".join(f"{g} {v:.2f} ({share[g] * 100:.0f}%)" for g, v in list(gsum.items())[:5]))
                gates = model.refiner.gate_stats()      # refiner có rời identity không (0 lúc khởi tạo)
                log("          refiner |gate| attn/mlp theo block: " + " ".join(
                    f"{k_} {v[0]:.3f}/{v[1]:.3f}" for k_, v in gates.items() if not k_.startswith("head"))
                    + " | ‖head cuối‖ " + " ".join(f"{gates[k_]:.3f}" for k_ in gates if k_.startswith("head")))
                history.append({"iter": it, "loss": win["loss"] / win["n"], "lr": sched.get_last_lr()[0],
                                "grad_norm_p50": float(np.median(win["gn"])), "s_per_iter": spi, "skipped": win["skip"],
                                "elapsed_sec": el, **({} if peak is None else {"gpu_peak_gb": peak[0], "gpu_reserved_gb": peak[1]}),
                                **{k_: v / win["n"] for k_, v in win["extra"].items()}, "refiner_gates": gates})
                win = new_window()
                t_last = time.time()

            # train_loss: chỉ epoch TRỌN (như tác giả) — phần lẻ cuối không chấm, tổng dở dang vẫn vào last.pth
            is_eval = it % ipe == 0 if train_select else (it % tr["eval_every"] == 0 or it == tr["max_iter"])
            is_ckpt = is_eval or it % tr["ckpt_every"] == 0 or it == tr["max_iter"]
            is_best = False
            if train_select and is_ckpt:
                reduce_epoch_acc()
            if is_eval and train_select:
                if main_proc and ep_acc[1] > 0:
                    ep_loss = float(ep_acc[0] / ep_acc[1])
                    is_best = best is None or ep_loss < best["train_loss"]
                    if is_best:
                        best = {"iter": it, "epoch": -(-it // ipe), "train_loss": ep_loss}
                    log(f"[epoch {-(-it // ipe)}] it {it} | loss train TB {ep_loss:.5f} ({int(ep_acc[1])} bước)"
                        + (" | *best" if is_best else f" | best {best['train_loss']:.5f} @ it {best['iter']}"))
                    history.append({"iter": it, "epoch": -(-it // ipe), "epoch_train_loss": ep_loss, "epoch_steps": int(ep_acc[1])})
                ep_acc.zero_()
            elif is_eval:
                gpu_peak(dev)                                           # cửa sổ đo riêng cho val
                res = run_eval(model, ev_loader, text_table, cfg, log, rank, world)
                model.train()
                if main_proc:
                    vpeak = gpu_peak(dev)
                    if vpeak is not None:
                        res["gpu_peak_gb"], res["gpu_reserved_gb"] = vpeak
                    is_best = best is None or res[sm] > best[sm]
                    K = cfg["eval"]["n_samples"]
                    keys = ("mean_iou_any", "box_hit50_any", f"best_iou@{K}_any", f"hit50@{K}_any", f"best_iou@{K}_latest",
                            "hole_cover", "on_object", "cnll")
                    if is_best:
                        best = {"iter": it, sm: res[sm], **{k: res[k] for k in keys}}
                    log(f"[eval it {it}] {ev_split} ({res['n']} mẫu, {K} mẫu/ảnh): " + " | ".join(f"{k} {res[k]:.4f}" for k in keys)
                        + f" | {fmt_time(res['eval_sec'])}" + fmt_gpu_peak(vpeak) + (" | *best" if is_best else ""))
                    history.append({"iter": it, "eval": res})
                if world > 1:
                    dist.barrier()
            if snap_every and it % ipe == 0 and (it // ipe) % snap_every == 0 and main_proc:
                t = time.time()
                path = ckm.save_snapshot({"model": model.state_dict(), "iter": it, "epoch": it // ipe, "config": cfg, "env": env},
                                         f"ep{it // ipe:03d}.pth")
                log(f"          📸 {os.path.relpath(path, a.save_dir)} ({fmt_time(time.time() - t)})")
            if is_ckpt:
                t = time.time()
                save(it, is_eval and main_proc and is_best)
                log(f"          💾 last.pth{' + best.pth' if is_eval and is_best else ''} ({fmt_time(time.time() - t)})")
                slowest_chunk = max(slowest_chunk, time.time() - chunk_t0)
                chunk_t0 = time.time()
                halt = torch.zeros((), device=dev)
                if main_proc and a.max_hours and it < tr["max_iter"] and \
                        time.time() - t_boot + 1.15 * slowest_chunk > a.max_hours * 3600:
                    log(f"DỪNG ở iter {it}: đoạn kế tiếp (~{fmt_time(slowest_chunk)}) có thể vượt --max-hours {a.max_hours}. "
                        f"Chạy lại đúng lệnh + --resume.")
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
        if train_select and tr["max_iter"] % ipe:
            log(f"⚠️  max_iter {tr['max_iter']} không chia hết {ipe} iter / epoch: {tr['max_iter'] % ipe} bước cuối không vào epoch nào")
        log(f"[done] {tr['max_iter']} iter | {fmt_time(time.time() - t_train)} | best {best} -> {a.save_dir}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
