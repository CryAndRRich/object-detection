#!/usr/bin/env python3
"""Train CE-Loc trên CE-130.

Chỉ số theo dõi mỗi epoch (trên val):
  oracle_recall (theo TỪNG tầng)  GT được ít nhất một box phủ, không nhìn score/matcher.
                                  Chọn checkpoint bằng chỉ số matcher KHÔNG thấy —
                                  `iou_matched` đã đánh lừa hai lần ở vòng 1.
  score_AUC                       box khớp GT có được score cao hơn box còn lại không.
  label_stability                 % nhãn giữ nguyên giữa hai epoch.
  grad norm theo nhóm module      biết con số tổng đến từ đâu.

⚠️ Chỉ số val ở đây là KHỬ NHIỄU MỘT BƯỚC từ GT đã thêm nhiễu, KHÔNG phải suy luận DDIM
từ nhiễu thuần (cao gần 2× số thật). Số để báo cáo chỉ lấy từ `eval.py`.

Chạy (> 5 phút => nohup nền, xem README):
  cd object-detection/ce_localization
  LOG=/mnt/disk1/aiotlab/haitn/log/train_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- train.py --save-dir checkpoints/<tên> \\
      > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.data.ce130_dataset import CE130Detection  # noqa: E402
from ce_localization.data.loader import make_loader, model_inputs  # noqa: E402
from ce_localization.models.criterion import SetCriterion  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.utils.checkpoint import (CheckpointManager, rng_state,  # noqa: E402
                                              set_rng_state)
from ce_localization.utils.grad_monitor import GradMonitor  # noqa: E402
from ce_localization.utils.log import (array_stats, fmt_time, print_banner,  # noqa: E402
                                       run_env)
from ce_localization.utils.metrics_np import oracle_hits, quality  # noqa: E402

# Chỉ số được phép chọn checkpoint — CAO hơn là tốt hơn. KHÔNG có `iou_matched`.
SELECT_METRICS = ("oracle_recall", "score_AUC")


def select_metric(cfg):
    m = cfg.get("eval", {}).get("select_metric", "oracle_recall")
    if m not in SELECT_METRICS:
        raise ValueError(f"eval.select_metric={m!r}, phải là một trong {SELECT_METRICS}")
    return m


def label_stability(before, after):
    """% cặp (image_id, pred_idx) -> gt_idx giữ nguyên giữa hai epoch."""
    if not before:
        return float("nan")
    shared = set(before) & set(after)
    if not shared:
        return 0.0
    return sum(before[k] == after[k] for k in shared) / len(shared)


def layer_recall(layers, targets):
    """(hit, n_gt) cho TỪNG tầng của một batch."""
    out = []
    for boxes, _ in layers:
        hit = tot = 0
        for b, gt in zip(boxes, targets):
            h, n = oracle_hits(b.detach().float().cpu().numpy(), gt.cpu().numpy())
            hit, tot = hit + h, tot + n
        out.append((hit, tot))
    return out


class Trainer:
    def __init__(self, cfg, model, ld_tr, ld_va, ckpt, dev, epochs, log_every):
        self.cfg, self.model, self.dev = cfg, model, dev
        self.ld_tr, self.ld_va, self.ckpt = ld_tr, ld_va, ckpt
        self.epochs, self.log_every = epochs, log_every
        tr, d = cfg["training"], cfg["diffusion"]
        self.n_train, self.n_eval = d["num_proposals_train"], d["num_proposals_eval"]
        self.grad_clip = tr["grad_clip"]
        self.sm = select_metric(cfg)

        self.crit = SetCriterion.from_config(cfg)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(trainable, lr=float(tr["lr"]),
                                     weight_decay=float(tr["weight_decay"]))
        self.gen = torch.Generator(device=dev.type).manual_seed(tr["seed"])
        self.gmon = GradMonitor(model, every=10)

        self.history, self.best, self.prev_labels = [], None, {}
        self.start_ep, self.elapsed_before = 0, 0.0

    # --------------------------------------------------------------- resume

    def resume(self):
        """Nạp last.pt: model + optimizer + RNG + history. Nạp về CPU vì trạng thái RNG
        phải là tensor CPU; load_state_dict tự chuyển tensor sang device của tham số."""
        st = self.ckpt.load_last(map_location="cpu")
        errs, warns = CheckpointManager.config_mismatch(st["config"], self.cfg)
        if errs:
            raise SystemExit(f"\nKHÔNG resume được: config khác checkpoint ở {errs}. "
                             f"Dùng --save-dir khác.\n")
        if warns:
            print(f"[resume] ⚠️  nhánh {warns} khác lần trước — vẫn tiếp tục", flush=True)
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["optimizer"])
        set_rng_state(st["rng"], self.gen)
        self.history, self.best, self.prev_labels = st["history"], st["best"], st["prev_labels"]
        self.start_ep = st["epoch"] + 1
        self.elapsed_before = self.history[-1]["elapsed_sec"] if self.history else 0.0
        print(f"[resume] {self.ckpt.last_path}: xong epoch {st['epoch']}, best {self.best}"
              f" -> tiếp từ epoch {self.start_ep}", flush=True)

    # ----------------------------------------------------------------- step

    def _forward(self, batch, n_prop, generator):
        targets = [b.to(self.dev) for b in batch["boxes"]]
        x_t, t, _ = self.model.build_inputs(targets, n_prop, batch["valid_h"],
                                            generator=generator)
        vh = torch.as_tensor(batch["valid_h"], dtype=torch.float32, device=self.dev)
        layers = self.model(x_t, t, valid_h=vh, **model_inputs(batch, self.dev))
        return targets, layers

    def train_epoch(self, ep):
        self.model.train()
        run, n_seen, labels_now, grad_norms, n_skip = {}, 0, {}, [], 0
        for bi, batch in enumerate(self.ld_tr):
            targets, layers = self._forward(batch, self.n_train, self.gen)
            loss, st, idx = self.crit(layers, targets)

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.gmon.maybe_record(bi)                 # đo TRƯỚC clip: độ lớn thật
            gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            if not torch.isfinite(gn):
                # Step với grad NaN sẽ ghi NaN vào MỌI weight (và last.pt). Bỏ bước này.
                n_skip += 1
                self.opt.zero_grad(set_to_none=True)
                continue
            grad_norms.append(float(gn))
            self.opt.step()

            for im, (pi, gi) in zip(batch["image_id"], idx):
                for p, g in zip(pi.tolist(), gi.tolist()):
                    labels_now[(im, p)] = g
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    run[k] = run.get(k, 0.0) + v
            n_seen += 1
            if self.log_every and bi % self.log_every == 0:
                print(f"  ep{ep:3d} b{bi:4d}/{len(self.ld_tr)} loss {float(loss):8.3f} "
                      f"(mean {st['loss_mean']:6.3f}) iou {st['iou_matched']:.4f}",
                      flush=True)

        stability = label_stability(self.prev_labels, labels_now)
        self.prev_labels = labels_now
        stats = {k: v / max(n_seen, 1) for k, v in run.items()}
        return stats, stability, grad_norms, n_skip

    @torch.no_grad()
    def validate(self):
        """Val cố định seed 0 để các epoch so được với nhau."""
        self.model.eval()
        gen = torch.Generator(device=self.dev.type).manual_seed(0)
        agg, n_batch, hits, iou_layer, scores, aucs = {}, 0, None, [], [], []
        for batch in self.ld_va:
            targets, layers = self._forward(batch, self.n_eval, gen)
            _, st, _ = self.crit(layers, targets)

            lr = layer_recall(layers, targets)
            hits = lr if hits is None else [(a + c, b + d) for (a, b), (c, d) in zip(hits, lr)]
            iou_layer.append(st["iou_matched_per_layer"])
            fb, fl = layers[-1]
            prob = fl.float().sigmoid().cpu().numpy()
            scores.append(prob.ravel())
            for i, gt in enumerate(targets):
                aucs.append(quality(fb[i].float().cpu().numpy(), prob[i],
                                    gt.cpu().numpy(), size=1)[3])
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    agg[k] = agg.get(k, 0.0) + v
            n_batch += 1

        out = {k: v / max(n_batch, 1) for k, v in agg.items()}
        out["oracle_recall_per_layer"] = [h / max(n, 1) for h, n in hits]
        out["oracle_recall"] = out["oracle_recall_per_layer"][-1]
        out["iou_per_layer"] = np.mean(iou_layer, axis=0).tolist()
        out["score"] = array_stats(np.concatenate(scores))
        a = np.array([x for x in aucs if not np.isnan(x)])
        out["score_AUC"] = float(a.mean()) if len(a) else float("nan")
        return out

    # ------------------------------------------------------------------ log

    def _warnings(self, va, stability, grad_norms, n_skip):
        rec, w = va["oracle_recall_per_layer"], []
        if rec[-1] <= rec[0] + 1e-4:
            w.append("recall KHÔNG tăng qua các tầng -> cộng dồn chưa mang lại gì")
        if not np.isnan(stability) and stability < 0.005:
            w.append(f"label_stability {stability:.3f} dưới sàn ngẫu nhiên")
        if va["score"].get("std", 1.0) < 0.05:
            w.append("std_score < 0.05 — head score có thể đang kẹt ở hằng số")
        # Grad norm lớn trước clip KHÔNG tự nó là vấn đề (AdamW gần như bất biến theo
        # thang); đáng báo là norm TĂNG DẦN so với epoch đầu.
        g0 = self.history[0]["grad_norm"].get("p50") if self.history else None
        g_now = float(np.median(grad_norms)) if grad_norms else float("nan")
        if g0 and g_now > 5 * g0:
            w.append(f"grad norm trung vị {g_now:.1f} gấp >5x epoch đầu ({g0:.1f})")
        if n_skip:
            w.append(f"bỏ {n_skip} bước vì grad norm NaN/inf")
        return w

    def _print_epoch(self, ep, tr, va, stability, grad_norms, gsum, warnings, t0, t_start):
        eta = (time.time() - t_start) / (ep - self.start_ep + 1) * (self.epochs - ep - 1)
        mark = lambda m: " *" if m == self.sm else ""                     # noqa: E731
        print(f"[ep {ep:3d}] train {tr['loss']:8.3f} | val {va['loss']:8.3f} "
              f"| oracle_recall {va['oracle_recall']:.4f}{mark('oracle_recall')} "
              f"| score_AUC {va['score_AUC']:.4f}{mark('score_AUC')} "
              f"| stab {stability:.3f} | {fmt_time(time.time() - t0)} | ETA {fmt_time(eta)}",
              flush=True)
        rec = va["oracle_recall_per_layer"]
        print(f"          recall/tầng: {' '.join(f'{v:.3f}' for v in rec)}", flush=True)
        share = GradMonitor.share(gsum)
        g_now = float(np.median(grad_norms)) if grad_norms else float("nan")
        print(f"          grad norm trung vị {g_now:.1f} | theo nhóm: "
              + ", ".join(f"{g} {v:.1f} ({share[g] * 100:.0f}%)"
                          for g, v in list(gsum.items())[:4]), flush=True)
        for w in warnings:
            print(f"          ⚠️  {w}", flush=True)

    def _write_history(self, env):
        top = max(self.history, key=lambda e: e["val"][self.sm])
        summary = {
            "select_metric": self.sm,
            "epochs_completed": len(self.history),
            f"best_{self.sm}": top["val"][self.sm],
            "best_oracle_recall": top["val"]["oracle_recall"],
            "best_epoch": top["epoch"],
            "total_time": fmt_time(self.history[-1]["elapsed_sec"]),
            "epochs_with_warnings": [e["epoch"] for e in self.history if e["warnings"]],
        }
        with open(os.path.join(self.ckpt.save_dir, "history.json"), "w") as f:
            json.dump({"summary": summary, "environment": env, "config": self.cfg,
                       "dataset": {"train": self.ld_tr.dataset.ds.stats(),
                                   "val": self.ld_va.dataset.ds.stats()},
                       "epochs": self.history}, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ fit

    def fit(self, env):
        t_start = time.time()
        for ep in range(self.start_ep, self.epochs):
            t0 = time.time()
            tr, stability, grad_norms, n_skip = self.train_epoch(ep)
            va = self.validate()
            gsum = self.gmon.summary()
            warnings = self._warnings(va, stability, grad_norms, n_skip)
            self._print_epoch(ep, tr, va, stability, grad_norms, gsum, warnings, t0, t_start)

            self.history.append({
                "epoch": ep, "train": tr, "val": va, "label_stability": stability,
                "grad_norm": array_stats(grad_norms), "grad_norm_by_group": gsum,
                "skipped_steps": n_skip, "warnings": warnings,
                "epoch_sec": time.time() - t0,
                "elapsed_sec": self.elapsed_before + time.time() - t_start})
            self._write_history(env)
            self.save(ep, va)
        print(f"[done] {fmt_time(time.time() - t_start)} | best {self.best}", flush=True)

    def save(self, ep, va):
        """last.pt MỖI epoch, chép sang best.pt khi cải thiện. Ghi nguyên tử."""
        is_best = self.best is None or va[self.sm] > self.best[self.sm]
        if is_best:
            self.best = {"epoch": ep, self.sm: va[self.sm],
                         "oracle_recall": va["oracle_recall"], "loss": va["loss"]}
        t = time.time()
        self.ckpt.save({"model": self.model.state_dict(), "optimizer": self.opt.state_dict(),
                        "config": self.cfg, "epoch": ep, "best": self.best,
                        "history": self.history, "prev_labels": self.prev_labels,
                        "rng": rng_state(self.gen)}, is_best)
        print(f"          💾 last.pt{' + best.pt' if is_best else ''} ({fmt_time(time.time() - t)})",
              flush=True)


def check_cache(cache_dir, config_path):
    """Báo thiếu cache NGAY, trước ~15 phút khởi động, kèm lệnh sinh cache."""
    for sp in ("train", "val"):
        meta = os.path.join(cache_dir, f"{sp}_meta.json")
        if not os.path.exists(meta):
            raise SystemExit(
                f"\nKHÔNG THẤY CACHE split '{sp}': {meta}\nSinh bằng:\n"
                f"  LOG=/mnt/disk1/aiotlab/haitn/log/cache_{sp}_$(date +%m%d_%H%M).log\n"
                f"  nohup python ../tools/run_on_free_gpu.py -- tools/build_cache.py \\\n"
                f"      --config {config_path} --split {sp} --out {cache_dir} \\\n"
                f"      > $LOG 2>&1 &\n  echo \"PID $! -> $LOG\"\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--save-dir", required=True, help="vd checkpoints/<tên>; .gitignore đã có")
    ap.add_argument("--cache", default="../data/cache_clip_1024",
                    help="thư mục cache patch token; --cache '' để chạy CLIP mỗi batch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="thu nhỏ dataset để chạy thử")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every-n-batch", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="train tiếp từ <save-dir>/last.pt. Không có cờ này mà last.pt đã "
                         "tồn tại thì DỪNG — không ghi đè nhầm một lần train đang dở.")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    tr_cfg = cfg["training"]
    epochs = a.epochs or tr_cfg["epochs"]
    bs = a.batch_size or tr_cfg["batch_size"]

    ckpt = CheckpointManager(a.save_dir)
    if ckpt.has_last() and not a.resume:
        raise SystemExit(f"\n{ckpt.last_path} ĐÃ TỒN TẠI.\n  train tiếp: thêm --resume\n"
                         f"  train mới : đổi --save-dir\n")
    if a.resume and not ckpt.has_last():
        raise SystemExit(f"\n--resume nhưng không thấy {ckpt.last_path}. Sai --save-dir?\n")
    if a.cache:
        check_cache(a.cache, a.config)

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(tr_cfg["seed"])
    env = run_env(cfg, dev)
    print_banner(f"TRAIN — {env['experiment']}", env)

    t_boot = time.time()
    ds_tr = CE130Detection.from_config(cfg, "train",
                                       flip_prob=cfg["data"].get("flip_prob", 0.0),
                                       seed=tr_cfg["seed"])
    ds_va = CE130Detection.from_config(cfg, "val")          # eval: KHÔNG lật ảnh
    if a.limit:
        ds_tr.items = ds_tr.items[: a.limit]
        ds_va.items = ds_va.items[: max(a.limit // 2, 1)]
    print(f"[train] {ds_tr.stats()}\n[val  ] {ds_va.stats()}", flush=True)
    nw = cfg["data"]["num_workers"]
    ld_tr = make_loader(ds_tr, a.cache, "train", bs, nw, dev, train=True)
    ld_va = make_loader(ds_va, a.cache, "val", bs, nw, dev, train=False)
    print(f"[boot ] dataset + loader: {fmt_time(time.time() - t_boot)}", flush=True)

    t_model = time.time()
    model = build_model(cfg).to(dev)
    print(f"[boot ] dựng model (tải CLIP): {fmt_time(time.time() - t_model)}", flush=True)
    n_learn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] tham số học được {n_learn / 1e6:.2f}M / tổng {n_all / 1e6:.1f}M | "
          f"N train/eval {cfg['diffusion']['num_proposals_train']}/"
          f"{cfg['diffusion']['num_proposals_eval']} | matcher {cfg['matcher']['method']} | "
          f"lr {tr_cfg['lr']}", flush=True)

    log_every = a.log_every_n_batch
    if log_every is None:
        log_every = max(len(ld_tr) // 5, 1) if len(ld_tr) >= 10 else 0
    trainer = Trainer(cfg, model, ld_tr, ld_va, ckpt, dev, epochs, log_every)
    if a.resume:
        trainer.resume()
    if trainer.start_ep >= epochs:
        raise SystemExit(f"\nĐã train đủ {epochs} epoch; muốn thêm thì tăng --epochs.\n")
    print(f"[boot ] tổng khởi động {fmt_time(time.time() - t_boot)} — bắt đầu epoch "
          f"{trainer.start_ep} ({len(ld_tr)} batch/epoch) | chọn best.pt theo val "
          f"{trainer.sm}", flush=True)
    trainer.fit(env)


if __name__ == "__main__":
    main()
