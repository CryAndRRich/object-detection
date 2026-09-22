#!/usr/bin/env python3
"""Train CE-Loc — EXPERIMENT A vòng 2. Thiết kế: `docs/EXPERIMENT_A_PLAN.md`.

BỐN CHỈ SỐ THEO DÕI, quan trọng ngang loss (vòng 1 thiếu chúng nên mù suốt 5 lượt sửa):

  1. `oracle_recall` THEO TỪNG TẦNG — chỉ số CHÍNH của thí nghiệm. Đường phẳng nghĩa là
     cộng dồn không mang lại gì, và kết luận đó phải nhìn thấy được trước khi xây thêm
     bất cứ thứ gì. Nó cũng là thứ `iou_matched` KHÔNG thấy: `iou_matched` chỉ trung bình
     trên những cặp matcher đã chọn, nên một tầng siết chặt box nó đã có trong khi ĐÁNH
     MẤT vùng phủ của GT khác sẽ có loss TỐT HƠN mà thực tế TỆ ĐI. Đo trên C1 vòng 1:
     `iou_matched` tăng 0,3426 -> 0,3528 qua sáu vòng còn `oracle_recall` GIẢM
     0,138 -> 0,133. Nó đã đánh lừa HAI LẦN.
  2. `label_stability` — % cặp (ảnh, box) -> gt giữ nguyên giữa hai epoch. Vòng 1 đo được
     **0,018** (hơn 98 % nhãn đổi mỗi epoch, suốt 299 epoch, không hề cải thiện). Đây là
     biến nền quan trọng nhất và là lý do vòng 2 đổi sang SimOTA.
  3. `std_score` — dưới 0,05 nghĩa là head kẹt ở hằng số (focal với alpha=0,25 và head
     không phân biệt hội tụ về một giá trị cố định).
  4. `iou_matched` — để đọc cho tiện, KHÔNG dùng chọn checkpoint.

CHẠY TRÊN SERVER (job dài -> chạy nền, kèm PID + logfile):
  cd object-detection/count_editing/CE-LocModel
  LOG=/mnt/disk1/aiotlab/haitn/log/round2_a_$(date +%m%d_%H%M).log
  nohup python tools/run_on_free_gpu.py -- train.py \
      --config config/experiment_a.yaml \
      --cache ../../data/cache_clip \
      --save-dir checkpoints/round2_a \
      > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ce130_dataset import PatchCache, normalize_for_clip  # noqa: E402
from data.factory import build_dataset  # noqa: E402
from models.criterion import SetCriterion, loss_from_layers  # noqa: E402
from models.detector import build_model  # noqa: E402
from utils.box_ops import box_iou, cxcywh_to_xyxy  # noqa: E402


class TorchWrap(Dataset):
    """Bọc dataset numpy (CE-130 hoặc COCO, xem `data/factory.py`) thành torch Dataset.

    Có `cache` thì trả patch/text token đã tính sẵn và BỎ HẲN ảnh — đo trên A30: CLIP
    chiếm 76,8 % thời gian mỗi batch, nên cache nhanh hơn ~4,3 lần (328 -> 76 ms).
    """

    def __init__(self, ds, cache=None):
        self.ds = ds
        self.cache = cache

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # Có cache thì ảnh không bao giờ dùng tới, nên đừng giải mã (16,9 ms/ảnh).
        m = self.ds.__getitem__(i, need_image=self.cache is None)
        out = {
            "boxes": torch.from_numpy(m["boxes"]).float(),
            "labels": torch.from_numpy(m["labels"]).long(),
            "text": m["text"],
            "valid_h": m["valid_h"],
            "image_id": m["image_id"],
        }
        if self.cache is None:
            out["pixel_values"] = torch.from_numpy(normalize_for_clip(m["image"]))
        else:
            patch, text = self.cache.get(m["image_id"], m["text"], m["flipped"])
            out["patch_raw"] = torch.from_numpy(patch)
            out["text_raw"] = torch.from_numpy(text)
        return out


def collate(batch):
    """Số box mỗi ảnh khác nhau -> giữ dạng list, không pad ở đây."""
    out = {k: [b[k] for b in batch]
           for k in ("boxes", "labels", "text", "valid_h", "image_id")}
    for k in ("pixel_values", "patch_raw", "text_raw"):
        if k in batch[0]:
            out[k] = torch.stack([b[k] for b in batch])
    return out


def fmt_time(seconds):
    """3661 -> '1h01m01s'. Dùng cho cả thời gian đã chạy lẫn ETA."""
    seconds = int(max(seconds, 0))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def array_stats(x):
    """Phân phối đầy đủ, không chỉ trung bình — trung bình che mất đuôi."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    q = np.percentile(x, [1, 25, 50, 75, 99])
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "p1": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p99": float(q[4])}


def label_stability(before, after):
    """% cặp (image_id, pred_idx) -> gt_idx giữ nguyên giữa hai epoch."""
    if not before:
        return float("nan")
    shared = set(before) & set(after)
    if not shared:
        return 0.0
    return sum(before[k] == after[k] for k in shared) / len(shared)


def model_inputs(batch, dev):
    """Trả kwargs cho model: ảnh thô, hoặc token đã cache.

    `non_blocking=True` là thứ làm `pin_memory` của loader có giá trị: trên bộ nhớ đã ghim
    thì copy host->device chạy bất đồng bộ và chồng lấn với tính toán.
    """
    nb = dev.type == "cuda"
    if "patch_raw" in batch:
        return {"patch_raw": batch["patch_raw"].to(dev, non_blocking=nb),
                "text_raw": batch["text_raw"].to(dev, non_blocking=nb)}
    return {"pixel_values": batch["pixel_values"].to(dev, non_blocking=nb),
            "texts": batch["text"]}


@torch.no_grad()
def oracle_recall(pred_cxcywh, gt_cxcywh, iou_thr=0.5):
    """Tỉ lệ GT được ÍT NHẤT MỘT box phủ, bỏ qua hoàn toàn score và matcher.

    Đây là đại lượng mà `loss` không nhìn thấy. Cùng định nghĩa với
    `tools/measure_box_quality.py::quality` nên số ở đây so trực tiếp được với tool đó.
    """
    # Trả về n_gt kể cả khi KHÔNG có dự đoán nào: những GT đó là chưa được phủ, không
    # phải không tồn tại. Trả 0 sẽ làm co mẫu số và báo cáo như thể ảnh khó chưa từng có.
    if gt_cxcywh.numel() == 0:
        return 0.0, 0
    if pred_cxcywh.numel() == 0:
        return 0.0, int(gt_cxcywh.shape[0])
    iou = box_iou(cxcywh_to_xyxy(pred_cxcywh), cxcywh_to_xyxy(gt_cxcywh))
    if isinstance(iou, tuple):
        iou = iou[0]
    return float((iou.max(dim=0).values >= iou_thr).sum()), int(gt_cxcywh.shape[0])


def per_layer_recall(layers, targets):
    """`oracle_recall` cho TỪNG tầng — chỉ số chính để đọc EXPERIMENT A."""
    out = []
    for boxes, _ in layers:
        hit = tot = 0
        for b, gt in zip(boxes, targets):
            h, n = oracle_recall(b.detach().cpu(), gt.cpu())
            hit += h
            tot += n
        out.append(hit / max(tot, 1))
    return out


def write_json(save_dir, env, cfg, history, ds_train, ds_val):
    """Ghi MỌI chỉ số vào history.json sau MỖI epoch.

    Ghi mỗi epoch (không đợi tới cuối) để job chết giữa chừng vẫn đọc được. Nó chứa đủ
    thứ để chẩn đoán mà không phải chạy lại: môi trường, toàn bộ config, thống kê
    dataset, và mọi chỉ số từng epoch kèm phân phối (mean/std/min/max/phân vị).
    """
    # Tính `best` TỪ history, không lấy từ đối số: hàm này được gọi TRƯỚC khi vòng train
    # cập nhật `best`, nên dùng đối số sẽ lệch đúng một epoch.
    # Chọn theo `oracle_recall` — CAO hơn là tốt hơn.
    top = max(history, key=lambda e: e["val"]["oracle_recall"]) if history else None
    summary = {
        "select_metric": "oracle_recall",
        "epochs_completed": len(history),
        "best_oracle_recall": top["val"]["oracle_recall"] if top else None,
        "best_epoch": top["epoch"] if top else None,
        "best_val_loss": top["val"]["loss"] if top else None,
        "total_time": fmt_time(history[-1]["elapsed_sec"]) if history else "0s",
        "epochs_with_warnings": [e["epoch"] for e in history if e["warnings"]],
    }
    if len(history) >= 2:
        v = [e["val"]["oracle_recall"] for e in history]
        summary["oracle_recall_first_last"] = [v[0], v[-1]]
        # Chuỗi epoch liên tiếp mà recall GIẢM — dấu hiệu quá khớp.
        summary["falling_streak"] = sum(
            1 for i in range(len(v) - 1, 0, -1) if v[i] < v[i - 1]) if v[-1] < v[-2] else 0

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump({"summary": summary, "environment": env, "config": cfg,
                   "dataset": {"train": ds_train.stats(), "val": ds_val.stats()},
                   "epochs": history}, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def run_val(model, loader, crit, n_prop, dev):
    """Validation. Trả dict thống kê, gồm đường cong theo tầng.

    `no_grad` nằm ở ĐÂY. Vòng 1 từng mất decorator này khi chèn một hàm mới ngay phía
    trên: decorator đi theo hàm mới, còn vòng val âm thầm dựng đồ thị rồi vỡ ở `.numpy()`.
    """
    model.eval()
    agg, layer_rec, layer_iou, scores, n_batch = {}, [], [], [], 0
    gen = torch.Generator(device=dev.type).manual_seed(0)
    for batch in loader:
        targets = [b.to(dev) for b in batch["boxes"]]
        labels = [b.to(dev) for b in batch["labels"]]
        x_t, t, _ = model.build_inputs(targets, n_prop, batch["valid_h"], generator=gen)
        vh = torch.as_tensor(batch["valid_h"], dtype=torch.float32, device=dev)
        layers = model(x_t, t, valid_h=vh, **model_inputs(batch, dev))
        _, st, _, logits = loss_from_layers(crit, layers, targets, labels)

        layer_rec.append(per_layer_recall(layers, targets))
        layer_iou.append(st["iou_matched_per_layer"])
        scores.append(logits.detach().float().sigmoid().cpu().numpy().ravel())
        for k, v in st.items():
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0.0) + v
        n_batch += 1

    out = {k: v / max(n_batch, 1) for k, v in agg.items()}
    out["oracle_recall_per_layer"] = np.mean(layer_rec, axis=0).tolist()
    out["iou_per_layer"] = np.mean(layer_iou, axis=0).tolist()
    out["oracle_recall"] = out["oracle_recall_per_layer"][-1]
    out["score"] = array_stats(np.concatenate(scores)) if scores else {}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--save-dir", default=None,
                    help="mặc định lấy từ config; để trong repo, .gitignore đã có")
    ap.add_argument("--cache", default="../../data/cache_clip",
                    help="thư mục cache patch token; --cache '' để chạy CLIP mỗi batch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="thu nhỏ để chạy thử")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every-n-batch", type=int, default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    tr_cfg, d_cfg = cfg["training"], cfg["diffusion"]
    epochs = a.epochs or tr_cfg["epochs"]
    bs = a.batch_size or tr_cfg["batch_size"]
    n_train, n_eval = d_cfg["num_proposals_train"], d_cfg["num_proposals_eval"]
    save_dir = a.save_dir or tr_cfg.get("save_dir", "checkpoints/round2_a")

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(tr_cfg["seed"])
    os.makedirs(save_dir, exist_ok=True)

    exp_name = cfg.get("experiment", "?")
    env = {
        "experiment": exp_name,
        "description": cfg.get("description", ""),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev), "torch": torch.__version__,
        "python": sys.version.split()[0], "hostname": socket.gethostname(),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": " ".join(sys.argv), "cwd": os.getcwd(),
    }
    print("=" * 78, flush=True)
    print(f"  TRAIN — EXPERIMENT {exp_name}", flush=True)
    print("-" * 78, flush=True)
    for k, v in env.items():
        print(f"  {k:22s} {v}", flush=True)
    print("=" * 78, flush=True)

    if a.cache:
        for sp in ("train", "val"):
            meta = os.path.join(a.cache, f"{sp}_meta.json")
            if not os.path.exists(meta):
                raise SystemExit(
                    f"\nKHÔNG THẤY CACHE cho split '{sp}': {meta}\n\n"
                    f"Sinh bằng:\n"
                    f"  LOG=/mnt/disk1/aiotlab/haitn/log/cache_{sp}_$(date +%m%d_%H%M).log\n"
                    f"  nohup python tools/run_on_free_gpu.py -- tools/build_cache.py \\\n"
                    f"      --config {a.config} --split {sp} --out {a.cache} \\\n"
                    f"      > $LOG 2>&1 &\n"
                    f"  echo \"PID $! -> $LOG\"\n")

    t_boot = time.time()
    ds_tr = build_dataset(cfg, "train")
    ds_va = build_dataset(cfg, "val")          # flip_prob 0.0: không tăng cường lúc eval
    if a.limit:
        ds_tr.items = ds_tr.items[: a.limit]
        ds_va.items = ds_va.items[: max(a.limit // 2, 1)]
    print(f"[train] {ds_tr.stats()}", flush=True)
    print(f"[val  ] {ds_va.stats()}", flush=True)

    nw = cfg["data"]["num_workers"]
    dl_kw = dict(num_workers=nw, collate_fn=collate, pin_memory=dev.type == "cuda",
                 persistent_workers=nw > 0)
    ld_tr = DataLoader(TorchWrap(ds_tr, PatchCache(a.cache, "train") if a.cache else None),
                       batch_size=bs, shuffle=True, drop_last=True, **dl_kw)
    ld_va = DataLoader(TorchWrap(ds_va, PatchCache(a.cache, "val") if a.cache else None),
                       batch_size=bs, shuffle=False, **dl_kw)

    print(f"[boot ] dataset + loader: {fmt_time(time.time()-t_boot)}", flush=True)
    t_model = time.time()
    model = build_model(cfg).to(dev)
    # CLIP tải về/khởi tạo có thể mất hàng phút mà không in gì — mốc này để biết
    # tiến trình còn sống chứ không phải treo.
    print(f"[boot ] dựng model (tải CLIP): {fmt_time(time.time()-t_model)}", flush=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] tham số học được {sum(p.numel() for p in trainable)/1e6:.2f}M "
          f"/ tổng {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
    print(f"[model] N train/eval={n_train}/{n_eval} | {cfg['model']['n_layer']} tầng "
          f"cộng dồn | RoI {cfg['model']['roi_k']}x{cfg['model']['roi_k']} MỖI tầng",
          flush=True)

    m_cfg = cfg["matcher"]
    crit = SetCriterion(m_cfg["method"],
                        **({"use_center_prior": m_cfg["use_center_prior"],
                            "radius_ratio": m_cfg["center_radius"],
                            "top_k": m_cfg.get("top_k", 10)}
                           if m_cfg["method"] == "simota" else {}))
    opt = torch.optim.AdamW(trainable, lr=float(tr_cfg["lr"]),
                            weight_decay=float(tr_cfg["weight_decay"]))
    # Loss CỘNG các tầng -> gradient lớn hơn vòng 1 6 lần. Nếu loss phân kỳ thì HẠ lr
    # xuống 5e-5, ĐỪNG quay lại chia trung bình.
    print(f"[loss ] matcher={m_cfg['method']} | lr={tr_cfg['lr']} "
          f"(loss CỘNG {cfg['model']['n_layer']} tầng)", flush=True)

    if a.log_every_n_batch is None:
        a.log_every_n_batch = max(len(ld_tr) // 5, 1) if len(ld_tr) >= 10 else 0

    history, best, prev_labels = [], None, {}
    gen = torch.Generator(device=dev.type).manual_seed(tr_cfg["seed"])
    t_start = time.time()
    print(f"[boot ] tổng khởi động {fmt_time(t_start-t_boot)} — bắt đầu epoch 0 "
          f"({len(ld_tr)} batch/epoch)", flush=True)

    for ep in range(epochs):
        model.train()
        t0, run, n_seen, labels_now, grad_norms = time.time(), {}, 0, {}, []

        for bi, batch in enumerate(ld_tr):
            targets = [b.to(dev) for b in batch["boxes"]]
            labels = [b.to(dev) for b in batch["labels"]]
            x_t, t, _ = model.build_inputs(targets, n_train, batch["valid_h"],
                                           generator=gen)
            vh = torch.as_tensor(batch["valid_h"], dtype=torch.float32, device=dev)
            layers = model(x_t, t, valid_h=vh, **model_inputs(batch, dev))
            loss, st, idx, _ = loss_from_layers(crit, layers, targets, labels)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            grad_norms.append(float(gn))
            opt.step()

            for im, (pi, gi) in zip(batch["image_id"], idx):
                for p, g in zip(pi.tolist(), gi.tolist()):
                    labels_now[(im, p)] = g
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    run[k] = run.get(k, 0.0) + v
            n_seen += 1

            if a.log_every_n_batch and bi % a.log_every_n_batch == 0:
                print(f"  ep{ep:3d} b{bi:4d}/{len(ld_tr)} loss {float(loss):8.3f} "
                      f"(mean {st['loss_mean']:6.3f}) iou {st['iou_matched']:.4f}",
                      flush=True)

        tr_stats = {k: v / max(n_seen, 1) for k, v in run.items()}
        stability = label_stability(prev_labels, labels_now)
        prev_labels = labels_now
        va_stats = run_val(model, ld_va, crit, n_eval, dev)
        rec = va_stats["oracle_recall_per_layer"]

        warnings = []
        if rec[-1] <= rec[0] + 1e-4:
            warnings.append("recall KHÔNG tăng qua các tầng -> cộng dồn chưa mang lại gì")
        if not np.isnan(stability) and stability < 0.005:
            warnings.append(f"label_stability {stability:.3f} dưới sàn ngẫu nhiên")
        if va_stats.get("score", {}).get("std", 1.0) < 0.05:
            warnings.append("std_score < 0.05 — head score có thể đang kẹt ở hằng số")
        if grad_norms and np.mean(grad_norms) > 100:
            warnings.append(f"grad norm {np.mean(grad_norms):.1f} rất lớn — cân nhắc hạ lr")

        eta = (time.time() - t_start) / (ep + 1) * (epochs - ep - 1)
        print(f"[ep {ep:3d}] train {tr_stats['loss']:8.3f} | val {va_stats['loss']:8.3f} "
              f"| oracle_recall {va_stats['oracle_recall']:.4f} "
              f"| stab {stability:.3f} | {fmt_time(time.time()-t0)} | ETA {fmt_time(eta)}",
              flush=True)
        print(f"          recall/tầng: {' '.join(f'{v:.3f}' for v in rec)}", flush=True)
        for w in warnings:
            print(f"          ⚠️  {w}", flush=True)

        history.append({"epoch": ep, "train": tr_stats, "val": va_stats,
                        "label_stability": stability,
                        "grad_norm": array_stats(grad_norms),
                        "warnings": warnings,
                        "epoch_sec": time.time() - t0,
                        "elapsed_sec": time.time() - t_start})
        write_json(save_dir, env, cfg, history, ds_tr, ds_va)

        # CHỌN CHECKPOINT BẰNG oracle_recall, KHÔNG BAO GIỜ bằng iou_matched hay loss:
        # matcher không nhìn thấy oracle_recall nên nó không bị đánh lừa như hai lần ở
        # vòng 1 (C1 và E1).
        if best is None or va_stats["oracle_recall"] > best["oracle_recall"]:
            best = {"epoch": ep, "oracle_recall": va_stats["oracle_recall"],
                    "loss": va_stats["loss"]}
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep,
                        "best": best}, os.path.join(save_dir, "best.pt"))

    print(f"[done] {fmt_time(time.time()-t_start)} | best {best}", flush=True)


if __name__ == "__main__":
    main()
