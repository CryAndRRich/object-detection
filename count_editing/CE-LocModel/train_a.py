#!/usr/bin/env python3
"""Train EXPERIMENT A (vòng 2) — xem `docs/EXPERIMENT_A_PLAN.md`.

TÁI DÙNG, KHÔNG CHÉP: `TorchWrap`, `collate`, `model_inputs`, `oracle_recall`,
`label_stability`, `array_stats`, `fmt_time`, `write_json` lấy nguyên từ `train.py`.
Chúng không dính gì tới kiến trúc, và chép lại chỉ tạo ra hai bản dễ lệch nhau.

KHÁC `train.py` ĐÚNG BỐN CHỖ:
  1. dựng `CELocDetectorA` thay `CELocDetector`
  2. `DeepSetCriterion` (CỘNG loss các tầng) thay `SetCriterion` (chia trung bình)
  3. `forward` nhận thêm `valid_h` — box sinh ra không được lấy mẫu trong vùng đệm
  4. log thêm đường cong THEO TẦNG: `oracle_recall` và `iou` của từng tầng.
     Đây là chỉ số CHÍNH để đọc thí nghiệm — đường phẳng nghĩa là cộng dồn không mang
     lại gì, và kết luận đó phải nhìn thấy được trước khi xây thêm bất cứ thứ gì.

CHẠY TRÊN SERVER (job dài -> chạy nền, kèm PID + logfile):
  cd object-detection/count_editing/CE-LocModel
  LOG=/mnt/disk1/aiotlab/haitn/log/round2_a_$(date +%m%d_%H%M).log
  nohup python tools/run_on_free_gpu.py -- train_a.py \
      --config config/round2_experiment_a.yaml \
      --cache /mnt/disk1/aiotlab/haitn/cache \
      --save-dir /mnt/disk1/aiotlab/haitn/checkpoints/round2_a \
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.ce130_dataset import PatchCache
from data.factory import build_dataset
from models.criterion_a import DeepSetCriterion, loss_from_layers
from models.detector_a import build_model_a
from train import (                      # noqa: F401  — dùng lại, không chép
    TorchWrap,
    array_stats,
    collate,
    fmt_time,
    label_stability,
    model_inputs,
    oracle_recall,
)
from utils.box_ops import cxcywh_to_xyxy


def per_layer_quality(layers, targets, dev):
    """`oracle_recall` cho TỪNG tầng — chỉ số chính để đọc EXPERIMENT A.

    Tách riêng khỏi loss vì `iou_matched` MÙ với GT mà không box nào chạm tới, và nó đã
    đánh lừa HAI LẦN ở vòng 1 (C1 và E1): một tầng siết chặt những box nó đã có trong
    khi ĐÁNH MẤT vùng phủ của GT khác sẽ có loss TỐT HƠN mà thực tế TỆ ĐI. Đo trên C1:
    `iou_matched` tăng 0,3426 -> 0,3528 qua sáu vòng còn `oracle_recall` GIẢM
    0,138 -> 0,133.
    """
    out = []
    for boxes, _ in layers:
        hit = tot = 0
        for b, gt in zip(boxes, targets):
            h, n = oracle_recall(b.detach().cpu(), gt.cpu())
            hit += h
            tot += n
        out.append(hit / max(tot, 1))
    return out


@torch.no_grad()
def run_val(model, loader, crit, n_prop, dev):
    """Validation. Trả về dict thống kê, gồm đường cong theo tầng.

    `no_grad` nằm ở ĐÂY. Vòng 1 từng mất decorator này khi chèn một hàm mới ngay phía
    trên: decorator đi theo hàm mới, còn vòng val âm thầm dựng đồ thị rồi vỡ ở `.numpy()`.
    """
    model.eval()
    agg, layer_rec, layer_iou, n_batch = {}, [], [], 0
    gen = torch.Generator(device=dev.type).manual_seed(0)
    for batch in loader:
        targets = [b.to(dev) for b in batch["boxes"]]
        labels = [b.to(dev) for b in batch["labels"]]
        x_t, t, _ = model.build_inputs(targets, n_prop, batch["valid_h"], generator=gen)
        vh = torch.as_tensor(batch["valid_h"], dtype=torch.float32, device=dev)
        layers = model(x_t, t, valid_h=vh, **model_inputs(batch, dev))
        _, st, _, _ = loss_from_layers(crit, layers, targets, labels)

        layer_rec.append(per_layer_quality(layers, targets, dev))
        layer_iou.append(st["iou_matched_per_layer"])
        for k, v in st.items():
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0.0) + v
        n_batch += 1

    out = {k: v / max(n_batch, 1) for k, v in agg.items()}
    out["oracle_recall_per_layer"] = np.mean(layer_rec, axis=0).tolist()
    out["iou_per_layer"] = np.mean(layer_iou, axis=0).tolist()
    out["oracle_recall"] = out["oracle_recall_per_layer"][-1]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/round2_experiment_a.yaml")
    ap.add_argument("--save-dir", required=True)
    ap.add_argument("--cache", default=None,
                    help="thư mục cache patch token; không có thì chạy CLIP mỗi batch")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="thu nhỏ để chạy thử")
    ap.add_argument("--device", default=None)
    ap.add_argument("--log-every-n-batch", type=int, default=50)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    tr_cfg, d_cfg = cfg["training"], cfg["diffusion"]
    epochs = a.epochs or tr_cfg["epochs"]
    bs = a.batch_size or tr_cfg["batch_size"]
    n_train = d_cfg["num_proposals_train"]
    n_eval = d_cfg["num_proposals_eval"]

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(tr_cfg["seed"])
    os.makedirs(a.save_dir, exist_ok=True)

    ds_tr = build_dataset(cfg, "train")
    ds_va = build_dataset(cfg, "val")
    if a.limit:
        ds_tr.items = ds_tr.items[: a.limit]
        ds_va.items = ds_va.items[: max(a.limit // 4, 1)]
    cache_tr = PatchCache(a.cache, "train") if a.cache else None
    cache_va = PatchCache(a.cache, "val") if a.cache else None

    nw = cfg["data"]["num_workers"]
    ld_tr = DataLoader(TorchWrap(ds_tr, cache_tr), batch_size=bs, shuffle=True,
                       num_workers=nw, collate_fn=collate, pin_memory=dev.type == "cuda",
                       drop_last=True, persistent_workers=nw > 0)
    ld_va = DataLoader(TorchWrap(ds_va, cache_va), batch_size=bs, shuffle=False,
                       num_workers=nw, collate_fn=collate, pin_memory=dev.type == "cuda",
                       persistent_workers=nw > 0)

    model = build_model_a(cfg).to(dev)
    m_cfg = cfg["matcher"]
    crit = DeepSetCriterion(
        matcher_method=m_cfg["method"],
        **({"use_center_prior": m_cfg["use_center_prior"],
            "radius_ratio": m_cfg["center_radius"],
            "top_k": m_cfg["top_k"]} if m_cfg["method"] == "simota" else {}))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tr_cfg["lr"], weight_decay=tr_cfg["weight_decay"])

    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[round2-A] thiết bị={dev} | tham số học được={n_par/1e6:.2f}M | "
          f"N train/eval={n_train}/{n_eval} | matcher={m_cfg['method']} | "
          f"batch={bs} | epoch={epochs}", flush=True)
    print(f"[round2-A] train={len(ds_tr)} ảnh, val={len(ds_va)} ảnh, "
          f"cache={'có' if a.cache else 'KHÔNG'}", flush=True)
    # Loss CỘNG 6 tầng -> gradient lớn hơn vòng 1 6 lần. Nếu loss phân kỳ thì HẠ lr
    # xuống 5e-5, ĐỪNG quay lại chia trung bình.
    print(f"[round2-A] lr={tr_cfg['lr']} (loss CỘNG {cfg['model']['n_layer']} tầng)",
          flush=True)

    history, best, prev_labels = [], None, {}
    gen = torch.Generator(device=dev.type).manual_seed(tr_cfg["seed"])
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        ep_t0, run, n_seen, labels_now = time.time(), {}, 0, {}
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr_cfg["grad_clip"])
            opt.step()

            # Độ ổn định nhãn: % cặp (ảnh, box) -> gt giữ nguyên giữa hai epoch.
            # Vòng 1 đo được 0,018 (>98 % đổi mỗi epoch, suốt 299 epoch) — biến nền
            # quan trọng nhất, và là lý do vòng 2 đổi sang SimOTA.
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
        print(f"[ep {ep:3d}] train {tr_stats['loss']:8.3f} | val {va_stats['loss']:8.3f} "
              f"| oracle_recall {va_stats['oracle_recall']:.4f} "
              f"| label_stability {stability:.3f} | {fmt_time(time.time()-ep_t0)}",
              flush=True)
        print(f"          recall theo tầng: "
              f"{' '.join(f'{v:.3f}' for v in rec)}", flush=True)

        # CẢNH BÁO, không tự dừng: người dùng quyết định có huỷ job hay không.
        if rec[-1] <= rec[0] + 1e-4:
            print("          ⚠️  recall KHÔNG tăng qua các tầng -> cộng dồn chưa mang "
                  "lại gì. Nếu kéo dài nhiều epoch thì cân nhắc đối chứng 'gốc cố định'.",
                  flush=True)
        if not np.isnan(stability) and stability < 0.005:
            print("          ⚠️  label_stability dưới sàn ngẫu nhiên -> nhãn gần như "
                  "vô nghĩa; cân nhắc denoising query.", flush=True)

        history.append({"epoch": ep, "train": tr_stats, "val": va_stats,
                        "label_stability": stability,
                        "time_sec": time.time() - ep_t0})

        # CHỌN CHECKPOINT BẰNG oracle_recall, KHÔNG BAO GIỜ bằng iou_matched hay loss:
        # matcher không nhìn thấy oracle_recall, nên nó không bị đánh lừa như hai lần ở
        # vòng 1.
        score = va_stats["oracle_recall"]
        if best is None or score > best["oracle_recall"]:
            best = {"epoch": ep, "oracle_recall": score, "loss": va_stats["loss"]}
            torch.save({"model": model.state_dict(), "config": cfg, "epoch": ep,
                        "best": best}, os.path.join(a.save_dir, "best.pt"))

        # Ghi history MỖI epoch, không đợi đến cuối: job chết giữa chừng vẫn đọc được.
        with open(os.path.join(a.save_dir, "history.json"), "w") as f:
            json.dump({"config": cfg, "best": best, "epochs": history}, f,
                      indent=2, ensure_ascii=False)

    print(f"[round2-A] xong sau {fmt_time(time.time()-t0)} | best {best}", flush=True)


if __name__ == "__main__":
    main()
