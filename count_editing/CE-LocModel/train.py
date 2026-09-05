#!/usr/bin/env python3
"""Train CE-Loc vòng 2.

BA CHỈ SỐ THEO DÕI (quan trọng ngang loss — vòng 1 thiếu nên mù suốt 5 vòng sửa):

  1. % cặp match GIỮ NGUYÊN giữa 2 epoch — vòng luẩn quẩn score<->toạ độ đã phá
     được chưa. Vòng 1 chỉ ~55 %, tức hơn nửa nhãn đổi mỗi epoch nên score head
     không bao giờ học được.
  2. std của sigmoid(score) — < 0,05 nghĩa là head kẹt ở hằng số (focal alpha=0,25
     với head không phân biệt hội tụ về một giá trị cố định).
  3. IoU trung bình của cặp matched — tách khỏi loss nên dễ đọc.

Chạy nền, ghi log ra file (trên server: /mnt/disk1/aiotlab/haitn/log/):
  nohup python3 train.py --config config/experiment_a.yaml > <log> 2>&1 & echo $!
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

from data.ce130_dataset import CE130Detection, PatchCache, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402


class TorchWrap(Dataset):
    """Bọc CE130Detection (numpy) thành torch Dataset.

    Có `cache` thì trả patch/text token đã tính sẵn và BỎ HẲN ảnh — đo trên A30:
    CLIP chiếm 76,8 % thời gian mỗi batch, cache cho ~4,3x tốc độ.
    """

    def __init__(self, ds, cache=None):
        self.ds = ds
        self.cache = cache

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        m = self.ds[i]
        ra = {
            "boxes": torch.from_numpy(m["boxes"]).float(),
            "text": m["text"],
            "valid_h": m["valid_h"],
            "image_id": m["image_id"],
        }
        if self.cache is None:
            ra["pixel_values"] = torch.from_numpy(normalize_for_clip(m["image"]))
        else:
            patch, text = self.cache.lay(m["image_id"], m["text"], m["flipped"])
            ra["patch_raw"] = torch.from_numpy(patch)
            ra["text_raw"] = torch.from_numpy(text)
        return ra


def collate(batch):
    """Số box khác nhau mỗi ảnh -> giữ dạng list, không pad ở đây."""
    ra = {
        "boxes": [b["boxes"] for b in batch],
        "text": [b["text"] for b in batch],
        "valid_h": [b["valid_h"] for b in batch],
        "image_id": [b["image_id"] for b in batch],
    }
    for k in ("pixel_values", "patch_raw", "text_raw"):
        if k in batch[0]:
            ra[k] = torch.stack([b[k] for b in batch])
    return ra


def dinh_dang_tg(giay):
    """3661 -> '1h01m01s'. Dùng cho cả thời gian đã chạy lẫn ETA."""
    giay = int(max(giay, 0))
    h, m, s = giay // 3600, (giay % 3600) // 60, giay % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def thong_ke_mang(x):
    """Phân bố đầy đủ của một mảng — để đọc lại được khi có vấn đề."""
    if x is None or len(x) == 0:
        return {}
    x = np.asarray(x, dtype=np.float64)
    q = np.percentile(x, [1, 25, 50, 75, 99])
    return {"mean": float(x.mean()), "std": float(x.std()),
            "min": float(x.min()), "max": float(x.max()),
            "p1": float(q[0]), "p25": float(q[1]), "p50": float(q[2]),
            "p75": float(q[3]), "p99": float(q[4])}


def do_on_dinh_nhan(truoc, sau):
    """% cặp (image_id, pred_idx) -> gt_idx giữ nguyên giữa 2 epoch."""
    if not truoc:
        return float("nan")
    chung = set(truoc) & set(sau)
    if not chung:
        return 0.0
    return sum(truoc[k] == sau[k] for k in chung) / len(chung)


def ghi_json(save_dir, moi_truong, cfg, lich_su, best, ds_train, ds_val):
    """Ghi TOÀN BỘ số liệu ra history.json sau MỖI epoch.

    Ghi mỗi epoch (không đợi train xong) để nếu job chết giữa chừng vẫn đọc được.
    Chứa đủ thứ để chẩn đoán mà không cần chạy lại: môi trường, config đầy đủ,
    thống kê dataset, và mọi chỉ số per-epoch kèm phân bố (mean/std/min/max/
    percentile) chứ không chỉ giá trị trung bình.
    """
    # Tính best TỪ lich_su, không dùng biến `best` truyền vào — hàm này được gọi
    # TRƯỚC khi vòng train cập nhật `best`, nên dùng nó sẽ lệch một epoch.
    tot = min(lich_su, key=lambda e: e["val"]["loss"]) if lich_su else None
    tom_tat = {
        "so_epoch_da_chay": len(lich_su),
        "best_val_loss": tot["val"]["loss"] if tot else None,
        "best_epoch": tot["epoch"] if tot else None,
        "tong_thoi_gian": dinh_dang_tg(lich_su[-1]["da_chay_giay"]) if lich_su else "0s",
        "epoch_co_canh_bao": [e["epoch"] for e in lich_su if e["canh_bao"]],
    }
    if len(lich_su) >= 2:
        v = [e["val"]["loss"] for e in lich_su]
        tom_tat["val_loss_dau_cuoi"] = [v[0], v[-1]]
        tom_tat["val_tang_lien_tiep"] = sum(
            1 for i in range(len(v) - 1, 0, -1) if v[i] > v[i - 1]) if v[-1] > v[-2] else 0

    with open(os.path.join(save_dir, "history.json"), "w") as f:
        json.dump({
            "tom_tat": tom_tat,
            "moi_truong": moi_truong,
            "config": cfg,
            "dataset": {"train": ds_train.thong_ke(), "val": ds_val.thong_ke()},
            "epochs": lich_su,
        }, f, indent=2, ensure_ascii=False)


def dau_vao_model(batch, dev):
    """Trả kwargs cho model: ảnh thô, hoặc token đã cache."""
    if "patch_raw" in batch:
        return {"patch_raw": batch["patch_raw"].to(dev),
                "text_raw": batch["text_raw"].to(dev)}
    return {"pixel_values": batch["pixel_values"].to(dev), "texts": batch["text"]}


@torch.no_grad()
def chay_val(model, loader, crit, N, dev):
    """Val loss — với 1.911 ảnh và class 3 split RỜI NHAU, không có val thì không
    biết khi nào bắt đầu overfit. Dùng seed cố định cho `t` để so được giữa các
    epoch (nếu t ngẫu nhiên thì val loss nhiễu, không đọc được xu hướng)."""
    model.eval()
    khoa = ["loss", "loss_l1", "loss_giou", "loss_ce", "iou_matched", "n_matched"]
    tong, nb, diem = {k: 0.0 for k in khoa}, 0, []
    t0 = time.time()
    g = torch.Generator().manual_seed(1234)
    for batch in loader:
        tg = [b.to(dev) for b in batch["boxes"]]
        x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"], generator=g)
        pb, lg = model(x_t, tt, **dau_vao_model(batch, dev))
        _, st, _ = crit(pb, lg, tg)
        for k in khoa:
            tong[k] += st[k]
        diem.append(lg.sigmoid().cpu().numpy().ravel())
        nb += 1
    model.train()
    kq = {k: v / max(nb, 1) for k, v in tong.items()}
    kq["score"] = thong_ke_mang(np.concatenate(diem)) if diem else {}
    kq["giay"] = time.time() - t0
    return kq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cắt nhỏ dataset để thử")
    ap.add_argument("--device", default=None)
    ap.add_argument("--cache", default=None,
                    help="thư mục cache patch token (tools/build_cache.py). Bật thì "
                         "bỏ hẳn CLIP forward lúc train — đo được ~4,3x nhanh hơn.")
    ap.add_argument("--log-moi-n-batch", type=int, default=None,
                    help="in tiến độ mỗi N batch; mặc định tự chia 5 lần/epoch")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    if a.epochs:
        cfg["training"]["epochs"] = a.epochs
    if a.batch_size:
        cfg["training"]["batch_size"] = a.batch_size

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["training"]["seed"])

    ten_tn = cfg.get("experiment", "?")
    moi_truong = {
        "experiment": ten_tn,
        "mo_ta": cfg.get("mo_ta", ""),
        "thoi_diem_bat_dau": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "hostname": socket.gethostname(),
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "lenh": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }
    print("=" * 78, flush=True)
    print(f"  TRAIN — EXPERIMENT {ten_tn}", flush=True)
    print("-" * 78, flush=True)
    for k, v in moi_truong.items():
        print(f"  {k:22s} {v}", flush=True)
    print("-" * 78, flush=True)
    for muc in ["data", "model", "diffusion", "loss", "matcher", "training"]:
        print(f"  {muc:10s} {json.dumps(cfg[muc], ensure_ascii=False)}", flush=True)
    print("=" * 78, flush=True)

    ds = CE130Detection(cfg["data"]["root"], "train",
                        cfg["data"]["image_size"], cfg["data"]["flip_prob"],
                        seed=cfg["training"]["seed"])
    if a.limit:
        ds.items = ds.items[: a.limit]
    print(f"[data] {ds.thong_ke()}", flush=True)

    cache_tr = cache_va = None
    if a.cache:
        cache_tr = PatchCache(a.cache, "train")
        cache_va = PatchCache(a.cache, "val")
        print(f"[cache] dùng {a.cache} — bỏ CLIP forward lúc train "
              f"({cache_tr.n_ver} bản/ảnh)", flush=True)

    loader = DataLoader(TorchWrap(ds, cache_tr), batch_size=cfg["training"]["batch_size"],
                        shuffle=True, num_workers=cfg["data"]["num_workers"],
                        collate_fn=collate, drop_last=False)

    # val: KHÔNG flip (không augment lúc đánh giá)
    ds_val = CE130Detection(cfg["data"]["root"], "val", cfg["data"]["image_size"])
    if a.limit:
        ds_val.items = ds_val.items[: max(a.limit // 2, 1)]
    val_loader = DataLoader(TorchWrap(ds_val, cache_va), batch_size=cfg["training"]["batch_size"],
                            shuffle=False, num_workers=cfg["data"]["num_workers"],
                            collate_fn=collate)
    print(f"[val ] {ds_val.thong_ke()}", flush=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], cfg["model"]["dropout"],
        cfg["model"]["freeze_clip"],
    ).to(dev)

    hoc_duoc = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] tham số học được: {sum(p.numel() for p in hoc_duoc)/1e6:.2f}M "
          f"/ tổng {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    crit = SetCriterion(cfg["matcher"]["method"],
                        **({"use_center_prior": cfg["matcher"]["use_center_prior"],
                            "radius_ratio": cfg["matcher"]["center_radius"]}
                           if cfg["matcher"]["method"] == "simota" else {}))
    opt = torch.optim.AdamW(hoc_duoc, lr=float(cfg["training"]["lr"]),
                            weight_decay=float(cfg["training"]["weight_decay"]))

    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    N = cfg["diffusion"]["num_proposals_train"]
    if a.log_moi_n_batch is None:
        a.log_moi_n_batch = max(len(loader) // 5, 1) if len(loader) >= 10 else 0
    best, lich_su, nhan_truoc = float("inf"), [], {}
    t_bat_dau = time.time()

    for ep in range(cfg["training"]["epochs"]):
        model.train()
        t0 = time.time()
        tong = {"loss": 0.0, "loss_l1": 0.0, "loss_giou": 0.0, "loss_ce": 0.0,
                "iou_matched": 0.0, "n_matched": 0}
        nhan_nay, diem, nb = {}, [], 0
        grad_norms, so_gt, t_batch = [], [], []

        for batch in loader:
            t_b = time.time()
            tg = [b.to(dev) for b in batch["boxes"]]

            x_t, tt, _ = model.build_inputs(tg, N, batch["valid_h"])
            pb, lg = model(x_t, tt, **dau_vao_model(batch, dev))
            loss, st, idx = crit(pb, lg, tg)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(hoc_duoc, 1.0)
            opt.step()

            for k in tong:
                tong[k] += st[k]
            nb += 1
            grad_norms.append(float(gn))
            so_gt += [len(b) for b in batch["boxes"]]
            t_batch.append(time.time() - t_b)
            diem.append(lg.detach().sigmoid().cpu().numpy().ravel())

            # Tiến độ trong epoch — với 1.911 ảnh mỗi epoch mất nhiều phút, không
            # nên im lặng suốt. In 5 lần/epoch (không dùng tqdm để log đọc được
            # trong file).
            if a.log_moi_n_batch and nb % a.log_moi_n_batch == 0:
                da = time.time() - t0
                print(f"      ... batch {nb}/{len(loader)} | loss {st['loss']:.4f} | "
                      f"{1000*da/nb:.0f}ms/batch | {dinh_dang_tg(da)} | "
                      f"còn {dinh_dang_tg(da/nb*(len(loader)-nb))}", flush=True)
            for i, (pi, gi) in enumerate(idx):                 # chỉ số 1
                iid = batch["image_id"][i]
                for p, g in zip(pi.tolist(), gi.tolist()):
                    nhan_nay[(iid, p)] = g

        tb = {k: v / max(nb, 1) for k, v in tong.items()}
        on_dinh = do_on_dinh_nhan(nhan_truoc, nhan_nay)
        nhan_truoc = nhan_nay
        tk_score = thong_ke_mang(np.concatenate(diem)) if diem else {}
        std_score = tk_score.get("std", 0.0)
        giay_train = time.time() - t0

        val = chay_val(model, val_loader, crit, N, dev)
        giay_ep = time.time() - t0
        da_chay = time.time() - t_bat_dau
        con_lai = cfg["training"]["epochs"] - (ep + 1)
        eta = (da_chay / (ep + 1)) * con_lai

        # --- LOG: 3 dòng cố định mỗi epoch, in đủ thứ đọc được ---
        print(f"[ep {ep+1:4d}/{cfg['training']['epochs']}] "
              f"train {tb['loss']:8.4f} (l1 {tb['loss_l1']:.4f} giou {tb['loss_giou']:.4f} "
              f"ce {tb['loss_ce']:.4f})   val {val['loss']:8.4f} "
              f"(l1 {val['loss_l1']:.4f} giou {val['loss_giou']:.4f} ce {val['loss_ce']:.4f})",
              flush=True)
        print(f"           IoU train {tb['iou_matched']:.4f} / val {val['iou_matched']:.4f} | "
              f"matched {tb['n_matched']:.1f}/{N} ({100*tb['n_matched']/N:.0f}%) | "
              f"GT/ảnh {np.mean(so_gt):.1f} | ổn_định_nhãn {on_dinh:.3f} | "
              f"lr {opt.param_groups[0]['lr']:.2e} | grad {np.mean(grad_norms):.3f}",
              flush=True)
        print(f"           score μ {tk_score.get('mean', 0):.4f} σ {std_score:.4f} "
              f"[{tk_score.get('min', 0):.3f}, {tk_score.get('max', 0):.3f}] "
              f"p50 {tk_score.get('p50', 0):.4f} | "
              f"{dinh_dang_tg(giay_train)}+{dinh_dang_tg(val['giay'])} "
              f"({1000*np.mean(t_batch):.0f}ms/batch) | "
              f"đã chạy {dinh_dang_tg(da_chay)} | ETA {dinh_dang_tg(eta)}", flush=True)

        canh_bao = []
        if std_score < 0.05:
            canh_bao.append("std_score < 0,05 — score head có thể kẹt ở hằng số")
        if not np.isnan(on_dinh) and on_dinh < 0.4:
            canh_bao.append(f"ổn_định_nhãn {on_dinh:.2f} < 0,40 — nhãn đổi quá nhiều mỗi epoch")
        if np.mean(grad_norms) > 100:
            canh_bao.append(f"grad norm {np.mean(grad_norms):.1f} rất lớn")
        for c in canh_bao:
            print(f"           ⚠ {c}", flush=True)

        lich_su.append({
            "epoch": ep + 1,
            "train": {**tb, "score": tk_score,
                      "grad_norm": thong_ke_mang(grad_norms),
                      "gt_moi_anh": thong_ke_mang(so_gt),
                      "ms_moi_batch": thong_ke_mang([1000 * x for x in t_batch]),
                      "so_batch": nb, "giay": giay_train},
            "val": val,
            "on_dinh_nhan": on_dinh,
            "lr": opt.param_groups[0]["lr"],
            "giay_epoch": giay_ep,
            "da_chay_giay": da_chay,
            "eta_giay": eta,
            "canh_bao": canh_bao,
            "thoi_diem": datetime.now().isoformat(timespec="seconds"),
        })
        ghi_json(save_dir, moi_truong, cfg, lich_su, best, ds, ds_val)

        # chọn best theo VAL loss, không phải train loss
        if val["loss"] < best:
            best = val["loss"]
            # CHỈ lưu tham số HỌC ĐƯỢC (~8,3M). Lưu cả CLIP frozen thì checkpoint
            # nặng 698 MB trong khi 98 % là trọng số tải lại được từ HuggingFace,
            # và còn buộc phải khớp đúng phiên bản CLIP lúc load.
            hoc_sd = {k: v for k, v in model.state_dict().items()
                      if not (k.startswith("encoder.vision.") or k.startswith("encoder.text."))}
            torch.save({"epoch": ep, "model": hoc_sd, "optimizer": opt.state_dict(),
                        "loss": best, "cfg": cfg, "chi_tham_so_hoc_duoc": True},
                       os.path.join(save_dir, "best.pth"))
            print(f"  -> lưu best (val_loss {best:.4f})", flush=True)

    tong_tg = time.time() - t_bat_dau
    print("=" * 78, flush=True)
    print(f"XONG — EXPERIMENT {ten_tn} — {dinh_dang_tg(tong_tg)} ({len(lich_su)} epoch, "
          f"{dinh_dang_tg(tong_tg/max(len(lich_su),1))}/epoch)", flush=True)
    if lich_su:
        tot = min(lich_su, key=lambda e: e["val"]["loss"])
        print(f"  best: epoch {tot['epoch']} | val_loss {tot['val']['loss']:.4f} | "
              f"val_IoU {tot['val']['iou_matched']:.4f}", flush=True)
        print(f"  val_loss: {lich_su[0]['val']['loss']:.4f} -> "
              f"{lich_su[-1]['val']['loss']:.4f}", flush=True)
        if tot["epoch"] == len(lich_su):
            print("  ⚠ best rơi vào epoch CUỐI — chưa bão hoà, nên train dài hơn "
                  "(vòng 1 gặp đúng thế 4 lần liên tiếp)", flush=True)
        canh = [e["epoch"] for e in lich_su if e["canh_bao"]]
        if canh:
            print(f"  ⚠ {len(canh)}/{len(lich_su)} epoch có cảnh báo: "
                  f"{canh[:10]}{'...' if len(canh) > 10 else ''}", flush=True)
    print(f"  số liệu đầy đủ: {os.path.join(save_dir, 'history.json')}", flush=True)
    print("=" * 78, flush=True)


if __name__ == "__main__":
    main()
