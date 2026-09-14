#!/usr/bin/env python3
"""CỬA CHẶN cho hướng "Conv1x1 + SpatialSoftmax → K toạ độ làm memory".

CÂU HỎI DUY NHẤT
----------------
`Conv1x1(768 → K)` train trên CLIP ViT-B/16 **FROZEN** có sinh ra K bản đồ mà **mỗi bản đồ
có một đỉnh nằm đúng trên một vật** không?

Nếu KHÔNG thì K toạ độ là rác, và toàn bộ thiết kế (memory = K token toạ độ, cross-attention
với box token, cộng bias 4 góc kiểu BoxRPB) là vô nghĩa — hỏng y hệt cách soft-argmax cục bộ
đã hỏng (docs/01-bai-toan.md mục 6).

VÌ SAO ĐO ĐƯỢC MÀ KHÔNG CẦN VIẾT MODEL
--------------------------------------
Giả thuyết nằm trọn trong 2 lớp: Conv1x1 + SpatialSoftmax (~49K tham số với K=64). Không cần
decoder, không cần diffusion, không cần box token. Train riêng 2 lớp đó với loss hình học
thuần, rồi đo bằng ĐÚNG chỉ số đã giết ý tưởng trước.

BASELINE ĐÃ CÓ (docs/01-bai-toan.md mục 6)
------------------------------------------
CLIP cosine frozen (không train): đỉnh lệch tâm GT **trung vị 3,60 ô lưới**, chỉ **11,7 %**
trong vòng 1 ô — trong khi box CE-130 chỉ rộng **2,4–4,4 ô** (lưới 32x32).

NGƯỠNG CHỐT TRƯỚC KHI CHẠY (không được đọc kết quả rồi mới đặt ngưỡng)
---------------------------------------------------------------------
  ĐIỀU KIỆN TIÊN QUYẾT : số điểm PHÂN BIỆT >= K/2. Không đạt -> "KHÔNG ĐỌC ĐƯỢC",
                         mọi chỉ số offset vô nghĩa (xem dưới).
  ĐẠT       : median_offset < 1,5 ô  VÀ  pct_within_1cell > 40 %
  KHÔNG ĐẠT : median_offset > 2,5 ô  (≈ không khá hơn baseline một cách có ý nghĩa)
  XÁM       : ở giữa — cần bàn, không tự quyết

⚠️ LẦN CHẠY ĐẦU (2026-09-14) HỎNG VÌ LOSS SAI, ghi lại để không lặp:
   Chamfer hai chiều không phạt việc các điểm TRÙNG NHAU -> nghiệm tối ưu là "cả 64 điểm
   đứng cùng một chỗ", và model tìm đúng nghiệm đó (spread 0,28 ô ~ 4,5 px). Khi đó
   median_offset 1,37 ô trông như gần ĐẠT, nhưng nó chỉ đo "một điểm cách vật gần nhất bao
   xa" -- mà CE-130 có 20-30 vật/ảnh nên chỗ nào cũng gần một vật. Đã đổi sang Hungarian
   (ghép 1-1, hai điểm không thể cùng nhận một GT) + repulsion hinge, và thêm cột "#rõ"
   (số điểm phân biệt) làm điều kiện tiên quyết.

⚠️ Đo trên tập VAL (mặc định) — 28 class CHƯA HỀ THẤY lúc train, giao với 72 class train
   = 0, nên trả lời câu hỏi zero-shot y hệt test. Cache của EXPERIMENT A chỉ dựng train+val.
   Cột `train` chỉ để biết model có học được gì không; cột eval mới là cột quyết định.
   Chênh lệch train↔test lớn = Conv1x1 học thuộc 72 class train, MẤT zero-shot (rủi ro R4).

⚠️ CHỈ SỐ NÀY ĐO ĐIỂM, KHÔNG ĐO VÙNG. Bài học docs/01-bai-toan.md mục 6.1: AUC 0,782 của
   check_exemplar_signal.py chứng minh phân biệt VÙNG, KHÔNG chứng minh định vị ĐIỂM. Đừng
   lặp lại lỗi đó — ở đây đo thẳng khoảng cách điểm↔tâm GT.

CHẠY (TRÊN SERVER, không chạy ở local)
--------------------------------------
  python tools/check_keypoint_head.py --cache <thư-mục-cache> --epochs 30 --K 64
  (mặc định --eval-split val, vì cache chỉ có train+val)

  ⚠️ PHẢI truyền --cache: cache patch token fp16 đã dựng sẵn từ EXPERIMENT A
     (tools/build_cache.py). Đó là lý do train chỉ mất ~2 giờ. Không truyền thì tool chạy
     lại CLIP từ đầu — lãng phí thuần và rủi ro lệch tiền xử lý so với lúc train.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, PatchCache, normalize_for_clip  # noqa: E402
from models.detector import build_model  # noqa: E402


# --------------------------------------------------------------------------- model


class KeypointHead(nn.Module):
    """Conv1x1(768 → K) + SpatialSoftmax. Đây là TOÀN BỘ phần học được của giả thuyết.

    SpatialSoftmax giữ nguyên công thức của CE-Loc gốc
    (refs/repos/Count-Editing/CE-LocModel/models/spatial_softmax.py): softmax trên H*W riêng
    từng kênh, rồi lấy KỲ VỌNG với lưới toạ độ. Đầu ra là TOẠ ĐỘ, không phải embedding.
    """

    def __init__(self, d_in=768, K=64, temperature=1.0):
        super().__init__()
        self.conv = nn.Conv2d(d_in, K, kernel_size=1)
        self.K = K
        self.temperature = temperature

    def forward(self, patch_raw, grid):
        """patch_raw [B, grid*grid, 768] -> pts [B, K, 2] trong [0,1], peak [B, K]."""
        B = patch_raw.shape[0]
        x = patch_raw.transpose(1, 2).reshape(B, -1, grid, grid)   # [B,768,g,g]
        maps = self.conv(x)                                        # [B,K,g,g]

        flat = maps.reshape(B, self.K, -1)
        attn = F.softmax(flat / self.temperature, dim=-1)

        # Lưới toạ độ trong [0,1] để khớp hệ cxcywh chuẩn của dự án (docs/02 mục 4).
        lin = torch.linspace(0.0, 1.0, grid, device=maps.device, dtype=attn.dtype)
        pos_y, pos_x = torch.meshgrid(lin, lin, indexing="ij")
        pos_x = pos_x.reshape(-1)
        pos_y = pos_y.reshape(-1)

        exp_x = (attn * pos_x).sum(-1)                              # [B,K]
        exp_y = (attn * pos_y).sum(-1)
        pts = torch.stack([exp_x, exp_y], dim=-1)                   # [B,K,2]

        # "Độ mạnh": bản đồ có đỉnh nhọn hay phẳng lì. SpatialSoftmax vứt mất thông tin này
        # (một map phẳng và một map đỉnh nhọn có thể cho CÙNG một toạ độ), nên giữ lại để
        # chẩn đoán — KHÔNG dùng trong loss.
        peak = flat.amax(-1) - flat.mean(-1)                        # [B,K]
        return pts, peak


# --------------------------------------------------------------------------- loss


def hungarian_loss(pts, centers):
    """Ghép 1-1 điểm <-> GT bằng Hungarian, rồi phạt khoảng cách trên các cặp đã ghép.

    ⚠️ ĐÂY LÀ CHỖ BẢN CHẠY ĐẦU (2026-09-14) SAI, PHẢI SỬA. Bản đó dùng Chamfer hai chiều:

        chamfer  : mỗi GT cần MỘT điểm gần nó   -> chỉ cần 1 điểm tốt là thoả
        coverage : mỗi điểm cần gần MỘT GT      -> mọi điểm đứng CÙNG 1 vật đều thoả

    Không thành phần nào phạt việc các điểm TRÙNG NHAU, nên nghiệm tối ưu của loss đó đúng
    là "tất cả K điểm đứng ở chỗ tốt nhất" -- và model tìm đúng nghiệm đó: đo được
    spread = 0,28 ô (64 điểm nằm gọn trong ~4,5 px = CÙNG MỘT ĐIỂM lặp 64 lần). Khi đó
    median_offset 1,37 ô KHÔNG có nghĩa "mỗi vật có điểm riêng", chỉ có nghĩa "CE-130 đông
    vật nên chỗ nào cũng gần một vật". Loss vẫn giảm đều (0,1375 -> 0,0494): model học tốt,
    bài toán đặt sai.

    Hungarian sửa đúng bản chất: mỗi GT chỉ được MỘT điểm "nhận", nên hai điểm không thể
    cùng nhận một GT -> ép phân công thay vì ép xích lại gần. Cùng cơ chế matcher box của
    dự án (utils/matcher.py:73).

    Điểm/GT thừa không vào loss: ảnh có n_gt > K thì K điểm phủ K vật; n_gt < K thì các
    điểm dư do `repulsion_loss` lo.
    """
    if centers.numel() == 0:
        return None
    d = torch.cdist(centers[None], pts[None])[0]        # [n_gt, K]
    r, c = linear_sum_assignment(d.detach().cpu().numpy())
    return d[r, c].mean()


def repulsion_loss(pts, min_dist):
    """Phạt mỗi cặp điểm gần nhau hơn `min_dist`. Hinge, nên cặp đã đủ xa KHÔNG bị phạt.

    Hungarian một mình chưa đủ khi n_gt < K: các điểm dư không có GT nào để nhận, không ai
    kéo chúng đi, nên vẫn dồn được. Đây là lớp phòng thủ thứ hai cho rủi ro R2.
    """
    K = pts.shape[0]
    d = torch.cdist(pts[None], pts[None])[0]
    off = d[~torch.eye(K, dtype=torch.bool, device=d.device)]
    return F.relu(min_dist - off).mean()


# --------------------------------------------------------------------------- metric


@torch.no_grad()
def measure(pts, centers, grid):
    """Trả về khoảng cách (ĐƠN VỊ Ô LƯỚI) từ mỗi tâm GT tới điểm gần nhất.

    Dùng đúng đơn vị của baseline 3,60 ô để so trực tiếp được.
    """
    if centers.numel() == 0:
        return np.empty(0)
    d = torch.cdist(centers[None], pts[None])[0]        # [n_gt, K], toạ độ [0,1]
    return (d.min(dim=1).values * grid).cpu().numpy()   # -> ô lưới


@torch.no_grad()
def n_distinct_points(pts, grid, tol_cells=1.0):
    """Đếm số điểm THẬT SỰ phân biệt: gom cụm tham lam, hai điểm cách < tol_cells là một.

    `spread` (trung vị khoảng cách từng cặp) có thể đánh lừa khi vài điểm tách ra còn phần
    lớn dồn cục. Con số này trả lời thẳng: memory thực tế có bao nhiêu token hữu ích?
    K=64 mà ra 1-2 thì memory chỉ là 1-2 token, bất kể median_offset đẹp cỡ nào.
    """
    p = pts.cpu().numpy() * grid
    keep = []
    for q in p:
        if all(float(np.hypot(*(q - k))) >= tol_cells for k in keep):
            keep.append(q)
    return len(keep)


@torch.no_grad()
def spread_of_points(pts):
    """Khoảng cách trung vị giữa các điểm, đơn vị ô lưới. Chẩn đoán rủi ro R2 (sụp mode).

    Nếu K điểm dồn về cùng một chỗ thì con số này ~0 và memory chỉ còn 1 token hữu ích.
    """
    d = torch.cdist(pts[None], pts[None])[0]
    K = d.shape[0]
    off = d[~torch.eye(K, dtype=torch.bool, device=d.device)]
    return off.median().item()


# --------------------------------------------------------------------------- data


def load_split(ds, cache, encoder, device, limit, log_every=400):
    """-> (list[patch_raw fp16 CPU], list[tâm GT]).

    ⚠️ DÙNG LẠI CACHE CÓ SẴN (`tools/build_cache.py`, fp16 memmap) — KHÔNG tự chạy lại CLIP.
    Cache đã dựng từ EXPERIMENT A và là lý do train chỉ mất ~2 giờ (CLIP chiếm 76,8 % thời
    gian mỗi batch; cache cho ~4,3x). Chạy lại CLIP ở đây là lãng phí thuần và còn rủi ro
    lệch tiền xử lý so với lúc train.

    `flipped=False`: cửa chặn này không augment. Cache có 2 phiên bản (gốc + lật) vì KHÔNG
    thể lật token đã cache — ViT trộn thông tin toàn cục qua 12 layer nên token (i,j) không
    còn là "feature của riêng ô (i,j)".

    `encoder` chỉ dùng khi KHÔNG có cache (đường dự phòng, chậm).
    """
    feats, cents = [], []
    n = min(limit, len(ds)) if limit else len(ds)
    t0 = time.time()
    for i in range(n):
        s = ds.__getitem__(i, need_image=cache is None)
        if cache is not None:
            patch, _ = cache.get(s["image_id"], s["text"], False)
            feats.append(torch.from_numpy(patch).half())
        else:
            # encode_image_raw() ĐÒI ảnh đã CLIP-normalise, không phải [0,1] thuần. Chia 255
            # rồi đưa thẳng vào là sai âm thầm: vẫn chạy, vẫn ra feature, chỉ lệch phân phối.
            img = torch.from_numpy(normalize_for_clip(s["image"]))[None].to(device)
            feats.append(encoder.encode_image_raw(img)[0].half().cpu())
        b = s["boxes"]
        cents.append(torch.from_numpy(np.asarray(b[:, :2], dtype=np.float32)))
        if (i + 1) % log_every == 0:
            print(f"    {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return feats, cents


# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--cache", default=None,
                    help="THƯ MỤC CACHE patch token có sẵn (tools/build_cache.py) — DÙNG "
                         "CÁI ĐÃ DỰNG TỪ EXPERIMENT A, đừng dựng lại. Không truyền thì "
                         "chạy CLIP tại chỗ, chậm hơn nhiều.")
    ap.add_argument("--data-root", default=None,
                    help="mặc định lấy cfg['data']['root']")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--w-repulsion", type=float, default=1.0)
    ap.add_argument("--min-dist-cells", type=float, default=2.0,
                    help="hai điểm gần hơn ngưỡng này (ô lưới) thì bị phạt. Mặc định 2,0 vì "
                         "box CE-130 rộng 2,4-4,4 ô -> hai điểm cách < 2 ô gần như chắc "
                         "chắn đang bám cùng một vật.")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"],
                    help="MẶC ĐỊNH val: cache của EXPERIMENT A chỉ dựng train+val (train.py "
                         "chỉ cần 2 split đó). val cũng có 28 class GIAO=0 với 72 class "
                         "train nên trả lời câu hỏi zero-shot y hệt test, và val KHÔNG dính "
                         "lô annotation rác của test (4,2 % GT — docs/02 mục 7.2). Chọn "
                         "test thì phải dựng cache test trước.")
    ap.add_argument("--limit-train", type=int, default=0, help="0 = toàn bộ")
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="keypoint_gate.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  K={a.K}  epochs={a.epochs}  lr={a.lr}", flush=True)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    grid = cfg["data"]["image_size"] // 16                 # ViT-B/16@512 -> 32

    root = a.data_root or cfg["data"]["root"]
    size = cfg["data"]["image_size"]
    ev = a.eval_split
    ds_tr = CE130Detection(root, "train", size)
    ds_te = CE130Detection(root, ev, size)
    print(f"[1/3] dữ liệu: train {len(ds_tr)} ảnh | {ev} {len(ds_te)} ảnh", flush=True)

    encoder = None
    if a.cache:
        print(f"[2/3] đọc CACHE CÓ SẴN: {a.cache}", flush=True)
        cache_tr, cache_te = PatchCache(a.cache, "train"), PatchCache(a.cache, ev)
    else:
        print("[2/3] ⚠️ KHÔNG có --cache -> chạy CLIP tại chỗ (chậm). Cache của "
              "EXPERIMENT A đã có sẵn, nên truyền --cache.", flush=True)
        cache_tr = cache_te = None
        encoder = build_model(cfg, dropout=0.0).eval().encoder.to(device)
        for p in encoder.parameters():
            p.requires_grad_(False)

    with torch.no_grad():
        Xtr, Ctr = load_split(ds_tr, cache_tr, encoder, device, a.limit_train)
        Xte, Cte = load_split(ds_te, cache_te, encoder, device, a.limit_eval)

    n_tok = Xtr[0].shape[0]
    assert n_tok == grid * grid, (
        f"patch token = {n_tok} nhưng grid*grid = {grid*grid}. Cache dựng với image_size "
        f"khác config? Lệch chỗ này làm SpatialSoftmax gán sai toạ độ mà không báo lỗi.")

    head = KeypointHead(d_in=Xtr[0].shape[-1], K=a.K, temperature=a.temperature).to(device)
    n_param = sum(p.numel() for p in head.parameters())
    print(f"[3/3] train head — {n_param:,} tham số (CLIP vẫn FROZEN)", flush=True)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)

    order = np.arange(len(Xtr))
    for ep in range(a.epochs):
        head.train()
        np.random.shuffle(order)
        tot, nb = 0.0, 0
        for i in order:
            c = Ctr[i].to(device)
            if c.numel() == 0:
                continue
            pts, _ = head(Xtr[i][None].float().to(device), grid)
            l_main = hungarian_loss(pts[0], c)
            l_rep = repulsion_loss(pts[0], a.min_dist_cells / grid)
            loss = l_main + a.w_repulsion * l_rep
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    epoch {ep+1:3d}/{a.epochs}  loss {tot/max(nb,1):.4f}", flush=True)

    print("\n[đo]  — đơn vị Ô LƯỚI (baseline CLIP cosine frozen = 3,60 ô)", flush=True)
    head.eval()
    res = {}
    for name, X, C in (("train", Xtr, Ctr), (ev, Xte, Cte)):
        offs, spreads, peaks, distincts = [], [], [], []
        with torch.no_grad():
            for i in range(len(X)):
                c = C[i].to(device)
                if c.numel() == 0:
                    continue
                pts, pk = head(X[i][None].float().to(device), grid)
                offs.append(measure(pts[0], c, grid))
                spreads.append(spread_of_points(pts[0]))
                distincts.append(n_distinct_points(pts[0], grid))
                peaks.append(float(pk.mean()))
        offs = np.concatenate(offs) if offs else np.empty(0)
        res[name] = {
            "n_gt": int(offs.size),
            "median_offset_cells": float(np.median(offs)),
            "mean_offset_cells": float(offs.mean()),
            "pct_within_1cell": float((offs < 1.0).mean() * 100),
            "pct_within_2cells": float((offs < 2.0).mean() * 100),
            "median_point_spread_cells": float(np.median(spreads)),
            "median_n_distinct_points": float(np.median(distincts)),
            "mean_peak_sharpness": float(np.mean(peaks)),
        }

    print()
    print("  split |  n_gt  | median | trong 1 ô | trong 2 ô | spread | #rõ | peak")
    print("  ------+--------+--------+-----------+-----------+--------+-----+------")
    for k, v in res.items():
        print(f"  {k:5s} | {v['n_gt']:6d} | {v['median_offset_cells']:6.2f} | "
              f"{v['pct_within_1cell']:8.1f} % | {v['pct_within_2cells']:8.1f} % | "
              f"{v['median_point_spread_cells']:6.2f} | "
              f"{v['median_n_distinct_points']:3.0f} | {v['mean_peak_sharpness']:.3f}")
    print()
    print("  BASELINE (CLIP cosine frozen, không train): median 3,60 ô | 11,7 % trong 1 ô")
    print()

    t = res[ev]
    # ĐIỀU KIỆN TIÊN QUYẾT: K điểm phải PHÂN TÁN. Nếu chúng trùng nhau thì median_offset
    # KHÔNG đọc được -- nó chỉ đo "một điểm cách vật gần nhất bao xa", mà CE-130 có 20-30
    # vật/ảnh nên chỗ nào cũng gần một vật. Đây đúng là cách bản chạy đầu (spread 0,28 ô)
    # cho median 1,37 ô trông như thành công.
    n_ok = t["median_n_distinct_points"]
    if n_ok < 0.5 * a.K:
        verdict = "KHÔNG ĐỌC ĐƯỢC"
        note = (f"K điểm SỤP MODE: chỉ {n_ok:.0f}/{a.K} điểm phân biệt (spread "
                f"{t['median_point_spread_cells']:.2f} ô). Mọi chỉ số offset bên trên VÔ "
                f"NGHĨA -- chúng đo một điểm duy nhất trên ảnh 20-30 vật. Tăng "
                f"--w-repulsion hoặc --min-dist-cells rồi chạy lại.")
    elif t["median_offset_cells"] < 1.5 and t["pct_within_1cell"] > 40:
        verdict = "ĐẠT"
        note = "Conv1x1 học được bản đồ có đỉnh trên vật -> hướng SỐNG, viết model tiếp."
    elif t["median_offset_cells"] > 2.5:
        verdict = "KHÔNG ĐẠT"
        note = ("CLIP frozen không cho ra toạ độ được bằng tổ hợp tuyến tính -> toàn bộ nhánh "
                "'ảnh -> toạ độ' CHẾT. C3 (box<->box) là đường duy nhất còn lại.")
    else:
        verdict = "XÁM"
        note = "Ở giữa hai ngưỡng — KHÔNG tự quyết, mang số về bàn."
    print(f"  => {verdict}: {note}")

    gap = res["train"]["median_offset_cells"] - t["median_offset_cells"]
    if abs(gap) > 0.8:
        print(f"  ⚠️ train↔eval lệch {abs(gap):.2f} ô — Conv1x1 có thể học thuộc 72 class "
              f"train, MẤT zero-shot (rủi ro R4).")

    res["_meta"] = {"eval_split": ev, "K": a.K, "epochs": a.epochs, "lr": a.lr, "n_param": n_param,
                    "temperature": a.temperature, "w_repulsion": a.w_repulsion,
                    "min_dist_cells": a.min_dist_cells,
                    "grid": grid, "verdict": verdict,
                    "baseline_median_cells": 3.60, "baseline_pct_within_1cell": 11.7}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
