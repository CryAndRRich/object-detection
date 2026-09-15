#!/usr/bin/env python3
"""CỬA CHẶN cho hướng "Conv1x1 + SpatialSoftmax -> K toạ độ làm memory".

CÂU HỎI
-------
`Conv1x1(768 -> K)` trên CLIP ViT-B/16 **FROZEN** có sinh ra K bản đồ mà **mỗi bản đồ có một
đỉnh nằm đúng trên một vật** không? Nếu không thì K toạ độ là rác, và toàn bộ thiết kế
(memory = K token toạ độ, cross-attention với box token, bias 4 góc kiểu BoxRPB) vô nghĩa.

Giả thuyết nằm trọn trong 2 lớp (~49K tham số với K=64) nên đo được mà KHÔNG cần viết
decoder/diffusion/box token.

BỐN MỐC SO SÁNH
---------------
  lưới đều       : sqrt(K) x sqrt(K) điểm cách đều, KHÔNG nhìn ảnh, KHÔNG train  <- tool tự đo
  CLIP cosine    : median 3,60 ô | 11,7 % trong 1 ô   (docs/01-bai-toan.md mục 6)
  box CE-130     : rộng trung vị 1,96 ô, cao 1,71 ô (train) -> bán kính ~1 ô
  box/ảnh        : trung vị 20 (train) / 21 (val), mean 37,6 / 42,2, p90 ~90

⚠️ MỐC "LƯỚI ĐỀU" LÀ QUAN TRỌNG NHẤT và trước giờ CHƯA đo. CE-130 có 20+ vật rải khắp ảnh
   nên rải điểm đều cũng tự động "gần" một vật. Không có mốc này thì median 1,34 ô (lần chạy
   trước) không phân biệt được "model đọc ảnh" với "dữ liệu đông vật".

NGƯỠNG — CHỐT TRƯỚC KHI CHẠY
----------------------------
  TIÊN QUYẾT     : #điểm phân biệt >= K/2, nếu không -> "KHÔNG ĐỌC ĐƯỢC"
  KHÔNG ĐẠT      : hơn lưới đều < 0,30 ô  (model không thật sự đọc ảnh)
  ĐẠT            : median < 1,5 ô VÀ > 40 % GT trong 1 ô
  XÁM            : còn lại — mang số về bàn, KHÔNG tự quyết

SÁU CHỈ SỐ, mỗi cái trả lời một câu hỏi riêng
---------------------------------------------
  median / 1 ô / 2 ô   : GT có điểm gần không (recall)          <- cho phép 1 điểm phục vụ nhiều GT
  điểm nằm trong box   : K có THỪA không (precision)
  GT ghép 1-1          : mỗi vật có TOKEN RIÊNG không           <- ép phân công, không lạc quan
  #rõ                  : memory thực tế có bao nhiêu token hữu ích
  ổn định (std vị trí) : kênh k có CHUYÊN HOÁ theo ảnh không, hay luôn trỏ một chỗ
  Δloss cuối           : đã bão hoà chưa

⚠️ ĐO TRÊN VAL. 28 class giao=0 với 72 class train nên trả lời zero-shot y hệt test; và ĐO
   ĐƯỢC test khác hẳn train/val (30 vs 20-21 box/ảnh, box 2,50 vs 1,96 ô, 815 box >50 % ảnh
   vs 0) nên test thêm nhiễu không liên quan giả thuyết.

⚠️ CHỈ SỐ NÀY ĐO ĐIỂM, KHÔNG ĐO VÙNG (bài học docs/01 mục 6.1: AUC 0,782 chứng minh phân
   biệt VÙNG, không chứng minh định vị ĐIỂM).

HAI BUG ĐÃ SỬA trong các lần chạy trước — ghi để không lặp
----------------------------------------------------------
  (a) Chamfer hai chiều không phạt điểm TRÙNG NHAU -> nghiệm tối ưu là "cả K điểm đứng cùng
      một chỗ". Đổi sang Hungarian (ghép 1-1) + repulsion hinge.
  (b) `spread` thiếu `* grid` -> in ra hệ [0,1] mà nhãn ghi "ô lưới"; 0,28 bị đọc nhầm là
      "sụp mode" trong khi số thật là 9,0 ô. Dấu hiệu lộ bug: #rõ = 51 điểm phân biệt thì
      spread KHÔNG THỂ là 0,32 ô. Bài học: hai chỉ số cùng đo một thứ mà mâu thuẫn thì dừng
      lại kiểm ĐƠN VỊ, đừng diễn giải.
  (c) `--min-dist-cells` từng đặt 2,0 theo con số SAI trong docs ("box rộng 2,4-4,4 ô"). Số
      đúng là 1,96 ô -> ngưỡng 2,0 đẩy điểm ra xa hơn cả một box. Nay mặc định 1,0.

CHẠY (TRÊN SERVER)
------------------
  python tools/run_on_free_gpu.py -- tools/check_keypoint_head.py \
      --cache <thư-mục-cache> --out <đường-dẫn>.json

  ⚠️ PHẢI truyền --cache (cache fp16 đã dựng từ EXPERIMENT A, tools/build_cache.py). Không
     truyền thì chạy lại CLIP từ đầu — lãng phí và rủi ro lệch tiền xử lý so với lúc train.
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
    """CLIP frozen -> [bottleneck] -> vài lớp conv -> K bản đồ -> SpatialSoftmax -> K toạ độ.

    SpatialSoftmax giữ nguyên công thức của CE-Loc gốc
    (refs/repos/Count-Editing/CE-LocModel/models/spatial_softmax.py): softmax trên H*W riêng
    từng kênh, rồi lấy KỲ VỌNG với lưới toạ độ. Đầu ra là TOẠ ĐỘ, không phải embedding.

    ⚠️ VÌ SAO CÓ THAM SỐ `ksize`/`depth` — LỖ HỔNG TRONG KẾT LUẬN LẦN TRƯỚC.
    Bản đầu chỉ có `Conv2d(768, K, kernel_size=1)`. Nó KHÔNG ĐẠT (hơn baseline lưới đều chỉ
    +0,15 ô ở K=144), và đã suýt bị kết luận là "CLIP frozen không chứa thông tin tâm vật".
    Kết luận đó VƯỢT QUÁ bằng chứng: cái đo được là "**tổ hợp tuyến tính 1x1** của CLIP frozen
    không rút được tâm vật".

    Conv 1x1 nhìn ĐÚNG MỘT ô lưới, không thấy ô lân cận. Nhưng "tâm vật" theo định nghĩa cần
    lân cận -- phải thấy biên trái VÀ biên phải mới biết giữa ở đâu. Với box CE-130 rộng
    trung vị 1,96 ô (ĐO ĐƯỢC, tools/check_data_facts.py), một kernel 3x3 đã phủ trọn vật.

    Nên `--ksize 1 --depth 1` tái lập CHÍNH XÁC bản đã chạy, còn `--ksize 3 --depth 3` là
    phép thử mới. So hai cái đó tách bạch được hai giả thuyết:
      - CLIP frozen KHÔNG CÓ thông tin  -> cả hai đều thua lưới đều
      - kiến trúc head quá nghèo        -> 3x3 vượt lên
    """

    def __init__(self, d_in=768, K=64, temperature=1.0, ksize=3, depth=3, hidden=256):
        super().__init__()
        layers = []
        c_in = d_in
        if depth > 1:
            # Bottleneck 1x1 hạ 768 -> hidden TRƯỚC khi vào conv không gian. Không có nó thì
            # một lớp 3x3 trên 768 kênh tốn 768*hidden*9 tham số, lấn át phần còn lại và biến
            # phép thử "thêm lân cận" thành phép thử "thêm sức chứa".
            layers += [nn.Conv2d(d_in, hidden, 1), nn.GELU()]
            c_in = hidden
            for _ in range(depth - 2):
                layers += [nn.Conv2d(hidden, hidden, ksize, padding=ksize // 2), nn.GELU()]
        layers += [nn.Conv2d(c_in, K, ksize, padding=ksize // 2)]
        self.net = nn.Sequential(*layers)
        self.K = K
        self.temperature = temperature

    def forward(self, patch_raw, grid):
        """patch_raw [B, grid*grid, 768] -> pts [B, K, 2] trong [0,1], peak [B, K]."""
        B = patch_raw.shape[0]
        x = patch_raw.transpose(1, 2).reshape(B, -1, grid, grid)   # [B,768,g,g]
        maps = self.net(x)                                         # [B,K,g,g]

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


# --------------------------------------------------------------- cấu hình head

# BA CẤU HÌNH, tách bạch hai biến. Bản chạy trước chỉ có cái đầu, nên khi nó KHÔNG ĐẠT thì
# không phân biệt được "CLIP frozen không có thông tin" với "head quá nghèo".
#
#   1x1_shallow : TÁI LẬP bản đã chạy (49K @ K=64). Mốc so.
#   1x1_deep    : sâu + rộng NHƯNG vẫn 1x1 -> thêm SỨC CHỨA, KHÔNG thêm lân cận.
#   3x3_deep    : có lân cận, số tham số CỐ Ý khớp 1x1_deep (299.664 vs 281.424 @ K=144,
#                 lệch 6 %) -> chênh lệch giữa hai cái này CHỈ có thể do lân cận.
#
# Đọc kết quả:
#   cả ba đều thua lưới đều            -> CLIP frozen KHÔNG chứa thông tin tâm vật. Hướng chết.
#   1x1_deep ≈ 1x1_shallow < 3x3_deep  -> thiếu LÂN CẬN. Hướng sống, đi tiếp.
#   1x1_deep ≈ 3x3_deep > 1x1_shallow  -> chỉ là SỨC CHỨA, lân cận không giúp.
HEAD_CONFIGS = {
    "1x1_shallow": dict(ksize=1, depth=1, hidden=256),
    "1x1_deep":    dict(ksize=1, depth=3, hidden=256),
    "3x3_deep":    dict(ksize=3, depth=3, hidden=96),
}


# --------------------------------------------------------------------------- loss


def hungarian_loss(pts, boxes):
    """Ghép 1-1 điểm <-> GT bằng Hungarian, rồi phạt khoảng cách trên các cặp đã ghép.

    ⚠️ ĐÂY LÀ CHỖ BẢN CHẠY ĐẦU (2026-09-14) SAI, PHẢI SỬA. Bản đó dùng Chamfer hai chiều:

        chamfer  : mỗi GT cần MỘT điểm gần nó   -> chỉ cần 1 điểm tốt là thoả
        coverage : mỗi điểm cần gần MỘT GT      -> mọi điểm đứng CÙNG 1 vật đều thoả

    Không thành phần nào phạt việc các điểm TRÙNG NHAU, nên nghiệm tối ưu của loss đó đúng
    là "tất cả K điểm đứng ở chỗ tốt nhất". Đổi sang Hungarian để chặn khả năng đó.

    ⚠️ Nhưng lần chạy v1 KHÔNG thật sự sụp mode như đã đọc lúc đầu -- cột `spread` khi đó bị
    BUG ĐƠN VỊ (thiếu `* grid`, xem `spread_of_points`). Số thật là 0,28*32 = 9,0 ô, tức các
    điểm vẫn phân tán. Chênh lệch v1 -> v2 do đó nhỏ (median 1,37 -> 1,34).

    Hungarian sửa đúng bản chất: mỗi GT chỉ được MỘT điểm "nhận", nên hai điểm không thể
    cùng nhận một GT -> ép phân công thay vì ép xích lại gần. Cùng cơ chế matcher box của
    dự án (utils/matcher.py:73).

    Điểm/GT thừa không vào loss: ảnh có n_gt > K thì K điểm phủ K vật; n_gt < K thì các
    điểm dư do `repulsion_loss` lo.
    """
    if boxes.numel() == 0:
        return None
    d = torch.cdist(boxes[:, :2][None], pts[None])[0]   # [n_gt, K]
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
def measure(pts, boxes, grid):
    """Trả về khoảng cách (ĐƠN VỊ Ô LƯỚI) từ mỗi tâm GT tới điểm gần nhất.

    Dùng đúng đơn vị của baseline 3,60 ô để so trực tiếp được.
    """
    if boxes.numel() == 0:
        return np.empty(0)
    d = torch.cdist(boxes[:, :2][None], pts[None])[0]   # [n_gt, K], toạ độ [0,1]
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
def precision_in_box(pts, boxes):
    """% điểm nằm TRONG ít nhất một box GT. Đối xứng với pct_within_1cell (recall).

    Trả lời "K có THỪA không": #rõ = 51 điểm phân biệt trong khi CE-130 trung vị chỉ 20-30
    vật/ảnh -> nếu precision thấp thì hơn nửa số điểm đang đứng ở chỗ TRỐNG (bị repulsion
    đẩy ra), và memory sẽ đầy token rác. Không có chỉ số này thì không biết nên tăng hay
    giảm K.
    """
    if boxes.numel() == 0:
        return np.nan
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    px, py = pts[:, 0][:, None], pts[:, 1][:, None]          # [K,1]
    inside = ((px >= (cx - w / 2)[None]) & (px <= (cx + w / 2)[None]) &
              (py >= (cy - h / 2)[None]) & (py <= (cy + h / 2)[None]))   # [K, n_gt]
    return float(inside.any(dim=1).float().mean() * 100)


@torch.no_grad()
def matched_coverage(pts, boxes, grid, tol_cells=1.0):
    """% GT được ghép 1-1 với MỘT điểm riêng và cặp đó gần hơn tol_cells.

    Khác `pct_within_1cell`: chỉ số kia cho phép MỘT điểm phục vụ nhiều GT (nearest-neighbour),
    nên trên ảnh đông vật nó lạc quan. Cái này ép phân công 1-1 -> đo đúng thứ memory cần:
    mỗi vật có TOKEN RIÊNG của nó.
    """
    if boxes.numel() == 0:
        return np.nan
    d = torch.cdist(boxes[:, :2][None], pts[None])[0]
    r, c = linear_sum_assignment(d.cpu().numpy())
    return float(((d[r, c] * grid) < tol_cells).float().mean() * 100)


@torch.no_grad()
def uniform_grid_points(K, device, dtype):
    """Baseline ĐIỂM RẢI ĐỀU: sqrt(K) x sqrt(K) điểm cách đều trên ảnh, KHÔNG nhìn ảnh.

    ⚠️ ĐÂY LÀ BASELINE QUAN TRỌNG NHẤT và trước giờ CHƯA đo. CE-130 có 20-48 vật/ảnh, nên
    rải điểm đều cũng tự động "gần" một vật nào đó. Không có mốc này thì median 1,34 ô không
    biết là do model học được hay do dữ liệu đông vật. So với baseline CLIP-cosine (3,60 ô)
    là chưa đủ: cosine còn bị nhiễu ngữ nghĩa, còn lưới đều thì không.
    """
    n = int(round(K ** 0.5))
    lin = torch.linspace(0.5 / n, 1 - 0.5 / n, n, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


@torch.no_grad()
def channel_stability(all_pts, grid):
    """Độ lệch chuẩn vị trí của TỪNG kênh qua các ảnh (trung vị trên K kênh), ô lưới.

    Hỏi: kênh k có CHUYÊN HOÁ không? Nếu std nhỏ thì kênh k luôn trỏ về cùng một chỗ bất kể
    ảnh -> nó chỉ học vị trí trung bình, không đọc nội dung. Nếu std lớn thì nó thật sự phản
    ứng theo ảnh. (Không có ngưỡng cứng; đọc cùng precision.)
    """
    p = torch.stack(all_pts)                       # [n_img, K, 2]
    return float((p.std(dim=0) * grid).mean(dim=-1).median())


@torch.no_grad()
def spread_of_points(pts, grid):
    """Khoảng cách trung vị giữa các điểm, ĐƠN VỊ Ô LƯỚI. Chẩn đoán rủi ro R2 (sụp mode).

    ⚠️ BUG ĐÃ SỬA (v1+v2 đều dính): hàm này trả về khoảng cách trong hệ [0,1] nhưng docstring
    và bảng in đều ghi "ô lưới" -- THIẾU `* grid`. `measure()` có nhân, hàm này thì không.
    Hệ quả: v1 in "spread 0,28" và bị đọc là "64 điểm nằm trong 4,5 px = sụp mode", trong khi
    số thật là 0,28*32 = 9,0 ô. v2 in 0,32 tức 10,3 ô. Hai chỉ số trong cùng bảng mâu thuẫn
    nhau (#rõ = 51 điểm phân biệt thì spread KHÔNG THỂ là 0,32 ô) -- đó là dấu hiệu lộ bug.
    """
    d = torch.cdist(pts[None], pts[None])[0]
    K = d.shape[0]
    off = d[~torch.eye(K, dtype=torch.bool, device=d.device)]
    return off.median().item() * grid


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
        # Giữ CẢ 4 chiều: tâm cho khoảng cách, w/h cho phép đo "điểm có nằm TRONG box không"
        # (precision). Chỉ giữ tâm thì không trả lời được câu "K có thừa không".
        cents.append(torch.from_numpy(np.asarray(s["boxes"], dtype=np.float32)))
        if (i + 1) % log_every == 0:
            print(f"    {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)
    return feats, cents


# --------------------------------------------------------------------------- main


def evaluate(head, X, C, grid, device, K):
    """Chạy trên một split -> dict chỉ số. `head=None` => baseline LƯỚI ĐỀU."""
    offs, spreads, peaks, distincts, precs, covs, allpts = [], [], [], [], [], [], []
    with torch.no_grad():
        for i in range(len(X)):
            c = C[i].to(device)
            if c.numel() == 0:
                continue
            if head is None:
                pts = uniform_grid_points(K, device, torch.float32)
                pk = torch.zeros(pts.shape[0], device=device)
            else:
                p_, pk_ = head(X[i][None].float().to(device), grid)
                pts, pk = p_[0], pk_[0]
            offs.append(measure(pts, c, grid))
            spreads.append(spread_of_points(pts, grid))
            distincts.append(n_distinct_points(pts, grid))
            precs.append(precision_in_box(pts, c))
            covs.append(matched_coverage(pts, c, grid))
            peaks.append(float(pk.mean()))
            allpts.append(pts)
    offs = np.concatenate(offs) if offs else np.empty(0)
    return {
        "n_gt": int(offs.size),
        "median_offset_cells": float(np.median(offs)),
        "mean_offset_cells": float(offs.mean()),
        "pct_within_1cell": float((offs < 1.0).mean() * 100),
        "pct_within_2cells": float((offs < 2.0).mean() * 100),
        "pct_points_in_box": float(np.nanmean(precs)),
        "pct_gt_matched_1to1": float(np.nanmean(covs)),
        "median_point_spread_cells": float(np.median(spreads)),
        "median_n_distinct_points": float(np.median(distincts)),
        "mean_peak_sharpness": float(np.mean(peaks)),
        "channel_stability_cells": (float(channel_stability(allpts, grid))
                                    if head is not None else float("nan")),
    }


def train_head(Xtr, Ctr, grid, device, K, a, tag, hcfg):
    """Train MỘT head. Trả kèm lịch sử loss để biết đã bão hoà chưa."""
    head = KeypointHead(d_in=Xtr[0].shape[-1], K=K, temperature=a.temperature,
                        **hcfg).to(device)
    n_param = sum(p.numel() for p in head.parameters())
    print(f"  [{tag}] {n_param:,} tham số (CLIP vẫn FROZEN)", flush=True)
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    order = np.arange(len(Xtr))
    hist = []
    for ep in range(a.epochs):
        head.train()
        np.random.shuffle(order)
        tot, nb = 0.0, 0
        for i in order:
            c = Ctr[i].to(device)
            if c.numel() == 0:
                continue
            pts, _ = head(Xtr[i][None].float().to(device), grid)
            loss = (hungarian_loss(pts[0], c)
                    + a.w_repulsion * repulsion_loss(pts[0], a.min_dist_cells / grid))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        hist.append(tot / max(nb, 1))
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"      epoch {ep+1:3d}/{a.epochs}  loss {hist[-1]:.4f}", flush=True)
    return head, n_param, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--cache", default=None,
                    help="THƯ MỤC CACHE patch token có sẵn (tools/build_cache.py) — DÙNG CÁI "
                         "ĐÃ DỰNG TỪ EXPERIMENT A, đừng dựng lại.")
    ap.add_argument("--data-root", default=None, help="mặc định lấy cfg['data']['root']")
    ap.add_argument("--K-sweep", default="16,25,64,144",
                    help="Quét K. ĐO ĐƯỢC (tools/check_data_facts.py, 2026-09-14): CE-130 có "
                         "box/ảnh TRUNG VỊ 20 (train) / 21 (val), mean 37,6 / 42,2, p90 ~90. "
                         "Dải này bao quanh trung vị và chạm tới mean. Mọi K là SỐ CHÍNH "
                         "PHƯƠNG để baseline lưới đều sqrt(K) x sqrt(K) so được công bằng.")
    ap.add_argument("--epochs", type=int, default=60,
                    help="60 chứ không phải 30: lần chạy trước loss vẫn giảm đều tới epoch "
                         "cuối (0,0453 -> 0,0374), CHƯA bão hoà.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--heads", default="1x1_shallow,1x1_deep,3x3_deep",
                    help="Danh sách cấu hình head chạy trong CÙNG một lượt. Ba cái mặc định "
                         "tách bạch được HAI biến (lân cận vs sức chứa) -- xem HEAD_CONFIGS. "
                         "Truyền '' để dùng --ksize/--depth/--hidden đơn lẻ.")
    ap.add_argument("--ksize", type=int, default=3,
                    help="kernel không gian. 1 = tái lập bản đã chạy (KHÔNG ĐẠT); 3 = phép "
                         "thử mới. Box CE-130 rộng trung vị 1,96 ô nên 3x3 phủ trọn vật.")
    ap.add_argument("--depth", type=int, default=3,
                    help="số lớp conv. depth=1 -> đúng một Conv2d(768,K,ksize), không "
                         "bottleneck (tái lập bản cũ khi ksize=1).")
    ap.add_argument("--hidden", type=int, default=256,
                    help="bottleneck 1x1 hạ 768 -> hidden trước conv không gian, để phép thử "
                         "đo 'thêm lân cận' chứ không phải 'thêm sức chứa'.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--w-repulsion", type=float, default=1.0)
    ap.add_argument("--min-dist-cells", type=float, default=1.0,
                    help="Hai điểm gần hơn ngưỡng này (ô lưới) thì bị phạt. ⚠️ ĐÃ SỬA 2,0 -> "
                         "1,0: số 2,0 dựa trên con số SAI trong docs ('box rộng 2,4-4,4 ô'). "
                         "ĐO LẠI TỪ DỮ LIỆU: box rộng TRUNG VỊ 1,96 ô, cao 1,71 ô (train) -> "
                         "ngưỡng 2,0 đẩy các điểm ra XA HƠN CẢ MỘT BOX, tức tự tay tạo ra "
                         "điểm nằm ở chỗ trống. 1,0 ~ bán kính box trung vị.")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"],
                    help="MẶC ĐỊNH val. Cache của EXPERIMENT A chỉ dựng train+val. Và ĐO "
                         "ĐƯỢC: test khác hẳn train/val (30 vs 20-21 box/ảnh trung vị, box "
                         "rộng 2,50 vs 1,96 ô, 815 box >50%% ảnh vs 0) -> val gần train hơn "
                         "về phân bố VÀ vẫn là 28 class giao=0 với train, nên trả lời câu "
                         "hỏi zero-shot y hệt test mà không dính lô annotation rác.")
    ap.add_argument("--limit-train", type=int, default=0, help="0 = toàn bộ")
    ap.add_argument("--limit-eval", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="keypoint_gate.json")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ks = [int(k) for k in a.K_sweep.split(",")]
    print(f"device={device}  K_sweep={Ks}  epochs={a.epochs}  lr={a.lr}  "
          f"min_dist={a.min_dist_cells} ô  head=conv{a.ksize}x{a.ksize} "
          f"depth={a.depth} hidden={a.hidden}", flush=True)

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    grid = cfg["data"]["image_size"] // 16                 # ViT-B/16@512 -> 32
    root = a.data_root or cfg["data"]["root"]
    size = cfg["data"]["image_size"]
    ev = a.eval_split

    ds_tr = CE130Detection(root, "train", size)
    ds_te = CE130Detection(root, ev, size)
    print(f"[1/4] dữ liệu: train {len(ds_tr)} ảnh | {ev} {len(ds_te)} ảnh", flush=True)

    encoder = None
    if a.cache:
        print(f"[2/4] đọc CACHE CÓ SẴN: {a.cache}", flush=True)
        cache_tr, cache_te = PatchCache(a.cache, "train"), PatchCache(a.cache, ev)
    else:
        print("[2/4] ⚠️ KHÔNG có --cache -> chạy CLIP tại chỗ (chậm).", flush=True)
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
        f"khác config? Lệch chỗ này làm SpatialSoftmax gán sai toạ độ mà KHÔNG báo lỗi.")

    # BASELINE LƯỚI ĐỀU -- không nhìn ảnh, không train, không tham số.
    # ⚠️ ĐÂY LÀ MỐC QUAN TRỌNG NHẤT VÀ TRƯỚC GIỜ CHƯA ĐO. CE-130 có trung vị 20-21 vật/ảnh
    # rải khắp canvas, nên rải điểm ĐỀU cũng tự động "gần" một vật nào đó. Không có mốc này
    # thì median 1,34 ô không biết là model đọc được ảnh hay chỉ là hệ quả của dữ liệu đông.
    # So với baseline CLIP-cosine (3,60 ô) là CHƯA ĐỦ: cosine còn bị nhiễu ngữ nghĩa kéo
    # xuống, còn lưới đều thì không có nhược điểm đó.
    print("[3/4] baseline LƯỚI ĐỀU (không nhìn ảnh, không train)", flush=True)
    res = {"_baseline_uniform": {str(K): evaluate(None, Xte, Cte, grid, device, K) for K in Ks}}

    names = [h for h in a.heads.split(",") if h] or ["_custom"]
    cfgs = {n: (HEAD_CONFIGS[n] if n in HEAD_CONFIGS
                else dict(ksize=a.ksize, depth=a.depth, hidden=a.hidden)) for n in names}
    print(f"[4/4] {len(names)} head x {len(Ks)} K", flush=True)
    res["_sweep"] = {}
    for hname in names:
        for K in Ks:
            head, n_param, hist = train_head(Xtr, Ctr, grid, device, K, a,
                                             f"{hname} K={K}", cfgs[hname])
            head.eval()
            tail = max(1, len(hist) // 10)
            res["_sweep"][f"{hname}|{K}"] = {
                "head": hname, "K": K, **cfgs[hname],
                "train": evaluate(head, Xtr, Ctr, grid, device, K),
                ev: evaluate(head, Xte, Cte, grid, device, K),
                "n_param": n_param,
                "loss_first": hist[0], "loss_last": hist[-1],
                "loss_drop_last10pct": hist[-tail - 1] - hist[-1] if len(hist) > tail else 0.0,
            }

    # ------------------------------------------------------------------ báo cáo
    print()
    print()
    print(f"  === {ev} (28 class CHƯA THẤY lúc train) ===")
    print("        head    |   K | tham số | median | 1 ô  | điểm∈box | GT 1-1 | #rõ")
    print("  --------------+-----+---------+--------+------+----------+--------+-----")
    for K in Ks:
        b = res["_baseline_uniform"][str(K)]
        print(f"  {'LƯỚI ĐỀU':<13s} | {K:3d} |       0 | {b['median_offset_cells']:6.2f} | "
              f"{b['pct_within_1cell']:4.1f} | {b['pct_points_in_box']:7.1f} % | "
              f"{b['pct_gt_matched_1to1']:5.1f} % |    ")
        for hname in names:
            r = res["_sweep"][f"{hname}|{K}"]
            v = r[ev]
            print(f"  {hname:<13s} | {K:3d} | {r['n_param']:7,d} | "
                  f"{v['median_offset_cells']:6.2f} | {v['pct_within_1cell']:4.1f} | "
                  f"{v['pct_points_in_box']:7.1f} % | {v['pct_gt_matched_1to1']:5.1f} % | "
                  f"{v['median_n_distinct_points']:3.0f}")
        print("  " + "-" * 62)
    print()
    print("  Mốc khác — CLIP cosine frozen, không train: median 3,60 ô | 11,7 % trong 1 ô")
    print("  ĐO ĐƯỢC: box CE-130 rộng trung vị 1,96 ô, cao 1,71 ô (train) -> bán kính ~1 ô.")
    print()

    # So 1x1_deep vs 3x3_deep ở cùng K: CHỈ khác lân cận (tham số khớp ~6 %).
    if "1x1_deep" in names and "3x3_deep" in names:
        print("  === LÂN CẬN có giúp không? (tham số khớp, chỉ khác kernel) ===")
        for K in Ks:
            a1 = res["_sweep"][f"1x1_deep|{K}"][ev]["median_offset_cells"]
            a3 = res["_sweep"][f"3x3_deep|{K}"][ev]["median_offset_cells"]
            s0 = res["_sweep"][f"1x1_shallow|{K}"][ev]["median_offset_cells"] \
                if "1x1_shallow" in names else float("nan")
            print(f"    K={K:3d}:  1x1_shallow {s0:5.2f}  ->  1x1_deep {a1:5.2f}  "
                  f"->  3x3_deep {a3:5.2f}   (lân cận: {a1-a3:+.2f} ô)")
        print()

    # ------------------------------------------------------------------ verdict
    best_key = min(res["_sweep"], key=lambda k: res["_sweep"][k][ev]["median_offset_cells"])
    best = res["_sweep"][best_key]
    best_K = best["K"]
    t = best[ev]
    b = res["_baseline_uniform"][str(best_K)]
    gain = b["median_offset_cells"] - t["median_offset_cells"]
    print(f"  Tốt nhất: {best_key}  ({best['n_param']:,} tham số, hơn lưới đều {gain:+.2f} ô)")

    if t["median_n_distinct_points"] < 0.5 * best_K:
        verdict = "KHÔNG ĐỌC ĐƯỢC"
        note = (f"SỤP MODE: {t['median_n_distinct_points']:.0f}/{best_K} điểm phân biệt. "
                f"Mọi chỉ số offset vô nghĩa.")
    elif gain < 0.3:
        verdict = "KHÔNG ĐẠT"
        note = (f"Chỉ hơn LƯỚI ĐỀU {gain:.2f} ô ({t['median_offset_cells']:.2f} vs "
                f"{b['median_offset_cells']:.2f}) -> head KHÔNG rút được tâm vật; con số "
                f"median đẹp là do CE-130 đông vật (20-21/ảnh). ⚠️ ĐỌC KỸ TRƯỚC KHI KẾT "
                f"LUẬN: xem khối 'LÂN CẬN có giúp không?' ở trên -- nếu 3x3 KHÔNG hơn 1x1 ở "
                f"cùng số tham số thì mới kết luận được CLIP frozen không chứa thông tin; "
                f"nếu 3x3 hơn rõ thì giới hạn nằm ở KIẾN TRÚC HEAD, còn thử tiếp được.")
    elif t["median_offset_cells"] < 1.5 and t["pct_within_1cell"] > 40:
        verdict = "ĐẠT"
        note = (f"Hơn lưới đều {gain:.2f} ô; {t['pct_points_in_box']:.0f} % điểm nằm TRONG "
                f"box; {t['pct_gt_matched_1to1']:.0f} % GT có token riêng -> viết model tiếp.")
    else:
        verdict = "XÁM"
        note = (f"Có đọc ảnh thật (hơn lưới đều {gain:.2f} ô) nhưng chưa đạt ngưỡng định vị. "
                f"Mang số về bàn, KHÔNG tự quyết.")
    print(f"  => {verdict}: {note}")

    gap = best["train"]["median_offset_cells"] - t["median_offset_cells"]
    if abs(gap) > 0.8:
        print(f"  ⚠️ train↔{ev} lệch {abs(gap):.2f} ô — có thể học thuộc 72 class train, "
              f"MẤT zero-shot (rủi ro R4).")
    if best["loss_drop_last10pct"] > 0.002:
        print("  ⚠️ loss VẪN đang giảm ở cuối — chưa bão hoà, thử --epochs lớn hơn.")
    if t["pct_points_in_box"] < 50:
        print(f"  ⚠️ chỉ {t['pct_points_in_box']:.0f} % điểm nằm trong một box — quá nửa số "
              f"token memory sẽ trỏ vào chỗ TRỐNG. Cân nhắc giảm K.")

    res["_meta"] = {"eval_split": ev, "K_sweep": Ks, "heads": names, "best": best_key, "epochs": a.epochs,
                    "lr": a.lr, "temperature": a.temperature, "w_repulsion": a.w_repulsion,
                    "min_dist_cells": a.min_dist_cells, "grid": grid, "verdict": verdict,
                    "ksize": a.ksize, "depth": a.depth, "hidden": a.hidden,
                    "gain_over_uniform_cells": gain,
                    "baseline_clip_cosine_median_cells": 3.60,
                    "data_box_width_cells_median_train": 1.96}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
