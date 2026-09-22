#!/usr/bin/env python3
"""CỬA CHẶN cho EXPERIMENT A — `Linear(256->4)` có đoán được HƯỚNG DỊCH không?

VÌ SAO TỒN TẠI
--------------
Cả EXPERIMENT A dựa trên đúng một giả định chưa ai đo:

    delta = box_delta(h)        # nn.Linear(256, 4)
    x     = update_box(x, delta)

Đây ĐÚNG DẠNG câu hỏi mà cửa chặn soft-argmax của vòng 1 đã đo và TRƯỢT: rút toạ độ
từ đặc trưng CLIP frozen (sai số trung vị 1,34 ô lưới, hơn lưới đều chỉ 0,15 ô).

Khác biệt duy nhất, và là lý do đáng thử lại: ở đây mạng không rút toạ độ TỪ ĐẦU mà rút
PHẦN DƯ so với box đã có, sau khi đã đọc ảnh TẠI CHÍNH BOX ĐÓ. Câu hỏi đổi từ
"vật nằm ở đâu trong ảnh?" thành "nhìn vùng này, box đang lệch sang trái hay phải?".

Lập luận "dễ hơn" đó CHƯA ĐƯỢC ĐO. Nếu sai thì mọi thứ xây trên nó vô nghĩa — và đo mất
vài phút, còn train mất ~10 giờ A30 (cạm bẫy 8 trong CLAUDE.md).

CÁCH ĐO
-------
1. Lấy GT box của val (28 lớp rời train, không dính lô annotation rác của test).
2. Dịch mỗi box đi một vector ngẫu nhiên độ dài `d` ô lưới -> `box_lech`.
3. `r = RoIFeatureSampler(patch_raw, box_lech)` với CLIP FROZEN (đọc từ cache).
4. Train `Linear(256->4)` dự đoán delta đưa box về GT (đúng tham số hoá của
   `update_box`: dịch tâm theo đơn vị kích thước box, kích thước nhân theo exp).
5. Đo cosine giữa delta dự đoán và delta thật, trên tập val giữ riêng.

TIÊU CHÍ: cosine > 0,5 ở `d = 1` ô. Dưới ngưỡng ⇒ cộng dồn vô nghĩa, DỪNG, thiết kế lại.

BA ĐỐI CHỨNG BẮT BUỘC, thiếu cái nào cũng đọc sai kết quả:
  - `shuffle`  : `r` bị xáo trộn giữa các mẫu. Nếu điểm không tụt thì `Linear` chỉ học
                 prior của phân bố delta, KHÔNG dùng ảnh.
  - `zero`     : `r` = 0. Trần của việc đoán mò.
  - theo NHÓM KÍCH THƯỚC: 18-29 % box CE-130 nhỏ hơn MỘT ô lưới, với chúng lưới 3x3 của
                 RoI thoái hoá thành 1x1 (đo được AUC đúng-cỡ = 0,000). Gộp chung sẽ che
                 mất việc nhóm này hỏng hoàn toàn.
  - theo `t`   : ở `t` lớn box gần nhiễu thuần, nên quét cả mức nhiễu chứ không chỉ `d`.

KHÔNG train model thật, KHÔNG cần GPU cho CLIP (đọc cache).

CHẠY TRÊN SERVER
----------------
  cd object-detection/count_editing/CE-LocModel
  python tools/run_on_free_gpu.py -- tools/gate_delta_direction.py \
      --cache /mnt/disk1/aiotlab/haitn/cache --split val \
      --out /mnt/disk1/aiotlab/haitn/log/gate_delta.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, PatchCache          # noqa: E402
from models.dit_blocks import MIN_WH                               # noqa: E402
from models.roi_sampler import RoIFeatureSampler                   # noqa: E402

GRID = 32                       # ViT-B/16 ở 512px -> lưới 32x32


def true_delta(box_lech, box_gt):
    """Delta mà `update_box` cần để đưa `box_lech` về `box_gt`.

    Đảo ngược đúng công thức object-normalized của V-DETR:
        cx' = cx + d_cx * w   ->   d_cx = (cx' - cx) / w
        w'  = w * exp(d_w)    ->   d_w  = log(w' / w)
    Dùng cùng `MIN_WH` như `update_box` để hai bên nhất quán.
    """
    cx, cy, w, h = box_lech.unbind(-1)
    gx, gy, gw, gh = box_gt.unbind(-1)
    w = w.clamp(min=MIN_WH)
    h = h.clamp(min=MIN_WH)
    return torch.stack([
        (gx - cx) / w,
        (gy - cy) / h,
        torch.log(gw.clamp(min=MIN_WH) / w),
        torch.log(gh.clamp(min=MIN_WH) / h),
    ], dim=-1)


def perturb(boxes, d_cells, rng):
    """Dịch tâm box đi đúng `d_cells` ô lưới theo hướng ngẫu nhiên. Kích thước giữ
    nguyên: cửa chặn hỏi về HƯỚNG DỊCH, thêm nhiễu kích thước sẽ trộn hai câu hỏi."""
    ang = torch.from_numpy(rng.uniform(0, 2 * np.pi, len(boxes))).float()
    step = d_cells / GRID
    out = boxes.clone()
    out[:, 0] = (boxes[:, 0] + step * torch.cos(ang)).clamp(0.0, 1.0)
    out[:, 1] = (boxes[:, 1] + step * torch.sin(ang)).clamp(0.0, 1.0)
    return out


def add_diffusion_noise(boxes, t, alphas_cumprod, snr_scale, rng):
    """Đưa box qua đúng đường mà `x_t` đi lúc train: encode -> q_sample -> decode.

    Cần thiết vì `update_box` nhân delta với `w`, mà ở `t` lớn `w` sau decode có đuôi
    chạm 0 (mục 7.1b của kế hoạch). Chỉ dịch `d` ô mà bỏ qua bước này thì cửa chặn sẽ
    đo một chế độ mà mô hình thật không bao giờ gặp.
    """
    from utils.box_ops import decode_diffusion, encode_diffusion
    x0 = encode_diffusion(boxes, snr_scale)
    noise = torch.from_numpy(rng.standard_normal(x0.shape)).float()
    a = alphas_cumprod[t]
    x_t = a.sqrt() * x0 + (1 - a).sqrt() * noise
    return decode_diffusion(x_t, snr_scale)


@torch.no_grad()
def collect(ds, cache, sampler, d_cells, t, alphas, snr_scale, seed, dev, max_img):
    """-> (r [K,256], delta_thật [K,4], rộng_ô [K]) cho mọi GT box của các ảnh đã chọn."""
    rng = np.random.default_rng(seed)
    R, D, W = [], [], []
    for i in range(min(len(ds), max_img)):
        s = ds.__getitem__(i, need_image=False)
        gt = torch.as_tensor(s["boxes"], dtype=torch.float32)
        if gt.numel() == 0:
            continue
        patch, _ = cache.get(s["image_id"], s["text"], False)
        praw = torch.from_numpy(patch).unsqueeze(0).to(dev)          # [1,T,768]

        box = perturb(gt, d_cells, rng)
        if t is not None:
            box = add_diffusion_noise(box, t, alphas, snr_scale, rng)
        r = sampler(praw, box.unsqueeze(0).to(dev))[0]               # [n,256]
        R.append(r.cpu())
        D.append(true_delta(box, gt))
        W.append(gt[:, 2] * GRID)
    return torch.cat(R), torch.cat(D), torch.cat(W)


def fit_and_score(r_tr, d_tr, r_te, d_te, w_te, epochs, lr, dev, seed):
    """Train `Linear(256->4)` trên tập train, chấm cosine trên tập test.

    Cosine chứ không phải MSE: cửa chặn hỏi về HƯỚNG. Một mô hình đoán đúng hướng nhưng
    sai độ lớn vẫn dùng được (6 tầng cộng dồn sẽ bù dần); đoán sai hướng thì không.
    """
    torch.manual_seed(seed)
    head = nn.Linear(r_tr.shape[1], 4).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    r_tr, d_tr = r_tr.to(dev), d_tr.to(dev)
    for _ in range(epochs):
        opt.zero_grad()
        nn.functional.mse_loss(head(r_tr), d_tr).backward()
        opt.step()

    with torch.no_grad():
        pred = head(r_te.to(dev)).cpu()
    # Chỉ trên 2 kênh TÂM: đó là "hướng dịch". Hai kênh kích thước là câu hỏi khác.
    cos = nn.functional.cosine_similarity(pred[:, :2], d_te[:, :2], dim=-1)

    small = w_te < 1.0                      # nhỏ hơn MỘT ô lưới
    mid = (w_te >= 1.0) & (w_te < 3.0)
    big = w_te >= 3.0
    grp = lambda m: float(cos[m].mean()) if m.any() else float("nan")  # noqa: E731
    return {
        "cosine": float(cos.mean()),
        "cosine_median": float(cos.median()),
        "pct_dung_huong": float((cos > 0).float().mean()),
        "cosine_box_nho": grp(small), "n_box_nho": int(small.sum()),
        "cosine_box_vua": grp(mid), "n_box_vua": int(mid.sum()),
        "cosine_box_to": grp(big), "n_box_to": int(big.sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/round2_experiment_a.yaml")
    ap.add_argument("--cache", required=True, help="thư mục cache patch token")
    ap.add_argument("--split", default="val",
                    help="val: 28 lớp rời train, không có lô annotation rác của test")
    ap.add_argument("--d-cells", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--timesteps", type=int, nargs="+", default=[249, 499, 749, 999],
                    help="-1 nghĩa là KHÔNG thêm nhiễu khuếch tán")
    ap.add_argument("--max-images", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="gate_delta.json")
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from utils.diffusion_math import cosine_alphas_cumprod
    alphas = cosine_alphas_cumprod(cfg["diffusion"]["num_timesteps"]).to(torch.float32)
    snr = cfg["diffusion"]["snr_scale"]

    ds = CE130Detection(cfg["data"]["root"], a.split, cfg["data"]["image_size"])
    cache = PatchCache(a.cache, a.split)
    torch.manual_seed(a.seed)
    sampler = RoIFeatureSampler(768, cfg["model"]["d_model"],
                                cfg["model"]["roi_k"], 0.0).to(dev).eval()
    # `roi.out` vốn zero-init để thí nghiệm bắt đầu từ 0 — ở đây phải khởi tạo thường,
    # nếu không `r` luôn bằng 0 và cửa chặn đo trên một hằng số.
    nn.init.xavier_uniform_(sampler.out.weight)
    nn.init.zeros_(sampler.out.bias)

    print(f"split={a.split}  ảnh={min(len(ds), a.max_images)}  thiết bị={dev}", flush=True)
    print(f"{'d(ô)':>6} {'t':>5} {'cosine':>8} {'trung vị':>9} {'%đúng':>7} "
          f"{'nhỏ<1ô':>8} {'vừa':>7} {'to':>7} {'xáo':>7} {'r=0':>7}", flush=True)

    res = {"config": vars(a), "rows": []}
    for d_cells in a.d_cells:
        for t in a.timesteps:
            tt = None if t < 0 else t
            r, d, w = collect(ds, cache, sampler, d_cells, tt, alphas, snr,
                              a.seed, dev, a.max_images)
            n = len(r)
            cut = int(n * 0.7)
            g = torch.Generator().manual_seed(a.seed)
            perm = torch.randperm(n, generator=g)
            tr, te = perm[:cut], perm[cut:]

            real = fit_and_score(r[tr], d[tr], r[te], d[te], w[te],
                                 a.epochs, a.lr, dev, a.seed)
            # ĐỐI CHỨNG 1: xáo `r` giữa các mẫu -> phá liên hệ ảnh<->delta.
            shuf = fit_and_score(r[tr][torch.randperm(cut, generator=g)], d[tr],
                                 r[te], d[te], w[te], a.epochs, a.lr, dev, a.seed)
            # ĐỐI CHỨNG 2: không có ảnh.
            zero = fit_and_score(torch.zeros_like(r[tr]), d[tr],
                                 torch.zeros_like(r[te]), d[te], w[te],
                                 a.epochs, a.lr, dev, a.seed)

            row = {"d_cells": d_cells, "t": t, "n_box": n,
                   "real": real, "shuffled": shuf, "zero": zero}
            res["rows"].append(row)
            print(f"{d_cells:6.1f} {t:5d} {real['cosine']:8.3f} "
                  f"{real['cosine_median']:9.3f} {real['pct_dung_huong']:7.2f} "
                  f"{real['cosine_box_nho']:8.3f} {real['cosine_box_vua']:7.3f} "
                  f"{real['cosine_box_to']:7.3f} {shuf['cosine']:7.3f} "
                  f"{zero['cosine']:7.3f}", flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print()
    print("  ĐỌC KẾT QUẢ:")
    print("    - TIÊU CHÍ: cosine > 0,5 ở d=1 ô. Dưới ngưỡng -> cộng dồn vô nghĩa, DỪNG.")
    print("    - 'xáo' và 'r=0' phải THẤP HƠN HẲN cột cosine. Nếu xấp xỉ nhau thì")
    print("      Linear chỉ học prior của delta, KHÔNG dùng ảnh -> kết quả vô giá trị.")
    print("    - 'nhỏ<1ô' dự kiến tệ nhất (18-29 % số box, RoI 3x3 thoái hoá thành 1x1).")
    print("      Nếu chỉ nhóm này hỏng thì thiết kế vẫn dùng được, chỉ giới hạn ở box to.")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
