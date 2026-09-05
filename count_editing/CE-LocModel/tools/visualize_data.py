#!/usr/bin/env python3
"""Vẽ box qua ĐÚNG chuỗi 6 bước lên canvas đã pad. Chỉ cần PIL + numpy.

Chạy TRƯỚC khi viết model. Vòng 1 visualize bắt được 3 lỗi mà toàn bộ test và 3
vòng rà soát code bỏ sót — nhìn ảnh là cách kiểm duy nhất bắt được lỗi về BẢN
CHẤT dữ liệu (loss chỉ tính trên cặp đã match nên không thấy 62 % placeholder).

  python3 tools/visualize_data.py --n 20 --out <thư mục>
  python3 tools/visualize_data.py --split test --placeholder --n 8
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection  # noqa: E402
from utils.box_ops_np import cxcywh_to_xyxy, decode_diffusion, encode_diffusion  # noqa: E402
from utils.diffusion_np import cosine_alphas_cumprod, prepare_diffusion_concat  # noqa: E402

XANH, DO, VANG, CAM = (0, 255, 0), (255, 60, 60), (255, 255, 0), (255, 150, 0)


def ve(mau, snr_scale=2.0, ve_placeholder=False, num_proposals=100, t=None, rng=None):
    """Trả về ảnh PIL đã vẽ. Box đi qua encode -> decode để kiểm cả chuỗi."""
    img = Image.fromarray(mau["image"])
    dr = ImageDraw.Draw(img)
    T = img.size[0]

    if ve_placeholder:
        ab = cosine_alphas_cumprod(1000)
        tt = int(rng.integers(0, 1000)) if t is None else t
        x_t, _, is_gt = prepare_diffusion_concat(
            mau["boxes"], num_proposals, tt, ab, snr_scale,
            valid_h=mau["valid_h"], rng=rng,
        )
        for b, thuc in zip(cxcywh_to_xyxy(decode_diffusion(x_t, snr_scale)) * T, is_gt):
            dr.rectangle(b.tolist(), outline=XANH if thuc else CAM, width=2)
    else:
        # GT qua trọn chuỗi: cxcywh[0,1] -> encode -> decode -> xyxy px
        back = decode_diffusion(encode_diffusion(mau["boxes"], snr_scale), snr_scale)
        for b in cxcywh_to_xyxy(back) * T:
            dr.rectangle(b.tolist(), outline=XANH, width=2)

    nh = int(round(mau["valid_h"] * T))
    if nh < T - 1:                                   # ranh giới vùng pad
        dr.line([(0, nh), (T, nh)], fill=VANG, width=3)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="../../data/all_phase2_V2")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=20, help="số ảnh; -1 = tất cả")
    ap.add_argument("--out", required=True)
    ap.add_argument("--placeholder", action="store_true", help="vẽ cả box giả")
    ap.add_argument("--num-proposals", type=int, default=100)
    ap.add_argument("--t", type=int, default=None, help="timestep cố định")
    ap.add_argument("--snr-scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ds = CE130Detection(a.root, a.split)
    print(f"{a.split}: {ds.thong_ke()}")

    idx = range(len(ds)) if a.n < 0 else np.linspace(0, len(ds) - 1, min(a.n, len(ds))).astype(int)
    dong = []
    for i in idx:
        m = ds[int(i)]
        img = ve(m, a.snr_scale, a.placeholder, a.num_proposals, a.t, rng)
        ten = f"{a.split}_{m['image_id']}_{m['text'].replace(' ', '-')}.png"
        img.save(os.path.join(a.out, ten))
        dong.append(f"{ten},{len(m['boxes'])},{m['valid_h']:.4f},{m['orig_size'][0]}x{m['orig_size'][1]}")

    with open(os.path.join(a.out, "index.csv"), "w") as f:
        f.write("file,so_box,valid_h,kich_thuoc_goc\n" + "\n".join(dong) + "\n")
    print(f"đã lưu {len(dong)} ảnh vào {a.out}")


if __name__ == "__main__":
    main()
