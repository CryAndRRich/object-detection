#!/usr/bin/env python3
"""Overfit MỘT ảnh — cửa chặn trước khi train dài.

Nếu loss không về gần 0 thì có lỗi ở matcher hoặc loss, DỪNG SỬA NGAY, đừng train
tiếp. Vòng 1 train 5 lần liên tiếp trên code có lỗi vì thiếu đúng bước này.

Dùng t nhỏ cố định để tách bạch: ở t lớn box đầu vào gần như thuần nhiễu nên
không overfit được, và đó KHÔNG phải lỗi.

  python3 tools/overfit_one.py --steps 300 --device cpu
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import CE130Detection, normalize_for_clip  # noqa: E402
from models.detector import CELocDetector  # noqa: E402
from models.criterion import SetCriterion  # noqa: E402
from utils.diffusion_math import prepare_diffusion_concat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--t", type=int, default=50, help="timestep cố định (nhỏ)")
    ap.add_argument("--num-proposals", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)

    ds = CE130Detection(cfg["data"]["root"], "train", cfg["data"]["image_size"])
    m = ds[a.index]
    px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
    gt = torch.from_numpy(m["boxes"]).float().to(dev)
    print(f"[ảnh] {m['image_id']} '{m['text']}' | {len(gt)} GT | N={a.num_proposals} "
          f"| t={a.t}", flush=True)

    model = CELocDetector(
        cfg["model"]["clip_name"], cfg["model"]["d_model"], cfg["model"]["n_layer"],
        cfg["model"]["n_head"], cfg["data"]["image_size"],
        cfg["diffusion"]["num_timesteps"], cfg["diffusion"]["snr_scale"],
        cfg["diffusion"]["sampling_steps"], 0.0, cfg["model"]["freeze_clip"],
    ).to(dev)
    model.train()
    crit = SetCriterion(cfg["matcher"]["method"])
    hoc = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(hoc, lr=a.lr)

    with torch.no_grad():                          # CLIP frozen -> encode 1 lần
        patch_raw = model.encoder.encode_image_raw(px)
        text_raw = model.encoder.encode_text_raw([m["text"]], dev)

    g = torch.Generator(device="cpu").manual_seed(0)
    x_t, _, _ = prepare_diffusion_concat(gt.cpu(), a.num_proposals, a.t,
                                         model.alphas_cumprod.cpu(),
                                         cfg["diffusion"]["snr_scale"],
                                         valid_h=m["valid_h"], generator=g)
    x_t = x_t.unsqueeze(0).to(dev)
    tt = torch.full((1,), a.t, dtype=torch.long, device=dev)

    dau = None
    for i in range(a.steps):
        pb, lg = model(x_t, tt, patch_raw=patch_raw, text_raw=text_raw)
        loss, st, _ = crit(pb, lg, [gt])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if dau is None:
            dau = st["loss"]
        if i % max(a.steps // 10, 1) == 0 or i == a.steps - 1:
            print(f"  [{i:4d}] loss {st['loss']:7.4f} | l1 {st['loss_l1']:.4f} "
                  f"giou {st['loss_giou']:.4f} ce {st['loss_ce']:.4f} | "
                  f"IoU {st['iou_matched']:.4f}", flush=True)

    ti_le = st["loss"] / max(dau, 1e-9)
    print(f"\nloss {dau:.4f} -> {st['loss']:.4f} (còn {100*ti_le:.1f} %) | "
          f"IoU_matched {st['iou_matched']:.4f}")
    if st["iou_matched"] > 0.7 and ti_le < 0.3:
        print("✓ ĐẠT — matcher và loss hoạt động đúng, có thể train tiếp")
    else:
        print("✗ KHÔNG ĐẠT — có lỗi ở matcher hoặc loss. DỪNG, đừng train dài.")


if __name__ == "__main__":
    main()
