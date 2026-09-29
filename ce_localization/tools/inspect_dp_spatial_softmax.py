#!/usr/bin/env python3
"""Soi SpatialSoftmax của Diffusion Policy (Push-T ảnh, CNN hybrid, checkpoint công bố epoch 1850,
score 0,898) — đối chiếu với `inspect_spatial_softmax.py` của CE-Loc.

Vì sao cần MỐC: vùng đích (chữ T xanh) luôn ở giữa khung, và 32 keypoint (lưới layer4 chỉ 3×3)
mặc định cũng tụ ở giữa khung -> nhìn toạ độ tuyệt đối không phân biệt được "nhìn chữ T xanh" với
"mặc định ở giữa". Mốc = KHUNG TRỐNG: median các khung đầu của 206 episode (khối T và agent ở chỗ
ngẫu nhiên nên bị lọc) -> chỉ còn nền + chữ T xanh.

Hai loại hình, cùng 2 cột (Original | SpatialSoftmax Output), crop giữa 84×84 = thứ encoder thấy:
  controls.png        4 hàng: khung trống / xoá chữ T xanh / dời chữ T xanh lên-trái / xuống-phải.
                      Chấm đi theo chữ T xanh -> encoder nhìn nó; đứng yên -> chỉ là vị trí mặc định.
  episode_XXX.png     3 hàng: khung ĐẦU / GIỮA / CUỐI của episode.
Cột 2: vòng rỗng xám = vị trí keypoint trên khung trống (mốc), chấm = vị trí trên ảnh này, vạch nối
hai vị trí; màu chấm = độ dịch so với mốc (pixel crop 84). Vạch chĩa về khối T xám = keypoint đó
mã hoá khối.

Số đo (log + metrics.json; trọng số w_k = ||W_lin[:, 2k:2k+2]||):
  shift_px        độ dịch trung bình so với mốc (pixel crop 84)
  toward_block    cos giữa hướng dịch và hướng mốc -> tâm khối T xám, trọng số w_k·|dịch|
                  (1 = mọi chấm dịch thẳng về khối, 0 = hướng ngẫu nhiên); toward_agent tương tự
  eff_cells       số ô hiệu dụng exp(entropy) của softmax, 1..9

Episode: split theo đúng config (val_ratio 0,02, max_train_episodes 90, seed 42) -> train / val /
unused (không vào train).

  python tools/inspect_dp_spatial_softmax.py --episodes 116 --controls --out ../../output/spatial_softmax/diffusion_policy
  python tools/inspect_dp_spatial_softmax.py --n 20 --out ../../output/spatial_softmax/diffusion_policy
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.legacy.dp_vision import CROP, RAW, grid_to_crop, load_dp_encoder, preprocess  # noqa: E402

ROWS = ["start", "middle", "end"]
ROW_LABELS = {"start": "Start", "middle": "Middle", "end": "End"}
CONTROLS = ["goal", "no_goal", "goal_up_left", "goal_down_right"]
CONTROL_LABELS = {"goal": "Empty scene", "no_goal": "Goal removed", "goal_up_left": "Goal moved ↖",
                  "goal_down_right": "Goal moved ↘"}
CONTROL_SHIFT = 24                     # pixel ảnh 96
MAX_SHIFT = 20                         # thang màu độ dịch (pixel crop 84)
FONT = {"family": "DejaVu Sans", "size": 12, "weight": "normal"}   # MỌI chữ trên hình, như CE-Loc
O = (RAW - CROP) // 2                  # lề crop giữa


# ----------------------------------------------------------------------------- dữ liệu

def episode_splits(n_episodes, val_ratio=0.02, max_train=90, seed=42):
    """Viết lại `get_val_mask` + `downsample_mask` của diffusion_policy/common/sampler.py."""
    val = np.zeros(n_episodes, bool)
    if val_ratio > 0:
        n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
        val[np.random.default_rng(seed=seed).choice(n_episodes, size=n_val, replace=False)] = True
    train = ~val
    if max_train is not None and train.sum() > max_train:
        idx = np.nonzero(train)[0]
        keep = idx[np.random.default_rng(seed=seed).choice(len(idx), size=int(max_train), replace=False)]
        train = np.zeros_like(train)
        train[keep] = True
    return np.where(train, "train", np.where(val, "val", "unused"))


def object_masks(img):
    """Ảnh render [H,W,3] uint8 -> mask theo màu: block LightSlateGray, agent RoyalBlue, goal LightGreen."""
    r, g, b = [img[..., i].astype(int) for i in range(3)]
    agent = (b > 180) & (r < 110)
    goal = (g - r >= 50) & (g - b >= 50)
    block = (b - r >= 25) & (b < 200) & ~agent
    return {"block": block, "agent": agent, "goal": goal}


def empty_scene(frames):
    """Median các khung [N,96,96,3] uint8 -> nền + chữ T xanh (khối, agent bị lọc)."""
    return np.median(frames, axis=0).round().astype(np.uint8)


def _dilate(mask, r=1):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def control_scenes(scene, shift=CONTROL_SHIFT):
    """Khung trống -> {tên: ảnh}: giữ / xoá / dời chữ T xanh (chỉ chữ T xanh, viền khung giữ nguyên)."""
    g = _dilate(object_masks(scene)["goal"])            # nở 1 px để xoá cả viền khử răng cưa
    blank = scene.copy()
    blank[g] = 255
    out = {"goal": scene, "no_goal": blank}
    ys, xs = np.nonzero(g)
    for name, d in (("goal_up_left", -shift), ("goal_down_right", shift)):
        im = blank.copy()
        ok = (ys + d >= 0) & (ys + d < RAW) & (xs + d >= 0) & (xs + d < RAW)
        im[ys[ok] + d, xs[ok] + d] = scene[ys[ok], xs[ok]]
        out[name] = im
    return out


# ----------------------------------------------------------------------------- chạy

@torch.no_grad()
def encode(enc, imgs):
    """[N,96,96,3] uint8 -> keypoint pixel trên crop [N,K,2] (ngang, dọc), eff_cells [N,K]."""
    _, kp, att, _ = enc(preprocess(imgs))
    att = att.numpy()
    eff = np.exp(-(att * np.log(att + 1e-12)).sum(axis=(2, 3)))
    return grid_to_crop(kp.numpy(), att.shape[-1]), eff


def toward(base, kp, target, w):
    """cos(hướng dịch, hướng mốc -> target), trọng số w·|dịch|. target (ngang, dọc) pixel crop."""
    d = kp - base
    v = np.asarray(target)[None] - base
    n = np.linalg.norm(d, axis=-1)
    cos = (d * v).sum(-1) / (n * np.linalg.norm(v, axis=-1) + 1e-9)
    return float((w * n * cos).sum() / ((w * n).sum() + 1e-12))


def centroid(mask):
    ys, xs = np.nonzero(mask)
    return None if len(xs) == 0 else (xs.mean() + 0.5, ys.mean() + 0.5)


def measure(imgs, kp, base, eff, w, names):
    rows = {}
    for i, name in enumerate(names):
        masks = object_masks(imgs[i][O:O + CROP, O:O + CROP])
        r = {"shift_px": float((w * np.linalg.norm(kp[i] - base, axis=-1)).sum() / w.sum()),
             "eff_cells": float((w * eff[i]).sum() / w.sum())}
        for obj in ("block", "agent", "goal"):
            c = centroid(masks[obj])
            r[f"toward_{obj}"] = float("nan") if c is None else toward(base, kp[i], c, w)
        rows[name] = r
    return rows


# ----------------------------------------------------------------------------- vẽ

def plot_rows(imgs, kp, base, labels, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": FONT["family"], "font.size": FONT["size"],
                         "font.weight": FONT["weight"], "axes.titleweight": FONT["weight"],
                         "axes.labelweight": FONT["weight"], "figure.titleweight": FONT["weight"],
                         "axes.titlesize": FONT["size"], "axes.labelsize": FONT["size"],
                         "figure.titlesize": FONT["size"], "xtick.labelsize": FONT["size"],
                         "ytick.labelsize": FONT["size"], "legend.fontsize": FONT["size"]})
    ext = [0, CROP, CROP, 0]
    n = len(labels)
    fig, axes = plt.subplots(n, 2, figsize=(2 * 3.2, n * 3.2), squeeze=False)
    for i, lab in enumerate(labels):
        rgb = imgs[i][O:O + CROP, O:O + CROP] / 255.0
        a0, a1 = axes[i]
        a0.imshow(rgb, extent=ext)
        a1.imshow(rgb * 0.45, extent=ext)
        shift = np.linalg.norm(kp[i] - base, axis=-1)
        for b, p in zip(base, kp[i]):
            a1.plot([b[0], p[0]], [b[1], p[1]], color="white", lw=0.7, alpha=0.8, zorder=2)
        ring = a1.scatter(base[:, 0], base[:, 1], s=30, facecolors="none", edgecolors="0.8", linewidths=0.8,
                          zorder=3, label="Empty scene")
        order = np.argsort(shift)                                        # chấm dịch nhiều vẽ sau
        sc = a1.scatter(kp[i, order, 0], kp[i, order, 1], s=30, c=shift[order], cmap="viridis",
                        vmin=0, vmax=MAX_SHIFT, edgecolors="white", linewidths=0.3, zorder=4, label="This image")
        a0.set_ylabel(lab)
        for a in (a0, a1):
            a.set_xlim(0, CROP)
            a.set_ylim(CROP, 0)
            a.set_xticks([])
            a.set_yticks([])
    for a, t in zip(axes[0], ["Original", "SpatialSoftmax Output"]):
        a.set_title(t)
    fig.subplots_adjust(top=1 - 0.9 / fig.get_figheight(), bottom=0.7 / fig.get_figheight(), right=0.84,
                        wspace=0.05, hspace=0.08)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.87, 0.3, 0.025, 0.4]))
    cb.set_label("Shift from empty scene (px)")
    fig.legend(handles=[ring, sc], loc="lower center", ncol=2, frameon=False)
    fig.suptitle(title)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    import zarr
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../weights/diffusion_policy/epoch=1850-test_mean_score=0.898.ckpt")
    ap.add_argument("--zarr", default="../data/pusht/pusht_cchi_v7_replay.zarr")
    ap.add_argument("--n", type=int, default=20, help="số episode (chọn ngẫu nhiên theo --seed)")
    ap.add_argument("--episodes", nargs="*", type=int, help="chỉ định episode, vd 0 17")
    ap.add_argument("--controls", action="store_true", help="vẽ thêm controls.png (dời / xoá chữ T xanh)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true", help="dùng weight model thường thay vì EMA")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="All-NaN slice")    # vật vắng khỏi mọi khung -> NaN
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    enc, info = load_dp_encoder(args.ckpt, use_ema=not args.no_ema)
    print(f"[{time.time() - t0:5.1f}s] checkpoint: {info}", flush=True)
    W = enc.nets[3].weight.detach().numpy()                              # [D, 2K]
    w = np.linalg.norm(W.reshape(W.shape[0], -1, 2), axis=(0, 2))        # [K]

    root = zarr.open(args.zarr, "r")
    ends = root["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    img = root["data/img"]
    to_u8 = lambda a: np.asarray(a).round().clip(0, 255).astype(np.uint8)   # noqa: E731
    scene = empty_scene(np.stack([to_u8(img[int(s)]) for s in starts]))
    base = encode(enc, scene[None])[0][0]                                # [K, 2] mốc
    split = episode_splits(len(ends))
    eps = args.episodes if args.episodes is not None else sorted(
        np.random.default_rng(args.seed).permutation(len(ends))[: args.n].tolist())
    print(f"[{time.time() - t0:5.1f}s] {len(ends)} episode (train {np.sum(split == 'train')}, "
          f"val {np.sum(split == 'val')}, unused {np.sum(split == 'unused')}); soi {len(eps)}", flush=True)

    result = dict(args=vars(args), checkpoint={k: str(v) for k, v in info.items()}, w_k=w.tolist(),
                  base_kp=base.tolist())
    if args.controls:
        sc = control_scenes(scene)
        ims = np.stack([sc[k] for k in CONTROLS])
        kp, eff = encode(enc, ims)
        result["controls"] = measure(ims, kp, base, eff, w, CONTROLS)
        for k in CONTROLS:
            r = result["controls"][k]
            print(f"  control {k:>16} | shift {r['shift_px']:5.1f}px toward_goal {r['toward_goal']:+.2f} "
                  f"eff {r['eff_cells']:.2f}", flush=True)
        if not args.no_figures:
            plot_rows(ims, kp, base, [CONTROL_LABELS[k] for k in CONTROLS], "Goal controls",
                      os.path.join(args.out, "controls.png"))

    per_ep = []
    for k, e in enumerate(eps):
        frames = [int(starts[e]), int((starts[e] + ends[e] - 1) // 2), int(ends[e] - 1)]
        ims = np.stack([to_u8(img[f]) for f in frames])
        kp, eff = encode(enc, ims)
        rows = measure(ims, kp, base, eff, w, ROWS)
        per_ep.append(dict(episode=int(e), split=str(split[e]), frames=frames, rows=rows))
        if not args.no_figures:
            plot_rows(ims, kp, base, [ROW_LABELS[r] for r in ROWS], f"Episode {e} ({split[e]})",
                      os.path.join(args.out, f"episode_{e:03d}.png"))
        el = time.time() - t0
        print(f"[{el:5.1f}s | ETA {el / (k + 1) * (len(eps) - k - 1):4.0f}s] {k + 1}/{len(eps)} ep {e:3d} "
              f"{split[e]:>6} | " + " | ".join(
                  f"{n} shift {rows[n]['shift_px']:4.1f}px block {rows[n]['toward_block']:+.2f} "
                  f"agent {rows[n]['toward_agent']:+.2f}" for n in ROWS), flush=True)

    cols = ["shift_px", "toward_block", "toward_agent", "toward_goal", "eff_cells"]
    if per_ep:
        print("\nTRUNG VỊ trên các episode (toward: 1 = dịch thẳng về vật, 0 = ngẫu nhiên):")
        print(f"{'row':>8} " + " ".join(f"{c:>13}" for c in cols))
        summary = {}
        for n in ROWS:
            summary[n] = {c: float(np.nanmedian([p["rows"][n][c] for p in per_ep])) for c in cols}
            print(f"{n:>8} " + " ".join(f"{summary[n][c]:13.3f}" for c in cols))
        result.update(median=summary, episodes=per_ep)
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{time.time() - t0:5.1f}s] xong -> {args.out}")


if __name__ == "__main__":
    main()
