#!/usr/bin/env python3
"""Soi SpatialSoftmax của Diffusion Policy (Push-T ảnh, CNN hybrid, checkpoint công bố epoch 1850,
score 0,898) — đối chiếu với `inspect_spatial_softmax.py` của CE-Loc.

Mỗi episode một hình 3 hàng (khung ĐẦU / GIỮA / CUỐI của episode) × 2 cột
(Original | SpatialSoftmax Output). Ảnh vẽ là crop giữa 84×84 — đúng thứ encoder thấy khi eval.
Cột 2 = 32 chấm, mỗi chấm là toạ độ kỳ vọng của MỘT keypoint; màu = số ô hiệu dụng của softmax
keypoint đó trên lưới 3×3 (vàng = nhọn, 1 ô; tím = trải đều 9 ô, chấm bị kéo về giữa khung).

Số đo (log + metrics.json; trung bình trên 32 keypoint, trọng số w_k = ||W_lin[:, 2k:2k+2]||):
  lift_<vật>      khối lượng attention trên ô phủ vật / tỉ lệ diện tích vật (1 = ngẫu nhiên);
                  vật tách theo màu render: block (khối T, xám), agent (tròn, xanh dương),
                  goal (vùng đích, xanh lá)
  eff_cells       số ô hiệu dụng exp(entropy), 1..9
  frac_localized  tỉ lệ (trọng số) keypoint có < 2 ô hiệu dụng

Episode: split theo đúng config (val_ratio 0,02, max_train_episodes 90, seed 42) -> train / val /
unused (không vào train).

  python tools/inspect_dp_spatial_softmax.py --n 2 --out ../../output/spatial_softmax/diffusion_policy
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
OBJECTS = ["block", "agent", "goal"]
LOCALIZED = 2                          # keypoint "nhọn" nếu < 2 ô hiệu dụng (lưới chỉ 9 ô)
FONT = {"family": "DejaVu Sans", "size": 12, "weight": "normal"}   # MỌI chữ trên hình, như CE-Loc


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


def coverage(mask, n_cell):
    c = CROP // n_cell
    return mask.astype(np.float32).reshape(n_cell, c, n_cell, c).mean(axis=(1, 3))


def lift(att, w, cov):
    chance = cov.mean()
    if chance == 0:
        return float("nan")
    mass = (att * cov).sum(axis=(1, 2))
    return float((w * mass).sum() / w.sum() / chance)


# ----------------------------------------------------------------------------- chạy

@torch.no_grad()
def run_episode(enc, imgs, w):
    """imgs [3,96,96,3] uint8 (đầu/giữa/cuối) -> số đo + keypoint pixel trên crop."""
    _, kp, att, _ = enc(preprocess(imgs))
    kp, att = kp.numpy(), att.numpy()
    n_cell = att.shape[-1]
    eff = np.exp(-(att * np.log(att + 1e-12)).sum(axis=(2, 3)))               # [3, K]
    o = (RAW - CROP) // 2
    rows = {}
    for i, name in enumerate(ROWS):
        masks = object_masks(imgs[i][o:o + CROP, o:o + CROP])
        r = {f"lift_{k}": lift(att[i], w, coverage(m, n_cell)) for k, m in masks.items()}
        r.update({f"area_{k}": float(m.mean()) for k, m in masks.items()})
        r["eff_cells"] = float((w * eff[i]).sum() / w.sum())
        r["frac_localized"] = float((w * (eff[i] < LOCALIZED)).sum() / w.sum())
        rows[name] = r
    return dict(rows=rows, kp=grid_to_crop(kp, n_cell), eff=eff)


# ----------------------------------------------------------------------------- vẽ

def plot_episode(imgs, res, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({"font.family": FONT["family"], "font.size": FONT["size"],
                         "font.weight": FONT["weight"], "axes.titleweight": FONT["weight"],
                         "axes.labelweight": FONT["weight"], "figure.titleweight": FONT["weight"],
                         "axes.titlesize": FONT["size"], "axes.labelsize": FONT["size"],
                         "figure.titlesize": FONT["size"], "xtick.labelsize": FONT["size"],
                         "ytick.labelsize": FONT["size"]})
    o = (RAW - CROP) // 2
    ext = [0, CROP, CROP, 0]
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(2 * 3.2, len(ROWS) * 3.2))
    for i, name in enumerate(ROWS):
        rgb = imgs[i][o:o + CROP, o:o + CROP] / 255.0
        a0, a1 = axes[i]
        a0.imshow(rgb, extent=ext)
        a1.imshow(rgb * 0.45, extent=ext)
        order = np.argsort(-res["eff"][i])                               # chấm nhọn vẽ sau (nằm trên)
        sc = a1.scatter(res["kp"][i, order, 0], res["kp"][i, order, 1], s=30, c=res["eff"][i, order],
                        cmap="viridis_r", norm=LogNorm(vmin=1, vmax=9), edgecolors="white", linewidths=0.3)
        a0.set_ylabel(ROW_LABELS[name])
        for a in (a0, a1):
            a.set_xlim(0, CROP)
            a.set_ylim(CROP, 0)
            a.set_xticks([])
            a.set_yticks([])
    for a, t in zip(axes[0], ["Original", "SpatialSoftmax Output"]):
        a.set_title(t)
    fig.subplots_adjust(top=1 - 0.9 / fig.get_figheight(), right=0.84, wspace=0.05, hspace=0.08)
    cb = fig.colorbar(sc, cax=fig.add_axes([0.87, 0.3, 0.025, 0.4]))
    cb.set_ticks([1, 2, 3, 5, 9], labels=["1", "2", "3", "5", "9"])
    cb.minorticks_off()
    cb.set_label("Effective cells")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ema", action="store_true", help="dùng weight model thường thay vì EMA")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    warnings.filterwarnings("ignore", message="All-NaN slice")    # vật vắng khỏi mọi khung -> lift NaN
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    enc, info = load_dp_encoder(args.ckpt, use_ema=not args.no_ema)
    print(f"[{time.time() - t0:5.1f}s] checkpoint: {info}", flush=True)
    W = enc.nets[3].weight.detach().numpy()                              # [D, 2K]
    w = np.linalg.norm(W.reshape(W.shape[0], -1, 2), axis=(0, 2))        # [K]

    root = zarr.open(args.zarr, "r")
    ends = root["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    split = episode_splits(len(ends))
    eps = args.episodes if args.episodes else sorted(
        np.random.default_rng(args.seed).permutation(len(ends))[: args.n].tolist())
    print(f"[{time.time() - t0:5.1f}s] {len(ends)} episode (train {np.sum(split == 'train')}, "
          f"val {np.sum(split == 'val')}, unused {np.sum(split == 'unused')}); soi {len(eps)}", flush=True)

    per_ep = []
    for k, e in enumerate(eps):
        frames = [int(starts[e]), int((starts[e] + ends[e] - 1) // 2), int(ends[e] - 1)]
        imgs = np.stack([root["data/img"][f] for f in frames]).round().clip(0, 255).astype(np.uint8)
        res = run_episode(enc, imgs, w)
        per_ep.append(dict(episode=int(e), split=str(split[e]), frames=frames, rows=res["rows"]))
        if not args.no_figures:
            plot_episode(imgs, res, f"Episode {e} ({split[e]})", os.path.join(args.out, f"episode_{e:03d}.png"))
        el = time.time() - t0
        r = res["rows"]
        print(f"[{el:5.1f}s | ETA {el / (k + 1) * (len(eps) - k - 1):4.0f}s] {k + 1}/{len(eps)} ep {e:3d} "
              f"{split[e]:>6} | " + " | ".join(
                  f"{n} block {r[n]['lift_block']:.2f} agent {r[n]['lift_agent']:.2f} goal {r[n]['lift_goal']:.2f} "
                  f"eff {r[n]['eff_cells']:.2f}" for n in ROWS), flush=True)

    cols = ["lift_block", "lift_agent", "lift_goal", "eff_cells", "frac_localized"]
    print("\nTRUNG VỊ trên các episode (lift: 1 = ngẫu nhiên):")
    print(f"{'row':>8} " + " ".join(f"{c:>14}" for c in cols))
    summary = {}
    for n in ROWS:
        summary[n] = {c: float(np.nanmedian([p["rows"][n][c] for p in per_ep])) for c in cols}
        print(f"{n:>8} " + " ".join(f"{summary[n][c]:14.3f}" for c in cols))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(dict(args=vars(args), checkpoint={k: str(v) for k, v in info.items()}, w_k=w.tolist(),
                       median=summary, episodes=per_ep), f, indent=1)
    print(f"[{time.time() - t0:5.1f}s] xong -> {args.out}")


if __name__ == "__main__":
    main()
