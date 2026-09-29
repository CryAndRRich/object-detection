#!/usr/bin/env python3
"""Soi SpatialSoftmax của Diffusion Policy (Push-T ảnh, CNN hybrid, checkpoint công bố epoch 1850,
score 0,898) — đối chiếu với `inspect_spatial_softmax.py` của CE-Loc.

Ba hình, cùng 3 hàng × 2 cột (Original | SpatialSoftmax Output), crop giữa 84×84 = thứ encoder thấy:
  pusht_episode_XXX.png   data thật: khung ĐẦU / GIỮA / CUỐI của một episode
  goal_only.png           xoá khối T xám + agent; chữ T xanh (vùng đích) ở giữa / dời ↖ / dời ↘
  block_only.png          xoá chữ T xanh + agent; khối T xám ở trên-trái / giữa / dưới-phải
Cột 2 = 32 chấm, mỗi chấm là toạ độ kỳ vọng của MỘT keypoint; màu = số ô hiệu dụng của softmax
keypoint đó trên lưới 3×3 (vàng = nhọn, 1 ô; tím = trải đều 9 ô).

Nền sạch = median các khung đầu của 206 episode (khối, agent ở chỗ ngẫu nhiên nên bị lọc) -> chỉ còn
nền + chữ T xanh. Khối T xám cho block_only cắt từ khung đầu đầu tiên mà khối không chạm chữ T xanh
lẫn agent, dán lên nền đã xoá chữ T xanh.

Số đo (log + metrics.json; trọng số w_k = ||W_lin[:, 2k:2k+2]||): center_dist_px (khoảng cách chấm ->
tâm ảnh), frac_center_cell (tỉ lệ chấm nằm trong ô giữa), eff_cells (1..9).

  python tools/inspect_dp_spatial_softmax.py --episode 116 --out ../../output/spatial_softmax/diffusion_policy
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.legacy.dp_vision import CROP, RAW, grid_to_crop, load_dp_encoder, preprocess  # noqa: E402

FONT = {"family": "DejaVu Sans", "size": 12, "weight": "normal"}   # MỌI chữ trên hình, như CE-Loc
O = (RAW - CROP) // 2                  # lề crop giữa
GOAL_SHIFT = 24                        # pixel ảnh 96
BLOCK_CENTERS = {"top_left": (27, 27), "center": (48, 48), "bottom_right": (69, 69)}   # (ngang, dọc), ảnh 96
FIGURES = {
    "episode": (["start", "middle", "end"], {"start": "Start", "middle": "Middle", "end": "End"}),
    "goal_only": (["center", "up_left", "down_right"], {"center": "Center", "up_left": "Moved ↖",
                                                         "down_right": "Moved ↘"}),
    "block_only": (list(BLOCK_CENTERS), {"top_left": "Top-left", "center": "Center",
                                          "bottom_right": "Bottom-right"}),
}
TITLES = {"goal_only": "Goal only (block removed)", "block_only": "Block only (goal removed)"}


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


def dilate(mask, r=1):
    out = mask.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(mask, dy, 0), dx, 1)
    return out


def empty_scene(frames):
    """Median các khung [N,96,96,3] uint8 -> nền + chữ T xanh (khối, agent bị lọc)."""
    return np.median(frames, axis=0).round().astype(np.uint8)


def paste(bg, ys, xs, vals, dy, dx):
    out = bg.copy()
    ok = (ys + dy >= 0) & (ys + dy < bg.shape[0]) & (xs + dx >= 0) & (xs + dx < bg.shape[1])
    out[ys[ok] + dy, xs[ok] + dx] = vals[ok]
    return out


def remove_goal(scene):
    g = dilate(object_masks(scene)["goal"])             # nở 1 px để xoá cả viền khử răng cưa
    blank = scene.copy()
    blank[g] = 255
    return blank, g


def goal_scenes(scene, shift=GOAL_SHIFT):
    """Nền + chữ T xanh -> chữ T xanh ở giữa / dời ↖ / dời ↘ (viền khung giữ nguyên)."""
    blank, g = remove_goal(scene)
    ys, xs = np.nonzero(g)
    vals = scene[ys, xs]
    return {"center": scene, "up_left": paste(blank, ys, xs, vals, -shift, -shift),
            "down_right": paste(blank, ys, xs, vals, shift, shift)}


def block_sprite(frame):
    """Khung thật -> (ys, xs, màu) của khối T xám, hoặc None nếu khối chạm chữ T xanh / agent / mép."""
    m = object_masks(frame)
    if m["block"].sum() == 0 or (dilate(m["block"], 2) & (m["goal"] | m["agent"])).any():
        return None
    ys, xs = np.nonzero(m["block"])
    if ys.min() < 2 or xs.min() < 2 or ys.max() > RAW - 3 or xs.max() > RAW - 3:
        return None
    sel = dilate(m["block"])                             # lấy cả viền khử răng cưa
    ys, xs = np.nonzero(sel)
    return ys, xs, frame[ys, xs]


def block_scenes(scene, sprite, centers=BLOCK_CENTERS):
    """Nền đã xoá chữ T xanh + khối T xám dán sao cho tâm khối ở từng vị trí (ngang, dọc)."""
    blank, _ = remove_goal(scene)
    ys, xs, vals = sprite
    cy, cx = ys.mean(), xs.mean()
    return {k: paste(blank, ys, xs, vals, int(round(y - cy)), int(round(x - cx))) for k, (x, y) in centers.items()}


# ----------------------------------------------------------------------------- chạy

@torch.no_grad()
def encode(enc, imgs):
    """[N,96,96,3] uint8 -> keypoint pixel trên crop [N,K,2] (ngang, dọc), eff_cells [N,K]."""
    _, kp, att, _ = enc(preprocess(imgs))
    att = att.numpy()
    eff = np.exp(-(att * np.log(att + 1e-12)).sum(axis=(2, 3)))
    return grid_to_crop(kp.numpy(), att.shape[-1]), eff


def measure(kp, eff, w, names):
    third = CROP / 3
    rows = {}
    for i, name in enumerate(names):
        d = np.linalg.norm(kp[i] - CROP / 2, axis=-1)
        inside = ((kp[i] >= third) & (kp[i] <= 2 * third)).all(-1)
        rows[name] = {"center_dist_px": float((w * d).sum() / w.sum()),
                      "frac_center_cell": float((w * inside).sum() / w.sum()),
                      "eff_cells": float((w * eff[i]).sum() / w.sum())}
    return rows


# ----------------------------------------------------------------------------- vẽ

def plot_rows(imgs, kp, eff, labels, title, path):
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
    ext = [0, CROP, CROP, 0]
    fig, axes = plt.subplots(len(labels), 2, figsize=(2 * 3.2, len(labels) * 3.2), squeeze=False)
    for i, lab in enumerate(labels):
        rgb = imgs[i][O:O + CROP, O:O + CROP] / 255.0
        a0, a1 = axes[i]
        a0.imshow(rgb, extent=ext)
        a1.imshow(rgb * 0.45, extent=ext)
        order = np.argsort(-eff[i])                                      # chấm nhọn vẽ sau (nằm trên)
        sc = a1.scatter(kp[i, order, 0], kp[i, order, 1], s=30, c=eff[i, order], cmap="viridis_r",
                        norm=LogNorm(vmin=1, vmax=9), edgecolors="white", linewidths=0.3)
        a0.set_ylabel(lab)
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
    ap.add_argument("--episode", type=int, default=116, help="episode cho hình data thật")
    ap.add_argument("--no-ema", action="store_true", help="dùng weight model thường thay vì EMA")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
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
    firsts = np.stack([to_u8(img[int(s)]) for s in starts])
    scene = empty_scene(firsts)
    src = next((e for e in range(len(ends)) if block_sprite(firsts[e]) is not None), None)
    if src is None:
        raise RuntimeError("không tìm được khung đầu nào có khối T xám tách rời chữ T xanh và agent")
    split = episode_splits(len(ends))
    e = args.episode
    frames = [int(starts[e]), int((starts[e] + ends[e] - 1) // 2), int(ends[e] - 1)]
    print(f"[{time.time() - t0:5.1f}s] episode {e} ({split[e]}) khung {frames}; khối T xám cắt từ khung đầu "
          f"episode {src}", flush=True)

    sets = {"episode": np.stack([to_u8(img[f]) for f in frames]),
            "goal_only": np.stack(list(goal_scenes(scene).values())),
            "block_only": np.stack(list(block_scenes(scene, block_sprite(firsts[src])).values()))}
    files = {"episode": f"pusht_episode_{e:03d}.png", "goal_only": "goal_only.png", "block_only": "block_only.png"}
    result = dict(args=vars(args), checkpoint={k: str(v) for k, v in info.items()}, w_k=w.tolist(),
                  episode_info=dict(id=e, split=str(split[e]), frames=frames), block_source_episode=int(src))
    for name, ims in sets.items():
        keys, labels = FIGURES[name]
        kp, eff = encode(enc, ims)
        result[name] = measure(kp, eff, w, keys)
        for k in keys:
            r = result[name][k]
            print(f"  {name:>10} {k:>12} | cách tâm {r['center_dist_px']:5.1f}px | trong ô giữa "
                  f"{r['frac_center_cell']:.0%} | eff {r['eff_cells']:.2f}", flush=True)
        if not args.no_figures:
            title = f"Episode {e}" if name == "episode" else TITLES[name]
            plot_rows(ims, kp, eff, [labels[k] for k in keys], title, os.path.join(args.out, files[name]))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{time.time() - t0:5.1f}s] xong -> {args.out}")


if __name__ == "__main__":
    main()
