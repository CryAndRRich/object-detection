"""Helper vẽ dùng chung của `tools/plot_*` (bài add): box / lỗ, chấm SpatialSoftmax (màu = số ô hiệu dụng, thang log), dọn trục.
Chỉ đọc tensor / numpy, không nạp model."""

import numpy as np
import torch

from ce_localization.utils.box_ops_np import box_iou

SHARP = 16                                     # < 16 ô hiệu dụng = kênh "nhọn"


def keypoint_pixels(xy, H, T=512):
    """[C,2] (dọc, ngang) trong [−1, 1] -> (x_px, y_px) trên canvas T; −1 / +1 = tâm ô đầu / cuối của lưới H."""
    cell = T / H
    to_px = lambda u: cell / 2 + (u + 1) / 2 * (T - cell)  # noqa: E731
    return to_px(xy[:, 1]), to_px(xy[:, 0])


def effective_cells(att):
    """[C,H,W] softmax -> exp(entropy) [C]."""
    p = att.reshape(att.shape[0], -1).clamp_min(1e-12)
    return torch.exp(-(p * p.log()).sum(-1))


def dot_sizes(proj):
    """Linear(2C -> D) chiếu toạ độ SpatialSoftmax -> (cỡ chấm [C] ∝ ‖W[:, 2c:2c+2]‖, chuẩn ‖W‖ [C])."""
    w = proj.weight.detach().reshape(proj.out_features, -1, 2).norm(dim=(0, 2)).cpu().numpy()
    return 4 + 60 * w / w.max(), w


def rect(ax, b, color, lw=1.0, alpha=1.0, ls="-"):
    import matplotlib.patches as mp
    ax.add_patch(mp.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, linewidth=lw,
                              alpha=alpha, linestyle=ls))


def sorted_box(b):
    """xyxy có thể w, h âm (box nhiễu không kẹp) -> xyxy (min, max) để vẽ / chấm."""
    b = np.asarray(b, dtype=np.float64)
    return np.array([min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])])


def center(b):
    b = np.asarray(b)
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], -1)


def iou_to(b, gts):
    """IoU box xyxy (toạ độ âm kẹp về 0) với mỗi box gts [M,4] -> [M]."""
    return box_iou(np.asarray(b, dtype=np.float64)[None].clip(min=0), np.asarray(gts, dtype=np.float64).reshape(-1, 4))[0][0]


def draw_holes(ax, holes):
    """Lỗ đã xoá: lỗ mới nhất (đích) đậm + dấu tâm, lỗ cũ nét đứt."""
    for hb in holes[:-1]:
        rect(ax, hb, "lime", lw=1.0, ls="--")
    rect(ax, holes[-1], "lime", lw=1.6)
    ax.plot(*center(holes[-1]), "+", color="lime", ms=10, mew=2)


def ss_scatter(fig, ax, bg, xy, att, size, T=512, label=None):
    """Chấm SpatialSoftmax lên ảnh bg: mỗi chấm một kênh, vị trí = toạ độ kỳ vọng, màu = số ô hiệu dụng (log), cỡ = `size`.
    -> (eff [C], px, py, tỉ lệ kênh nhọn)."""
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import FuncFormatter, NullFormatter
    eff = effective_cells(att).numpy()
    px, py = keypoint_pixels(xy.numpy(), att.shape[1], T)
    ax.imshow(bg)
    order = np.argsort(-eff)                                       # chấm nhọn vẽ sau cùng (nằm trên)
    sc = ax.scatter(px[order], py[order], c=eff[order], s=size[order], cmap="plasma_r",
                    norm=LogNorm(vmin=1, vmax=att.shape[1] * att.shape[2]), edgecolors="black", linewidths=0.3, alpha=0.9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02, **({"label": label} if label else {}))
    # nhãn trục log dạng số thường ("1", "10", "100"), KHÔNG mathtext "$10^{k}$": matplotlib cũ + pyparsing ≥ 3.3 (Kaggle) cảnh báo
    # PyparsingDeprecationWarning mỗi lần dựng / parse mathtext
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    cb.ax.yaxis.set_minor_formatter(NullFormatter())
    return eff, px, py, float((eff < SHARP).mean())


def tidy(axes, w, h):
    """Khung ảnh [0, w] × [0, h] (trục y xuống), bỏ vạch trục."""
    for ax in axes:
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
