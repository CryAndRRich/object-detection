"""BETA (docs/EXPERIMENT_BETA.md mục 1, 3): nhãn ĐIỂM lấy từ density, cỡ giả kNN, box giả.

Density CountGD = blob Gaussian cùng cỡ đặt tại mỗi điểm vật (không mang thông tin cỡ) ->
đỉnh cục bộ của mức density (giải mã jet bằng `alpha/density.py`) là điểm vật. Box giả
`(p, ŝ, ŝ)` chỉ để dựng x_start khuếch tán và phần HÌNH HỌC của matcher (prior, dynamic k);
loss của BETA đọc tâm = điểm và cỡ = ŝ từ chính box giả (`alpha/criterion.py`, chế độ `point`).

Quy ước toạ độ: pixel liên tục, pixel (hàng i, cột j) phủ [j, j+1] × [i, i+1] -> tâm pixel là
(j + 0.5, i + 0.5), cùng hệ với box xyxy pixel của CE-130.
"""

import json
import os

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

__all__ = ["find_peaks", "knn_distance", "pseudo_sizes", "pseudo_boxes", "match_points_to_boxes",
           "PointTable"]


def find_peaks(levels, tau, radius):
    """Mức density uint8 [H,W] -> điểm [K,2] (x, y) pixel liên tục.

    Đỉnh = pixel có mức >= tau VÀ bằng mức lớn nhất trong cửa sổ (2·radius+1)²; các pixel đỉnh
    liền nhau (vùng đỉnh phẳng do lượng tử hoá jet) gộp thành MỘT điểm = trọng tâm của vùng.
    """
    lv = np.asarray(levels)
    if lv.max(initial=0) < max(tau, 1):
        return np.zeros((0, 2))
    mx = ndimage.maximum_filter(lv, size=2 * radius + 1, mode="constant", cval=0)
    peak = (lv == mx) & (lv >= max(tau, 1))
    lab, n = ndimage.label(peak, structure=np.ones((3, 3)))
    if n == 0:
        return np.zeros((0, 2))
    cy, cx = np.asarray(ndimage.center_of_mass(peak, lab, range(1, n + 1))).reshape(-1, 2).T
    return np.stack([cx + 0.5, cy + 0.5], axis=1)


def knn_distance(points, k=3):
    """[M,2] -> [M] khoảng cách trung bình tới min(k, M−1) điểm gần nhất (M = 1 -> nan)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    m = len(p)
    if m <= 1:
        return np.full(m, np.nan)
    d = np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    kk = min(k, m - 1)
    return np.sort(d, axis=1)[:, :kk].mean(1)


def pseudo_sizes(points, k, beta, s_min, s_max):
    """Cỡ giả ŝ = clamp(β · kNN, s_min, s_max); ảnh chỉ 1 điểm -> s_max. Cùng đơn vị với points."""
    d = knn_distance(points, k)
    s = np.where(np.isnan(d), s_max, beta * d)
    return np.clip(s, s_min, s_max)


def pseudo_boxes(points, sizes):
    """Điểm [M,2] + cỡ [M] -> box giả VUÔNG xyxy [M,4], tâm ĐÚNG bằng điểm (không kẹp vào ảnh:
    kẹp sẽ làm lệch tâm, mà tâm chính là nhãn)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(sizes, dtype=np.float64).reshape(-1)[:, None] / 2
    return np.concatenate([p - h, p + h], axis=1)


def match_points_to_boxes(points, boxes_xyxy):
    """Ghép MỘT-MỘT điểm <-> box chứa nó (Hungarian, chi phí = khoảng cách tới tâm box chia cỡ
    box; điểm ngoài box thì không ghép được). Dùng để đo chất lượng nhãn ở cửa G0.

    -> dict: pairs [(i_point, j_box)], inside [P,G] bool.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    if not len(p) or not len(b):
        return {"pairs": [], "inside": np.zeros((len(p), len(b)), dtype=bool)}
    x, y = p[:, :1], p[:, 1:]
    inside = (x >= b[None, :, 0]) & (x <= b[None, :, 2]) & (y >= b[None, :, 1]) & (y <= b[None, :, 3])
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w, h = np.maximum(b[:, 2] - b[:, 0], 1e-6), np.maximum(b[:, 3] - b[:, 1], 1e-6)
    cost = np.sqrt(((x - cx) / w) ** 2 + ((y - cy) / h) ** 2)
    big = 1e6
    cost = np.where(inside, cost, big)
    r, c = linear_sum_assignment(cost)
    ok = cost[r, c] < big
    return {"pairs": list(zip(r[ok].tolist(), c[ok].tolist())), "inside": inside}


class PointTable:
    """Đọc `data/density_points.json` (tools/build_density_points.py): iid -> điểm [M,2] pixel ẢNH GỐC."""

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"thiếu {path} — dựng một lần (cửa G0): python tools/build_density_points.py --out {path}")
        with open(path) as f:
            d = json.load(f)
        self.params = d.get("params", {})
        self.points = {k: np.asarray(v, dtype=np.float64).reshape(-1, 2) for k, v in d["points"].items()}

    def __contains__(self, iid):
        return iid in self.points

    def __getitem__(self, iid):
        return self.points[iid]
