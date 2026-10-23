"""Density map (kênh 4 của ảnh vào): `samples/{train,test}/density/{iid}_{k}.png` — density CountGD tô màu jet trên TỪNG ảnh inpaint
(mỗi ảnh gốc 1–12 bản, mỗi bản thiếu các vật đã xoá tới lượt đó). Ảnh inpaint dùng density của chính mẫu (`data/turns.py`); ảnh GỐC dùng
bản `full` = bản có tổng diện tích blob lớn nhất của ảnh gốc đó — chỉ mục `data/density_index.json` (`tools/build_density_index.py`,
giải mã jet về mức 0..255 bằng màu gần nhất trong bảng jet 256 mức để đếm diện tích).
"""

import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

__all__ = ["JET", "decode_jet", "build_index", "DensityIndex"]

# `_jet_data` của matplotlib (matplotlib/_cm.py): (x, giá trị) từng kênh, nội suy tuyến tính.
_JET_SEG = (((0.0, 0.0), (0.35, 0.0), (0.66, 1.0), (0.89, 1.0), (1.0, 0.5)),
            ((0.0, 0.0), (0.125, 0.0), (0.375, 1.0), (0.64, 1.0), (0.91, 0.0), (1.0, 0.0)),
            ((0.0, 0.5), (0.11, 1.0), (0.34, 1.0), (0.65, 0.0), (1.0, 0.0)))


def _jet_lut(n=256):
    """Bảng jet N mức, uint8 như `cmap(x, bytes=True)` (nhân 255 rồi CẮT, không làm tròn)."""
    x = np.linspace(0.0, 1.0, n)
    c = np.stack([np.interp(x, [p[0] for p in s], [p[1] for p in s]) for s in _JET_SEG], 1)
    return (c * 255).astype(np.uint8)


JET = _jet_lut()


def decode_jet(rgb):
    """uint8 [H,W,3] tô jet -> (mức uint8 [H,W], khoảng cách màu lớn nhất tới bảng jet)."""
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    code = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    u, inv = np.unique(code.ravel(), return_inverse=True)       # ảnh thật chỉ ~200 màu
    uc = np.stack([u >> 16, (u >> 8) & 255, u & 255], 1)
    d = ((uc[:, None, :] - JET[None].astype(np.int32)) ** 2).sum(-1)
    lv = d.argmin(1).astype(np.uint8)
    return lv[inv.reshape(-1)].reshape(h, w), float(np.sqrt(d.min(1).max()))


# ----------------------------------------------------------------------------- chỉ mục

def _scan_one(job):
    root, rel = job
    lv, dist = decode_jet(np.asarray(Image.open(os.path.join(root, rel)).convert("RGB")))
    return rel, int((lv > 0).sum()), [int(lv.shape[1]), int(lv.shape[0])], dist


def iid_of(density_file):
    """`1077_12.png` -> `1077`."""
    return os.path.splitext(os.path.basename(density_file))[0].rsplit("_", 1)[0]


def build_index(samples_root, workers=8, log=print):
    """Quét mọi `<split>/density/*.png` dưới `samples_root` -> dict chỉ mục (đường dẫn TƯƠNG ĐỐI
    so với `samples_root`, nên chép sang máy khác chỉ cần đổi root)."""
    rels = []
    for s in sorted(os.listdir(samples_root)):
        d = os.path.join(samples_root, s, "density")
        if os.path.isdir(d):
            rels += [f"{s}/density/{f}" for f in sorted(os.listdir(d)) if f.endswith(".png")]
    if not rels:
        raise FileNotFoundError(f"không thấy <split>/density/*.png trong {samples_root}")
    jobs = [(samples_root, r) for r in rels]
    out, done = [], 0
    pool = Pool(workers) if workers > 0 else None
    for r in (pool.imap_unordered(_scan_one, jobs, chunksize=64) if pool else map(_scan_one, jobs)):
        out.append(r)
        done += 1
        if done % 2000 == 0 or done == len(jobs):
            log(f"  density {done}/{len(jobs)}")
    if pool:
        pool.close()
        pool.join()
    by_iid = defaultdict(list)
    for rel, area, size, _ in out:
        by_iid[iid_of(rel)].append([rel, area, size])
    variants = {k: sorted(v, key=lambda x: (-x[1], x[0])) for k, v in sorted(by_iid.items())}
    return {"n_files": len(out), "max_color_dist": max(r[3] for r in out), "variants": variants}


class DensityIndex:
    """Đọc chỉ mục JSON (`tools/build_density_index.py`); `full(iid)` -> đường dẫn bản density đầy đủ nhất của ảnh gốc."""

    def __init__(self, index_path, samples_root):
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"thiếu chỉ mục density {index_path} — dựng một lần: python tools/build_density_index.py "
                f"--samples {samples_root} --out {index_path}")
        with open(index_path) as f:
            self.variants = json.load(f)["variants"]
        self.root = samples_root

    def __contains__(self, iid):
        return iid in self.variants

    def full(self, iid):
        return os.path.join(self.root, self.variants[iid][0][0])
