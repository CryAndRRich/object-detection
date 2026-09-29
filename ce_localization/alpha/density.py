"""Density map cho ALPHA3 (docs/EXPERIMENT_ALPHA.md mục 5): kênh thứ 4 của R-50, như CE-Loc gốc.

Nguồn: `samples/{train,test}/density/{iid}_{k}.png` — density CountGD sinh trên TỪNG ảnh inpaint
(mỗi ảnh gốc có 1–12 bản, trung vị 7–9, mỗi bản thiếu các vật đã bị xoá tới lượt đó). Split của
`samples/` khác split detect nên tra theo `iid`, không theo split.

Giải mã: PNG tô màu **jet** của matplotlib (nền (0,0,127) = mức 0, đỏ đậm (127,0,0) = mức 255).
Đổi màu ngược về mức 0..255 bằng màu GẦN NHẤT trong bảng jet 256 mức (trên dữ liệu thật mọi màu
trùng đúng bảng, khoảng cách 0; bảng chỉ có 253 màu phân biệt: mức 29–32 cùng là (0,0,255) và giải
mã về 29 -> sai tối đa 3/255). CE-Loc gốc đọc bằng `.convert("L")` (nền 14/255, blob đậm nhất thành vòng rỗng) —
người dùng chốt giải mã jet (2026-09-29). Resize NEAREST + dán góc trên-trái như gốc, rồi /255.

Chế độ chọn bản density cho một ảnh (`DensityIndex.pick`):
  full    : bản có tổng diện tích blob (số pixel mức > 0) LỚN NHẤT — "bản count đầy đủ nhất"
  partial : bản diện tích NHỎ NHẤT trong các bản còn lại (ảnh chỉ có 1 bản -> dùng bản full)
  empty   : map trống (toàn 0)
  mix     : 1/3 full · 1/3 một bản ngẫu nhiên khác full (1 bản -> full) · 1/3 empty
ALPHA3.1 train `full`, ALPHA3.2 train `mix`; eval cả 3 chế độ full / partial / empty.
"""

import json
import os
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

__all__ = ["MODES", "EVAL_MODES", "JET", "decode_jet", "load_density_levels", "letterbox_density",
           "build_index", "DensityIndex"]

MODES = ("full", "partial", "empty", "mix")
EVAL_MODES = ("full", "partial", "empty")

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


def load_density_levels(path):
    """PNG density -> mức uint8 [H,W] (kích thước ảnh gốc)."""
    return decode_jet(np.asarray(Image.open(path).convert("RGB")))[0]


def letterbox_density(levels_u8, nw, nh, target):
    """Mức uint8 [H,W] -> float32 [T,T] trong [0,1]: resize (nw,nh) NEAREST, dán góc trên-trái
    lên canvas 0 — đúng `resize_and_pad` của CE-Loc gốc (phần density)."""
    canvas = np.zeros((target, target), dtype=np.float32)
    small = Image.fromarray(np.asarray(levels_u8, dtype=np.uint8)).resize((nw, nh), resample=Image.NEAREST)
    canvas[:nh, :nw] = np.asarray(small, dtype=np.float32) / 255.0
    return canvas


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
    """Đọc chỉ mục JSON (`tools/build_density_index.py`); `pick` chọn bản density theo chế độ."""

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

    def path(self, rel):
        return os.path.join(self.root, rel)

    def pick(self, iid, mode, rng=None):
        """-> (đường dẫn tương đối hoặc None = map trống, loại thực tế 'full'|'partial'|'empty')."""
        v = self.variants[iid]
        if mode == "full":
            return v[0][0], "full"
        if mode == "partial":
            return (v[-1][0], "partial") if len(v) > 1 else (v[0][0], "full")
        if mode == "empty":
            return None, "empty"
        if mode == "mix":
            r = int(rng.integers(3))
            if r == 0:
                return v[0][0], "full"
            if r == 1:
                return (v[1 + int(rng.integers(len(v) - 1))][0], "partial") if len(v) > 1 else (v[0][0], "full")
            return None, "empty"
        raise ValueError(f"density mode {mode!r} không thuộc {MODES}")
