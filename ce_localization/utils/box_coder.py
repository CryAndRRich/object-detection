"""Mã hoá box trong không gian khuếch tán của `BoxPolicy` (CE-Loc bài add) — `BoxCoder` của bản CE-Loc log-scale tác giả gửi
(`refs/CE-Loc-logScale/utils/normalization.py`, READ-ONLY, không import — viết lại ĐÚNG công thức; có test so với bản sao nguyên văn).
DELTA5 (docs/EXPERIMENT_DELTA.md mục 11), khoá config `model.box_coder`:

  linear (mặc định — MỌI config cũ không có khoá): (cx, cy, w, h) / canvas · 2 − 1
  log: cx, cy như linear; w, h -> log(w / canvas) (kẹp dưới ở `min_size_ratio`, mặc định 1/128 = 4 px trên 512), ánh xạ tuyến tính
       [log(min_size_ratio), 0] -> [−1, 1]. Giải mã kẹp giá trị chuẩn hoá về [−5, 5] trước exp (box nhiễu) ⇒ cạnh luôn > 0.
       Box hợp lệ (trong canvas, cạnh trong [min_size_ratio, 1] canvas) nằm trong [−1, 1]⁴ ở cả hai chế độ.

Mọi phép trong code đi qua `to_linear_unit` (box mã hoá -> box `linear` cùng canvas) rồi dùng lại công thức linear sẵn có (RoIAlign của
refiner, `unit_to_boxes`) — `linear` trả lại CHÍNH tensor đầu vào ⇒ config cũ trùng từng bit.
"""

import math

import torch

__all__ = ["BOX_CODERS", "LOG_DECODE_CLAMP", "box_coder_config", "is_log", "encode_rel", "to_linear_unit", "unit_fraction",
           "generalized_box_iou_cxcywh"]

BOX_CODERS = ("linear", "log")
LOG_DECODE_CLAMP = 5.0                       # tác giả: `box[..., 2:].clamp(-5, 5)` trước exp


def box_coder_config(cfg=None):
    """Khoá `model.box_coder` -> dict đủ khoá ({type, min_size_ratio}); None / thiếu khoá = linear như tác giả."""
    c = {"type": "linear", "min_size_ratio": 1 / 128, **(cfg or {})}
    if c["type"] not in BOX_CODERS:
        raise ValueError(f"box_coder.type {c['type']!r} không thuộc {BOX_CODERS}")
    if not 0 < float(c["min_size_ratio"]) < 1:
        raise ValueError(f"box_coder.min_size_ratio {c['min_size_ratio']!r} phải trong (0, 1)")
    return c


def is_log(coder):
    return coder is not None and coder["type"] == "log"


def encode_rel(rel, coder=None):
    """rel [..., 4] (cx, cy, w, h) / canvas -> mã hoá. linear: rel·2 − 1 (đúng phép cũ)."""
    out = rel * 2 - 1
    if not is_log(coder):
        return out
    mr = float(coder["min_size_ratio"])
    lm = math.log(mr)
    size = rel[..., 2:].clamp(min=mr).log()
    return torch.cat([out[..., :2], (size - lm) / -lm * 2 - 1], dim=-1)


def to_linear_unit(u, coder=None):
    """Box mã hoá [..., 4] -> box mã hoá `linear` cùng canvas (w, h = size·2 − 1). linear: trả lại CHÍNH `u`."""
    if not is_log(coder):
        return u
    lm = math.log(float(coder["min_size_ratio"]))
    size = ((u[..., 2:].clamp(-LOG_DECODE_CLAMP, LOG_DECODE_CLAMP) + 1) / 2 * -lm + lm).exp()
    return torch.cat([u[..., :2], size * 2 - 1], dim=-1)


def unit_fraction(u, coder=None):
    """Box mã hoá -> (cx, cy, w, h) theo tỉ lệ canvas — `BoxCoder.decode(box, canvas_size=1)` của tác giả."""
    return (to_linear_unit(u, coder) + 1) / 2


def generalized_box_iou_cxcywh(a, b):
    """GIoU từng cặp của box (cx, cy, w, h) cùng hệ, [...]; cạnh âm coi như 0 — `generalized_box_iou` của tác giả."""
    def xyxy(x):
        w, h = x[..., 2].clamp(min=0), x[..., 3].clamp(min=0)
        return torch.stack([x[..., 0] - w / 2, x[..., 1] - h / 2, x[..., 0] + w / 2, x[..., 1] + h / 2], dim=-1)
    a, b = xyxy(a), xyxy(b)
    iw = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    ih = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    inter = iw * ih
    union = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    iou = inter / (union + 1e-6)
    ew = torch.maximum(a[..., 2], b[..., 2]) - torch.minimum(a[..., 0], b[..., 0])
    eh = torch.maximum(a[..., 3], b[..., 3]) - torch.minimum(a[..., 1], b[..., 1])
    enclose = ew * eh
    return iou - (enclose - union) / (enclose + 1e-6)
