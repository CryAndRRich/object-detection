"""Phân rã grad norm theo NHÓM MODULE — biết con số tổng (trước clip) đến từ đâu: backbone, FPN,
chiếu RoI, decoder hay head của từng stage. Lấy mẫu mỗi `every` batch rồi `.item()` — không đồng bộ
GPU mỗi bước.
"""

import re
from collections import defaultdict

import numpy as np
import torch

__all__ = ["GradMonitor", "group_of"]

_STAGE = re.compile(r"head\.stages\.(\d+)\.(\w+)")


def group_of(name):
    """Tên tham số của `models.detector.Detector` -> nhóm."""
    if name.startswith("backbone.fpn."):
        return "fpn"
    if name.startswith("backbone."):
        return "resnet"
    if name.startswith("memory."):
        return "memory." + name.split(".")[1]
    m = _STAGE.match(name)
    if m:
        i, sub = int(m.group(1)), m.group(2)
        if sub in ("roi_proj", "decoder"):
            return f"{sub}[{i}]"
        return f"heads[{i}]"
    return "khác"


class GradMonitor:
    def __init__(self, model, every=10, group_fn=group_of):
        """`group_fn`: tên tham số -> tên nhóm."""
        self.every = every
        self.groups = defaultdict(list)
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.groups[group_fn(n)].append(p)
        self.samples = defaultdict(list)

    def maybe_record(self, step):
        """Gọi SAU `backward`, TRƯỚC `clip_grad_norm_` (để đo độ lớn thật, chưa bị cắt)."""
        if step % self.every:
            return
        with torch.no_grad():
            for g, params in self.groups.items():
                sq = [p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None]
                if sq:
                    self.samples[g].append(float(torch.stack(sq).sum().sqrt()))

    def summary(self):
        """-> {nhóm: trung bình norm} cho epoch vừa xong, rồi xoá để đo epoch sau."""
        out = {g: float(np.mean(v)) for g, v in self.samples.items() if v}
        self.samples.clear()
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    @staticmethod
    def share(summary):
        """Tỉ phần của mỗi nhóm trong norm tổng (theo bình phương — cộng được)."""
        tot = sum(v * v for v in summary.values()) or 1.0
        return {g: v * v / tot for g, v in summary.items()}
