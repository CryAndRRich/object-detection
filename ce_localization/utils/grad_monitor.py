"""Phân rã grad norm theo NHÓM tham số (lấy mẫu mỗi `every` bước, trước clip) — biết norm tổng đến từ đâu."""

from collections import defaultdict

import numpy as np
import torch

__all__ = ["GradMonitor", "group_of"]


def group_of(name):
    """Tên tham số của `BoxPolicy` -> nhóm."""
    if name.startswith("vision.backbone."):
        return "resnet18"
    if name.startswith("noise_net."):
        return "unet1d"
    if name.startswith("refiner.stages."):
        return "refiner.blocks"
    if name.startswith("refiner.heads."):
        return "refiner.head"
    if name.startswith("refiner."):
        return "refiner.embed"
    if name.startswith("text_proj."):
        return "text_proj"
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
