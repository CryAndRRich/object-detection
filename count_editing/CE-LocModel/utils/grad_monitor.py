"""Phân rã grad norm theo NHÓM MODULE — biết con số tổng đến từ đâu.

VÌ SAO TỒN TẠI: log train chỉ in MỘT con số (~100, trước clip) và cảnh báo "rất lớn". Một
con số tổng không phân biệt được ba giả thuyết rất khác nhau:

  (a) CHUỖI CỘNG DỒN: `x` không detach giữa các tầng (có chủ đích — box là luồng chính),
      nên `box_delta` tầng 1 nhận gradient từ loss của CẢ 6 tầng. Nếu đúng thì
      `box_delta[0]` phải lớn hơn hẳn `box_delta[5]`.
  (b) ĐẦU VÀO CLIP THÔ: `patch_raw` là `last_hidden_state` TRƯỚC `post_layernorm`, luồng
      dư của ViT có vài kênh outlier rất lớn. Gradient của `Linear` tỉ lệ với đầu vào,
      nên nếu đúng thì `proj_patch` / `roi.proj_point` áp đảo.
  (c) HEAD SCORE / focal.

Lấy mẫu mỗi `every` batch rồi `.item()` — không đồng bộ GPU mỗi bước.
"""

import re
from collections import defaultdict

import numpy as np
import torch

__all__ = ["GradMonitor", "group_of"]

_LAYER = re.compile(r"decoder\.layers\.(\d+)\.(\w+)")


def group_of(name):
    """Tên tham số -> tên nhóm. `box_delta` tách theo tầng; phần khác gộp qua các tầng."""
    m = _LAYER.match(name)
    if m:
        i, sub = int(m.group(1)), m.group(2)
        if sub == "box_delta":
            return f"box_delta[{i}]"
        if sub == "roi":
            return "roi.out" if ".roi.out." in name else "roi.proj_point"
        return sub                                  # self_attn, cross_attn, ff_h, ...
    if name.startswith("encoder."):
        return name.split(".")[1]                   # proj_patch, proj_text
    if name.startswith("decoder.score_head"):
        return "score_head"
    return "embed/khác"                             # box_embed, time_cond, pos_emb...


class GradMonitor:
    def __init__(self, model, every=10):
        self.every = every
        self.groups = defaultdict(list)
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.groups[group_of(n)].append(p)
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
