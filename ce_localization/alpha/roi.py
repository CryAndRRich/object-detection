"""RoIAlign nhiều tầng FPN — tương đương `ROIPooler(ROIAlignV2, 7, sampling_ratio=2)` mà
DiffusionDet dùng (`refs/repos/DiffusionDet/diffusiondet/head.py:124-144`).

- `ROIAlignV2` của detectron2 = `torchvision.ops.roi_align(..., aligned=True)`.
- Chọn tầng theo công thức detectron2 (`assign_boxes_to_levels`):
      lvl = floor(canonical_level + log2(sqrt(area) / canonical_box_size + 1e-8)),
  canonical 224 / tầng 4, kẹp [2, 5].
- KHÔNG dùng `torchvision.ops.MultiScaleRoIAlign`: nó gọi roi_align với aligned=False và eps
  chọn tầng khác (docs/EXPERIMENT_ALPHA.md mục 2.2).
"""

import math

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from ce_localization.alpha.backbone import STRIDES

__all__ = ["assign_levels", "MultiLevelRoIAlign"]


def assign_levels(boxes_xyxy, min_level=2, max_level=5, canonical_box_size=224, canonical_level=4):
    """[K,4] xyxy tuyệt đối -> chỉ số tầng tính từ `min_level` (0 = P2)."""
    w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0)
    h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0)
    size = torch.sqrt(w * h)
    lvl = torch.floor(canonical_level + torch.log2(size / canonical_box_size + 1e-8))
    return (lvl.clamp(min=min_level, max=max_level) - min_level).long()


class MultiLevelRoIAlign(nn.Module):
    def __init__(self, output_size=7, sampling_ratio=2, strides=STRIDES):
        super().__init__()
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio
        self.scales = [1.0 / s for s in strides]
        self.min_level = int(round(math.log2(strides[0])))
        self.max_level = int(round(math.log2(strides[-1])))

    def forward(self, feats, boxes):
        """feats: list [B,C,Hl,Wl] (P2..P5) ; boxes: [B,N,4] xyxy tuyệt đối.
        -> [B*N, C, S, S], thứ tự (ảnh 0: box 0..N-1, ảnh 1: ...)."""
        B, N = boxes.shape[:2]
        flat = boxes.reshape(-1, 4)
        bidx = torch.arange(B, device=boxes.device, dtype=flat.dtype).repeat_interleave(N)
        rois = torch.cat([bidx[:, None], flat], dim=1)
        lvl = assign_levels(flat, self.min_level, self.max_level)
        C = feats[0].shape[1]
        out = flat.new_zeros((B * N, C, self.output_size, self.output_size), dtype=feats[0].dtype)
        for li, (f, sc) in enumerate(zip(feats, self.scales)):
            idx = torch.nonzero(lvl == li, as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            out[idx] = roi_align(f, rois[idx].to(f.dtype), self.output_size, spatial_scale=sc,
                                 sampling_ratio=self.sampling_ratio, aligned=True)
        return out
