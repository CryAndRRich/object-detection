"""RoI: lấy đặc trưng CLIP BÊN TRONG mỗi box bằng lưới k x k điểm `grid_sample`.

Vì sao: cross-attention từ PE toạ độ sang patch token phải HỌC ánh xạ giữa hai hệ toạ độ
không liên quan; lấy mẫu thẳng tại toạ độ box thì quan hệ không gian đúng sẵn (như RoIAlign
của DiffusionDet). Vòng 1 đo: chỉ cross-attn AP50 1,52 -> thêm RoI 6,36.

Vì sao k=3 chứ không 1x1: lấy mỗi tâm thì mọi cỡ box cho CÙNG một vector (AUC đúng-cỡ vs
to-gấp-2 = 0,000); 3x3 cho 0,896. Vì sao không 7x7: box CE-130 trung vị ~2 patch @512px,
49 điểm là lấy trùng ở chi phí 5,4x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RoIFeatureSampler", "box_grid_points"]


def box_grid_points(boxes_norm, k=3):
    """[B,N,4] cxcywh in [0,1] -> [B,N,k*k,2] sample points in [0,1] image coords.

    Points sit at cell CENTRES of a k x k grid spanning the box, i.e. offsets
    (i+0.5)/k - 0.5 for i in 0..k-1. Cell centres, not corners, so no sample sits
    exactly on the box edge where it would straddle object and background.
    """
    dev, dt = boxes_norm.device, boxes_norm.dtype
    off = (torch.arange(k, device=dev, dtype=dt) + 0.5) / k - 0.5      # [k]
    gy, gx = torch.meshgrid(off, off, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)        # [k*k, 2]

    cx, cy, w, h = boxes_norm.unbind(-1)                               # [B,N] each
    x = cx.unsqueeze(-1) + pts[:, 0] * w.unsqueeze(-1)                 # [B,N,k*k]
    y = cy.unsqueeze(-1) + pts[:, 1] * h.unsqueeze(-1)
    return torch.stack([x, y], dim=-1)                                 # [B,N,k*k,2]


class RoIFeatureSampler(nn.Module):
    """Patch token CLIP thô + box -> một vector d_model mỗi box.

    "Chiếu trước, trộn sau": Linear(d_in -> d_model) dùng chung cho k*k điểm, rồi
    Linear(k*k*d_model -> d_model) trộn (0,79M tham số ở k=3). Giữ tâm và mép tách
    riêng — tín hiệu cho box biết nó to/nhỏ sai; pooling trước sẽ xoá mất.

    Lớp `out` ZERO-INIT: bước 0 module đóng góp 0 (luồng `r` mở dần qua gate); vẫn học
    ngay vì dL/dW = delta * x^T khác 0.
    """

    def __init__(self, d_in=768, d_model=256, k=3, dropout=0.0):
        super().__init__()
        self.k = k
        self.proj_point = nn.Linear(d_in, d_model)
        self.out = nn.Linear(k * k * d_model, d_model)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, patch_tokens, boxes_norm, grid=None):
        """
        patch_tokens : [B, P, d_in] raw CLIP patch tokens (P must be a square)
        boxes_norm   : [B, N, 4] cxcywh in [0,1]
        grid         : optional int, the patch grid side; inferred from P if None
        -> [B, N, d_model]
        """
        B, P, d_in = patch_tokens.shape
        g = grid or int(round(P ** 0.5))
        assert g * g == P, f"{P} patch tokens is not a square grid"

        fmap = patch_tokens.transpose(1, 2).reshape(B, d_in, g, g)

        pts = box_grid_points(boxes_norm, self.k)                      # [B,N,k*k,2]
        # grid_sample wants [-1,1] with align_corners=False matching the
        # pixel-centre convention the patch grid already uses.
        samp = F.grid_sample(fmap, pts * 2.0 - 1.0, mode="bilinear",
                             padding_mode="border", align_corners=False)
        # [B, d_in, N, k*k] -> [B, N, k*k, d_in]
        samp = samp.permute(0, 2, 3, 1)

        h = self.proj_point(samp.to(self.proj_point.weight.dtype))     # [B,N,k*k,d_model]
        h = self.drop(h.flatten(-2))                                   # [B,N,k*k*d_model]
        return self.out(h)                                             # [B,N,d_model]

