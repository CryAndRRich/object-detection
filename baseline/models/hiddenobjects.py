"""PlacementDETR của HiddenObjects (WACV 2027) — viết lại theo `refs/repos/HiddenObjects/distilled_model/{model,train}.py`.

ResNet-50 ImageNet ĐÓNG BĂNG (C4 + C5, gộp kiểu FPN -> d), PE sin 2D cộng một lần vào đầu vào encoder, `nn.Transformer{Encoder,Decoder}`
post-norm, Q query = (nhúng lớp) + `query_offsets`; mỗi query ra box xywh chuẩn hoá (góc trên trái, sigmoid) + logit plausibility.
Tên module / khoá state dict giữ như bản gốc ⇒ nạp thẳng checkpoint công bố (checkpoint không chứa backbone; `input_proj` của bản gốc
chỉ dùng cho feature cache dạng tensor ⇒ bỏ, khoá thừa).

Khác bản gốc (ZETA3, docs/EXPERIMENT_ZETA.md mục 12):
- `class_embed` (bảng 28 lớp đóng) -> `text_proj` = Linear trên CLIP text chuẩn hoá L2 (lớp CE-130 test không có trong train); khởi tạo
  bằng nghiệm min-norm `lstsq_text_proj` ⇒ trên các lớp của checkpoint ra đúng nhúng cũ.
- Train end-to-end trên ảnh (bản gốc train trên feature cache) ⇒ `train()` luôn giữ backbone ở `eval()` (không trôi thống kê BN).
- `hungarian_loss`: đích plausibility (max GIoU) được `detach` — bản gốc để grad của MSE chảy ngược vào box qua đích.
"""

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


class MultiScaleBackbone(nn.Module):
    """ResNet-50 đóng băng -> {c4 (stride 16, 1024 kênh), c5 (stride 32, 2048 kênh)}."""

    def __init__(self, weights="DEFAULT"):
        super().__init__()
        r = resnet50(weights=ResNet50_Weights.DEFAULT if weights == "DEFAULT" else None)
        self.stage1 = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.stage2, self.stage3, self.stage4, self.stage5 = r.layer1, r.layer2, r.layer3, r.layer4
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        c4 = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return c4, self.stage5(c4)


def sine_pos(B, H, W, d, device, temperature=10000.0):
    """PE sin 2D chuẩn hoá (DETR, như bản gốc) -> [B, H·W, d]."""
    y = torch.arange(H, device=device).float()[:, None].expand(H, W) / (H + 1e-6) * 2 * math.pi
    x = torch.arange(W, device=device).float()[None, :].expand(H, W) / (W + 1e-6) * 2 * math.pi
    dim_t = torch.arange(d // 2, device=device).float()
    dim_t = temperature ** (2 * (dim_t // 2) / (d // 2))
    px, py = x[..., None] / dim_t, y[..., None] / dim_t
    px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], -1).flatten(-2)
    py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], -1).flatten(-2)
    return torch.cat([py, px], -1).flatten(0, 1)[None].expand(B, -1, -1)


class PlacementDETR(nn.Module):
    def __init__(self, num_queries=50, hidden_dim=256, nheads=8, enc_layers=6, dec_layers=6, dim_feedforward=2048, dropout=0.1,
                 text_dim=512, backbone_weights="DEFAULT", **_):
        super().__init__()
        self.num_queries, self.hidden_dim = num_queries, hidden_dim
        self.backbone = MultiScaleBackbone(backbone_weights)
        self.proj_c4 = nn.Conv2d(1024, hidden_dim, 1)
        self.proj_c5 = nn.Conv2d(2048, hidden_dim, 1)
        self.fusion_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.query_offsets = nn.Parameter(torch.randn(num_queries, hidden_dim))
        enc = nn.TransformerEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, batch_first=True, norm_first=False)
        dec = nn.TransformerDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, batch_first=True, norm_first=False)
        self.transformer_encoder = nn.TransformerEncoder(enc, enc_layers, enable_nested_tensor=False)
        self.transformer_decoder = nn.TransformerDecoder(dec, dec_layers)
        self.bbox_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, 4), nn.Sigmoid())
        self.plausibility_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x, text):
        """x [B,3,S,S] đã chuẩn hoá ImageNet, text [B, text_dim] -> (box [B,Q,4] xywh chuẩn hoá, logit [B,Q])."""
        with torch.no_grad():
            c4, c5 = self.backbone(x)
        p4 = self.proj_c4(c4)
        f = self.fusion_conv(p4 + nn.functional.interpolate(self.proj_c5(c5), size=p4.shape[-2:], mode="bilinear", align_corners=False))
        B, d, H, W = f.shape
        memory = self.transformer_encoder(f.flatten(2).transpose(1, 2) + sine_pos(B, H, W, d, f.device).to(f.dtype))
        hs = self.transformer_decoder(self.text_proj(text)[:, None] + self.query_offsets[None], memory)
        return self.bbox_head(hs), self.plausibility_head(hs)[..., 0]


def xywh_to_xyxy(b):
    return torch.cat([b[..., :2], b[..., :2] + b[..., 2:]], -1)


def pairwise_giou(a, b):
    """xyxy [..., N, 4], [..., M, 4] -> GIoU [..., N, M]."""
    a, b = a[..., :, None, :], b[..., None, :, :]
    area = lambda z: (z[..., 2] - z[..., 0]).clamp(min=0) * (z[..., 3] - z[..., 1]).clamp(min=0)  # noqa: E731
    lt, rb = torch.maximum(a[..., :2], b[..., :2]), torch.minimum(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-7)
    hull = (torch.maximum(a[..., 2:], b[..., 2:]) - torch.minimum(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-7)


def hungarian_loss(boxes, logits, targets, w_l1=5.0, w_giou=2.0, w_plaus=0.5):
    """box [B,Q,4] xywh, logit [B,Q] (fp32), targets list B × [m_i >= 1, 4] xywh -> (loss / mẫu [B], {thành phần TB batch}).

    Như `hungarian_loss` bản gốc với reward = 1: ghép Hungarian theo cost L1 + (1 − GIoU); box = TB trên cặp khớp của 5·L1 + 2·(1 − GIoU);
    plausibility = MSE(sigmoid, clamp(max GIoU theo đích, 0, 1)) TB trên Q; loss = box + 0,5·plaus. Cost cả batch chép về CPU MỘT lần.
    """
    B, dev = boxes.shape[0], boxes.device
    ms = [len(t) for t in targets]
    if min(ms) < 1:
        raise ValueError("mẫu không có đích")
    tg = torch.zeros(B, max(ms), 4, device=dev, dtype=boxes.dtype)
    for i, t in enumerate(targets):
        tg[i, :len(t)] = t.to(dev, boxes.dtype)
    valid = torch.arange(tg.shape[1], device=dev)[None] < torch.tensor(ms, device=dev)[:, None]          # [B, M]
    l1 = (boxes[:, :, None] - tg[:, None]).abs().sum(-1)                                                   # [B, Q, M]
    giou = pairwise_giou(xywh_to_xyxy(boxes), xywh_to_xyxy(tg))
    cost = (l1 + 1 - giou).detach().cpu().numpy()
    bi, pi, gi = [], [], []
    for i, m in enumerate(ms):
        p, g = linear_sum_assignment(cost[i, :, :m])
        bi.append(np.full(len(p), i))
        pi.append(p)
        gi.append(g)
    bi, pi, gi = (torch.as_tensor(np.concatenate(v), device=dev, dtype=torch.long) for v in (bi, pi, gi))
    n = torch.zeros(B, device=dev).index_add_(0, bi, torch.ones(len(bi), device=dev))
    mean = lambda v: torch.zeros(B, device=dev, dtype=v.dtype).index_add_(0, bi, v) / n  # noqa: E731
    l1_m, giou_m = mean(l1[bi, pi, gi]), mean(1 - giou[bi, pi, gi])
    best = giou.detach().masked_fill(~valid[:, None], -2.0).amax(-1).clamp(0, 1)                         # [B, Q]
    plaus = ((torch.sigmoid(logits) - best) ** 2).mean(-1)
    loss = w_l1 * l1_m + w_giou * giou_m + w_plaus * plaus
    return loss, {"l1": l1_m.mean().detach(), "giou": giou_m.mean().detach(), "plaus": plaus.mean().detach()}


def lstsq_text_proj(text, emb):
    """Nghiệm min-norm của [text, 1] · [Wᵀ; b] = emb (float64, pinv) -> (W [H, D], b [H]). C lớp < D + 1 ⇒ khớp đúng từng lớp."""
    A = torch.cat([text.double(), torch.ones(len(text), 1, dtype=torch.float64)], 1)
    X = torch.linalg.pinv(A) @ emb.double()
    return X[:-1].T.float().contiguous(), X[-1].float().contiguous()
