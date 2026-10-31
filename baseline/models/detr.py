"""Khung DETR của BootPlace (tham khảo `refs/BOOTPLACE/models/{backbone,position_encoding,transformer,matcher}.py`): ResNet BN đóng
băng + mã hoá vị trí sin + transformer post-norm (decoder trả mọi tầng) + bộ mã hoá patch vật + Hungarian matcher. Tên module / tham số
như bản gốc ⇒ nạp thẳng checkpoint công bố (khoá của phần đã bỏ — `patch_decoder`, không dùng trong forward — chỉ bị bỏ qua).
"""

import copy
import math

import torch
import torch.nn.functional as F
import torchvision
from scipy.optimize import linear_sum_assignment
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.boxes import box_area

__all__ = ["nested", "box_cxcywh_to_xyxy", "box_xyxy_to_cxcywh", "generalized_box_iou", "Joiner", "Transformer", "HungarianMatcher",
           "build_backbone", "build_transformer"]


def nested(tensors):
    """list [C,H,W] -> (tensor đệm 0 [B,C,H,W], mask [B,H,W] True = ô đệm)."""
    c = tensors[0].shape[0]
    h, w = max(t.shape[1] for t in tensors), max(t.shape[2] for t in tensors)
    out = tensors[0].new_zeros((len(tensors), c, h, w))
    mask = torch.ones((len(tensors), h, w), dtype=torch.bool, device=tensors[0].device)
    for i, t in enumerate(tensors):
        out[i, :, :t.shape[1], :t.shape[2]].copy_(t)
        mask[i, :t.shape[1], :t.shape[2]] = False
    return out, mask


def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def generalized_box_iou(a, b):
    area_a, area_b = box_area(a), box_area(b)
    inter = (torch.min(a[:, None, 2:], b[:, 2:]) - torch.max(a[:, None, :2], b[:, :2])).clamp(min=0).prod(-1)
    union = area_a[:, None] + area_b - inter
    hull = (torch.max(a[:, None, 2:], b[:, 2:]) - torch.min(a[:, None, :2], b[:, :2])).clamp(min=0).prod(-1)
    return inter / union - (hull - union) / hull


# ----------------------------------------------------------------------------- backbone + mã hoá vị trí

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n, eps=1e-5):
        super().__init__()
        self.eps = eps
        for k, v in (("weight", torch.ones(n)), ("bias", torch.zeros(n)), ("running_mean", torch.zeros(n)), ("running_var", torch.ones(n))):
            self.register_buffer(k, v)

    def _load_from_state_dict(self, state_dict, prefix, *args):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(state_dict, prefix, *args)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        return x * scale[None, :, None, None] + bias[None, :, None, None]


class Backbone(nn.Module):
    """ResNet (BN đóng băng), lấy layer4; `train_backbone` = train layer2–4 (lr_backbone > 0)."""

    def __init__(self, name, train_backbone, dilation):
        super().__init__()
        net = getattr(torchvision.models, name)(replace_stride_with_dilation=[False, False, dilation], weights=None,
                                                norm_layer=FrozenBatchNorm2d)
        for n, p in net.named_parameters():
            if not train_backbone or all(x not in n for x in ("layer2", "layer3", "layer4")):
                p.requires_grad_(False)
        self.body = IntermediateLayerGetter(net, return_layers={"layer4": "0"})
        self.num_channels = 512 if name in ("resnet18", "resnet34") else 2048

    def forward(self, x, mask):
        f = self.body(x)["0"]
        return f, F.interpolate(mask[None].float(), size=f.shape[-2:], mode="nearest")[0].bool()


class PositionEmbeddingSine(nn.Module):
    """Mã hoá vị trí sin chuẩn hoá theo vùng ảnh thật (bỏ ô đệm), như DETR."""

    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats, self.temperature = num_pos_feats, temperature

    def forward(self, mask):
        not_mask = ~mask
        y = not_mask.cumsum(1, dtype=torch.float32)
        x = not_mask.cumsum(2, dtype=torch.float32)
        y = y / (y[:, -1:, :] + 1e-6) * 2 * math.pi
        x = x / (x[:, :, -1:] + 1e-6) * 2 * math.pi
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        px, py = x[..., None] / dim_t, y[..., None] / dim_t
        px = torch.stack([px[..., 0::2].sin(), px[..., 1::2].cos()], dim=4).flatten(3)
        py = torch.stack([py[..., 0::2].sin(), py[..., 1::2].cos()], dim=4).flatten(3)
        return torch.cat([py, px], dim=3).permute(0, 3, 1, 2)


class Joiner(nn.Sequential):
    """[0] = Backbone, [1] = mã hoá vị trí (giữ chỉ số như bản gốc cho khoá checkpoint `backbone.0.body.*`)."""

    def forward(self, x, mask):
        f, m = self[0](x, mask)
        return f, m, self[1](m).to(f.dtype)


def build_backbone(cfg):
    j = Joiner(Backbone(cfg["backbone"], cfg.get("train_backbone", False), cfg["dilation"]), PositionEmbeddingSine(cfg["hidden_dim"] // 2))
    j.num_channels = j[0].num_channels
    return j


# ----------------------------------------------------------------------------- transformer

class PatchEncoder(nn.Module):
    """Patch RGBA 64×64 -> vector d_model (chỉ cho điểm khớp vật–vùng)."""

    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 16 * 16, out_dim)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        return self.fc(F.max_pool2d(F.relu(self.conv2(x)), 2).flatten(1))


class EncoderLayer(nn.Module):
    def __init__(self, d, nhead, ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.linear1, self.linear2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout1, self.dropout2, self.dropout_ffn = nn.Dropout(dropout), nn.Dropout(dropout), nn.Dropout(dropout)

    def forward(self, src, key_padding_mask, pos):
        q = src + pos
        src = self.norm1(src + self.dropout1(self.self_attn(q, q, src, key_padding_mask=key_padding_mask)[0]))
        return self.norm2(src + self.dropout2(self.linear2(self.dropout_ffn(F.relu(self.linear1(src))))))


class DecoderLayer(nn.Module):
    def __init__(self, d, nhead, ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d, nhead, dropout=dropout)
        self.linear1, self.linear2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.dropout1, self.dropout2, self.dropout3, self.dropout_ffn = (nn.Dropout(dropout) for _ in range(4))

    def forward(self, tgt, memory, key_padding_mask, pos, query_pos):
        q = tgt + query_pos
        tgt = self.norm1(tgt + self.dropout1(self.self_attn(q, q, tgt)[0]))
        tgt2 = self.cross_attn(tgt + query_pos, memory + pos, memory, key_padding_mask=key_padding_mask)[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))
        return self.norm3(tgt + self.dropout3(self.linear2(self.dropout_ffn(F.relu(self.linear1(tgt))))))


class _Stack(nn.Module):
    def __init__(self, layer, n, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n)])
        self.norm = norm


class Transformer(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        self.encoder = _Stack(EncoderLayer(d_model, nhead, dim_feedforward, dropout), num_encoder_layers)
        self.decoder = _Stack(DecoderLayer(d_model, nhead, dim_feedforward, dropout), num_decoder_layers, nn.LayerNorm(d_model))
        self.patch_encoder = PatchEncoder(d_model)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.d_model = d_model

    def forward(self, src, mask, patches, query_embed, pos):
        """src [B,C,H,W], mask [B,H,W], patches [B,4,P·64,64] -> (hs [L,B,Q,D] (đã LayerNorm từng tầng), đặc trưng patch [B,P,D])."""
        B, C, H, W = src.shape
        src, pos = src.flatten(2).permute(2, 0, 1), pos.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        qpos = query_embed.unsqueeze(1).repeat(1, B, 1)
        s = patches.shape[-1]
        pf = torch.stack([self.patch_encoder(patches[:, :, i * s:(i + 1) * s]) for i in range(patches.shape[2] // s)], 1)
        mem = src
        for layer in self.encoder.layers:
            mem = layer(mem, mask, pos)
        out, hs = torch.zeros_like(qpos), []
        for layer in self.decoder.layers:
            out = layer(out, mem, mask, pos, qpos)
            hs.append(self.decoder.norm(out))
        return torch.stack(hs).transpose(1, 2), pf


def build_transformer(cfg):
    return Transformer(cfg["hidden_dim"], cfg["nheads"], cfg["enc_layers"], cfg["dec_layers"], cfg["dim_feedforward"], cfg["dropout"])


class HungarianMatcher(nn.Module):
    """Ghép 1–1 vùng dự đoán ↔ đích theo cost = c_cls·(−p lớp) + c_L1·L1 + c_giou·(−GIoU) (box cxcywh chuẩn hoá)."""

    def __init__(self, cost_class=1, cost_bbox=1, cost_giou=1):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Q = outputs["pred_logits"].shape[:2]
        prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        ob = outputs["pred_boxes"].flatten(0, 1)
        ids = torch.cat([t["labels"] for t in targets])
        tb = torch.cat([t["boxes"] for t in targets])
        c = (-self.cost_class * prob[:, ids] + self.cost_bbox * torch.cdist(ob, tb, p=1)
             - self.cost_giou * generalized_box_iou(box_cxcywh_to_xyxy(ob), box_cxcywh_to_xyxy(tb))).view(B, Q, -1).cpu()
        idx = [linear_sum_assignment(x[i]) for i, x in enumerate(c.split([len(t["boxes"]) for t in targets], -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in idx]
