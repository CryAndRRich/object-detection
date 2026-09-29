"""AlphaDetector = R-50+FPN -> memory kiểu DP -> head 6 stage (docs/EXPERIMENT_ALPHA.md mục 2).

`forward(images, text_raw, valid_hw, boxes, t)` là MỘT lượt khử nhiễu: ảnh + text + box nhiễu
(xyxy tuyệt đối) + t -> đầu ra mọi stage. Train gọi thẳng qua DDP; suy luận gọi qua
`sample()` (DDIM, phần ảnh chỉ tính một lần mỗi ảnh). ALPHA3: `model.in_channels: 4` -> ảnh
[B,4,H,W] = RGB chuẩn hoá + density (kênh 4 của conv1 khởi tạo 0).
"""

import torch
import torch.nn as nn

from ce_localization.alpha.backbone import ResNet50FPN
from ce_localization.alpha.diffusion import cosine_alphas_cumprod, ddim_sample
from ce_localization.alpha.head import AlphaHead
from ce_localization.alpha.memory import MemoryEncoder

__all__ = ["AlphaDetector", "build_model"]


class AlphaDetector(nn.Module):
    def __init__(self, memory="none", d_model=256, n_stage=6, n_head=4, dim_feedforward=1024,
                 dropout=0.3, text_dim=512, prior_prob=0.01, pretrained_backbone=True,
                 num_timesteps=1000, snr_scale=2.0, grid_size=None, in_channels=3):
        super().__init__()
        self.backbone = ResNet50FPN(d_model, pretrained=pretrained_backbone, in_channels=in_channels)
        self.memory = MemoryEncoder(memory, d_model, text_dim, feat_channels=d_model, feat_stride=32,
                                    grid_size=grid_size)
        self.head = AlphaHead(n_stage, d_model, n_head, dim_feedforward, dropout, prior_prob)
        self.snr_scale = snr_scale
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(num_timesteps), persistent=False)

    def encode_image(self, images, valid_hw):
        """-> (feats [P2..P5], img_tok, img_mask). Không phụ thuộc t."""
        f = self.backbone(images)
        feats = [f["p2"], f["p3"], f["p4"], f["p5"]]
        img_tok, img_mask = self.memory.image_tokens(f["p5"], valid_hw)
        return feats, img_tok, img_mask

    def denoise(self, feats, img_tok, img_mask, text_raw, boxes, t):
        mem, mask = self.memory(t, text_raw, img_tok, img_mask)
        return self.head(feats, boxes, mem, mask)

    def forward(self, images, text_raw, valid_hw, boxes, t):
        """-> (logits [S,B,N,1], boxes [S,B,N,4] xyxy tuyệt đối)."""
        feats, img_tok, img_mask = self.encode_image(images, valid_hw)
        return self.denoise(feats, img_tok, img_mask, text_raw, boxes, t)

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, whwh, num_proposals, steps=1, renewal=True,
               generator=None):
        feats, img_tok, img_mask = self.encode_image(images, valid_hw)
        fn = lambda boxes, t: self.denoise(feats, img_tok, img_mask, text_raw, boxes, t)  # noqa: E731
        return ddim_sample(fn, images.shape[0], num_proposals, whwh, self.alphas_cumprod,
                           self.snr_scale, steps, eta=1.0, renewal=renewal, generator=generator)


def build_model(cfg, pretrained_backbone=None):
    m, d = cfg["model"], cfg["diffusion"]
    pre = m.get("pretrained_backbone", True) if pretrained_backbone is None else pretrained_backbone
    return AlphaDetector(
        memory=m["memory"], d_model=m["d_model"], n_stage=m["n_stage"], n_head=m["n_head"],
        dim_feedforward=m["dim_feedforward"], dropout=m["dropout"], text_dim=m.get("text_dim", 512),
        prior_prob=cfg["loss"]["prior_prob"], pretrained_backbone=pre,
        num_timesteps=d["num_timesteps"], snr_scale=d["snr_scale"], grid_size=m.get("grid_size"),
        in_channels=m.get("in_channels", 3))
