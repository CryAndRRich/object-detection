"""Detector = R-50+FPN -> memory kiểu DP -> head 6 stage (docs/EXPERIMENT_ALPHA.md mục 2).

`forward(images, text_raw, valid_hw, boxes, t)` là MỘT lượt khử nhiễu: ảnh + text + box nhiễu
(xyxy tuyệt đối) + t -> đầu ra mọi stage. Train gọi thẳng qua DDP; suy luận gọi qua
`sample()` (DDIM, phần ảnh chỉ tính một lần mỗi ảnh). ALPHA3: `model.in_channels: 4` -> ảnh
[B,4,H,W] = RGB chuẩn hoá + density (kênh 4 của conv1 khởi tạo 0).
"""

import torch
import torch.nn as nn

from ce_localization.models.backbone import ResNet50FPN
from ce_localization.engine.diffusion import cosine_alphas_cumprod, ddim_sample
from ce_localization.models.head import DecoderHead
from ce_localization.models.memory import MemoryEncoder

__all__ = ["Detector", "build_model"]


class Detector(nn.Module):
    def __init__(self, memory="none", d_model=256, n_stage=6, n_head=4, dim_feedforward=1024,
                 dropout=0.3, text_dim=512, prior_prob=0.01, pretrained_backbone=True,
                 num_timesteps=1000, snr_scale=2.0, grid_size=None, in_channels=3):
        super().__init__()
        self.backbone = ResNet50FPN(d_model, pretrained=pretrained_backbone, in_channels=in_channels)
        self.memory = MemoryEncoder(memory, d_model, text_dim, feat_channels=d_model, feat_stride=32,
                                    grid_size=grid_size)
        self.head = DecoderHead(n_stage, d_model, n_head, dim_feedforward, dropout, prior_prob)
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


def _paper_keys(m):
    """Khoá bài add chọn giữa bản ALPHA (mặc định) và "CE-Loc gốc + R-50" (GAMMA, `models/box_policy.py`)."""
    return {"backbone_norm": m.get("backbone_norm", "frozen"), "density_init": m.get("density_init", "zero"),
            "ss_kind": m.get("ss_kind", "masked"), "box_norm": m.get("box_norm", "valid")}


def _box_policy_kw(m, d, pre):
    """Tham số `BoxPolicy` từ nhánh config model / diffusion (dùng cho `box_policy` và proposer của `propose_refine`)."""
    return dict(in_channels=m.get("in_channels", 3), pretrained_backbone=pre, fpn_dim=m.get("d_model", 256),
                vis_dim=m["vis_dim"], text_in=m.get("text_dim", 512), text_dim=m["text_proj_dim"],
                step_embed_dim=m["step_embed_dim"], down_dims=m["down_dims"], kernel_size=m["kernel_size"],
                n_groups=m["n_groups"], num_timesteps=d["num_timesteps"], beta_start=d["beta_start"],
                beta_end=d["beta_end"], ss_source=m.get("ss_source", "c5"), vision=m.get("vision", "r50_fpn"),
                ss_mask=m.get("ss_mask", False), **_paper_keys(m))


def load_proposer(model, path):
    """Nạp checkpoint CE-Loc pha 1 (train.py, `model.arch: box_policy`) vào `model.proposer` — strict, và SpatialSoftmax mask
    phải khớp (không có tham số nên strict không bắt được)."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["config"]["model"]
    if m.get("arch") != "box_policy" or m.get("ss_mask", False) != model.proposer.vision.ss_mask:
        raise ValueError(f"{path}: không phải CE-Loc pha 1 khớp proposer (arch {m.get('arch')}, ss_mask {m.get('ss_mask')})")
    model.proposer.load_state_dict(ck["model"])
    return ck.get("iter")


def build_model(cfg, pretrained_backbone=None):
    """`model.arch`: `detector` (mặc định — ALPHA / BETA, bài detect) | `box_policy` (GAMMA0 / GAMMA2 pha 1, bài add) |
    `box_refiner` (GAMMA1) | `propose_refine` (GAMMA2 / 2.1: CE-Loc pha 1 -> refine kiểu DiffusionDet, backbone chung)."""
    m, d = cfg["model"], cfg["diffusion"]
    pre = m.get("pretrained_backbone", True) if pretrained_backbone is None else pretrained_backbone
    if m.get("arch", "detector") == "box_refiner":
        from ce_localization.models.box_refiner import BoxRefiner
        return BoxRefiner(
            in_channels=m.get("in_channels", 3), pretrained_backbone=pre, d_model=m["d_model"], n_stage=m["n_stage"],
            n_head=m["n_head"], dim_feedforward=m["dim_feedforward"], dropout=m["dropout"],
            text_dim=m.get("text_dim", 512), num_timesteps=d["num_timesteps"], beta_start=d["beta_start"],
            beta_end=d["beta_end"], l1_weight=cfg["loss"]["l1_weight"], giou_weight=cfg["loss"]["giou_weight"],
            box_token=m.get("box_token", "roi"), **_paper_keys(m))
    if m.get("arch", "detector") == "box_policy":
        from ce_localization.models.box_policy import BoxPolicy
        return BoxPolicy(**_box_policy_kw(m, d, pre))
    if m.get("arch") == "propose_refine":
        from ce_localization.models.propose_refine import ProposeRefine
        model = ProposeRefine(
            _box_policy_kw(m["proposer"], d["proposer"], False), d_model=m["d_model"], n_stage=m["n_stage"],  # weight: pha 1
            dim_feedforward=m["dim_feedforward"], nhead=m["n_head"], dropout=m["dropout"], dim_dynamic=m["dim_dynamic"],
            num_dynamic=m["num_dynamic"], text_dim=m.get("text_dim", 512), num_timesteps=d["num_timesteps"],
            snr_scale=d["snr_scale"], l1_weight=cfg["loss"]["l1_weight"], giou_weight=cfg["loss"]["giou_weight"],
            freeze_proposer=m["freeze_proposer"], proposer_weight=cfg["loss"].get("proposer_weight", 1.0),
            geo=m.get("geo", False), geo_hidden=m.get("geo_hidden", 256))
        ck = cfg.get("init", {}).get("proposer_ckpt")
        if pre:                         # train: CE-Loc pha 1 ; eval (pre=False) nạp cả mô hình từ checkpoint của chính nó
            if not ck:
                raise ValueError("propose_refine cần init.proposer_ckpt (checkpoint CE-Loc pha 1) — --proposer-ckpt")
            load_proposer(model, ck)
        return model
    return Detector(
        memory=m["memory"], d_model=m["d_model"], n_stage=m["n_stage"], n_head=m["n_head"],
        dim_feedforward=m["dim_feedforward"], dropout=m["dropout"], text_dim=m.get("text_dim", 512),
        prior_prob=cfg["loss"]["prior_prob"], pretrained_backbone=pre,
        num_timesteps=d["num_timesteps"], snr_scale=d["snr_scale"], grid_size=m.get("grid_size"),
        in_channels=m.get("in_channels", 3))
