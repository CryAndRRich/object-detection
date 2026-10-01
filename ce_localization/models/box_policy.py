"""BoxPolicy — bộ sinh MỘT box kiểu CE-Loc gốc / Diffusion Policy cho bài ADD (docs/EXPERIMENT_GAMMA.md).

    ảnh [B,3|4,T,T] -> R-50 của ALPHA (`models/backbone.py`) -> C5 = layer4 [B,2048,T/32,T/32] (`ss_source: c5`,
        như bài: SpatialSoftmax trên đầu ra layer cuối của ResNet ĐÃ PRETRAIN)
        -> SpatialSoftmax có mask vùng thật (`models/memory.masked_spatial_softmax`)
        -> 2048 điểm (x, y) -> Linear(4096, vis_dim)                                  ┐
    text: CLIP pooler 512 (bảng `models/text.TextTable`) -> Linear(512, text_dim) -> Mish ┤ cond [B, vis+text]
    box nhiễu v_t [B,1,4] + t -> U-Net 1D (`models/unet1d.py`), FiLM cộng bias theo cond -> ε̂

Khác CE-Loc gốc ĐÚNG ở phần ảnh (người dùng chốt 2026-10-01): ResNet18 (BN train, `to_tensor`, SpatialSoftmax 512 kênh
C5, không mask) -> R-50 của ALPHA (FrozenBN, chuẩn hoá ImageNet, SpatialSoftmax 2048 kênh C5 có mask vùng thật — vì box
chuẩn hoá theo vùng thật, density kênh 4 khởi tạo 0). Text, U-Net 1D, lịch β tuyến tính 1e-4..0,02, ε-MSE giữ như bài.
`ss_source: p5` (P5 của FPN, 256 kênh) là bản đầu — G3 2026-10-01 HỎNG: FPN khởi tạo ngẫu nhiên ⇒ softmax phẳng, nhánh ảnh
gần như không có gradient; giữ khoá chỉ để tái lập.

Box trong không gian khuếch tán: cxcywh chia (nw, nh) của VÙNG ẢNH THẬT (`box_norm: valid`, quy ước `whwh` của ALPHA) hoặc
chia CANVAS như bài (`box_norm: canvas`) rồi `·2 − 1` -> [−1, 1] (`norm_whwh`).

"CE-Loc gốc + R-50" (người dùng chốt 2026-10-01, sau khi GAMMA0 bản ALPHA hỏng — EXPERIMENT_GAMMA mục 12.2): config đặt
`backbone_norm: bn` (BN train như bài), `density_init: rgb_mean`, `ss_kind: paper` (SpatialSoftmax KHÔNG mask trên cả
canvas, meshgrid 'ij', toạ độ [−1, 1] — đúng `_paper_spatial_softmax`), `box_norm: canvas`, cùng `data.input_style: paper`
(`to_tensor`, density `.convert("L")`). Mặc định của constructor giữ bản cũ (FrozenBN / mask / vùng thật) để tái lập.
Train: mỗi ảnh rút `noise_per_image` bộ (t, ε) độc lập — backbone chỉ tính MỘT lần mỗi ảnh, U-Net rất rẻ.

`vision="r18_paper"` (chỉ để NẠP checkpoint của bài, `BoxPolicy.load_celoc_paper`): phần ảnh đúng CE-Loc gốc
(`refs/repos/Count-Editing/CE-LocModel/models/{vision_encoder,spatial_softmax}.py`) — ResNet18 4 kênh, SpatialSoftmax
KHÔNG mask trên cả canvas (meshgrid 'ij': toạ độ đầu là DỌC), Linear(1024, 128); đầu vào `data.turns.paper_inputs`
(`to_tensor`, density `.convert("L")`), box chuẩn hoá theo CANVAS (whwh = T).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ce_localization.models.backbone import ResNet50FPN
from ce_localization.models.memory import masked_spatial_softmax, valid_cells_mask
from ce_localization.models.unet1d import ConditionalUnet1D
from ce_localization.utils.diffusion_math import linear_alphas_cumprod

__all__ = ["BoxPolicy", "PaperVisionEncoder", "boxes_to_unit", "unit_to_boxes", "norm_whwh", "ddpm_sample", "mock_sample",
           "VISIONS", "SS_KINDS", "BOX_NORMS"]

VISIONS = ("r50_fpn", "r18_paper")

P5_STRIDE = 32
C5_CHANNELS = 2048
SS_SOURCES = ("c5", "p5")
SS_KINDS = ("masked", "paper")
BOX_NORMS = ("valid", "canvas")


def norm_whwh(model, whwh, canvas):
    """whwh vùng thật [B,4] -> whwh dùng để chuẩn hoá box của `model` (`box_norm`): giữ nguyên, hoặc canvas T."""
    return torch.full_like(whwh, float(canvas)) if getattr(model, "box_norm", "valid") == "canvas" else whwh


def boxes_to_unit(boxes_xyxy, whwh):
    """xyxy pixel canvas [...,4] + whwh [...,4] (nw, nh, nw, nh) -> cxcywh/(nw,nh) ·2 − 1."""
    b = boxes_xyxy / whwh
    c = torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                     b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], dim=-1)
    return c * 2 - 1


def unit_to_boxes(u, whwh):
    """Ngược `boxes_to_unit`: [...,4] trong [−1,1] -> xyxy pixel canvas. KHÔNG kẹp (w, h âm giữ nguyên để
    đo tỉ lệ box suy biến; IoU tự kẹp w, h về 0)."""
    c = (u + 1) / 2
    xyxy = torch.stack([c[..., 0] - c[..., 2] / 2, c[..., 1] - c[..., 3] / 2,
                        c[..., 0] + c[..., 2] / 2, c[..., 1] + c[..., 3] / 2], dim=-1)
    return xyxy * whwh


def _snap(traj, record, t, x, eps, ab_t):
    """Ghi (t, x_t ĐẦU VÀO bước t, x̂0 suy từ ε̂) nếu t thuộc `record` (None = không ghi, "all" = mọi bước)."""
    if traj is not None and (record == "all" or t in record):
        traj.append({"t": int(t), "x_t": x.detach().cpu().clone(),
                     "x0_hat": ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).detach().cpu()})


@torch.no_grad()
def ddpm_sample(eps_fn, n_rows, alphas_cumprod, generator=None, device="cpu", record=None):
    """DDPM tổ tiên đúng công thức (Ho et al. 2020), T bước, phương sai β̃_t, không kẹp — bản `sample_ddpm`
    đúng của CE-Loc gốc viết lại (bài dùng vòng "mock" `x -= eps/100`, không phải DDPM).
    eps_fn(x [R,4], t [R] long) -> ε̂ [R,4]. -> x0 [R,4]; `record` (tập t hoặc "all") -> (x0, quỹ đạo)."""
    ab = alphas_cumprod.to(device)
    ab_prev = torch.cat([ab.new_ones(1), ab[:-1]])
    alphas = ab / ab_prev
    betas = 1 - alphas
    traj = [] if record is not None else None
    x = torch.randn((n_rows, 4), device=device, generator=generator)
    for t in reversed(range(ab.shape[0])):
        tb = torch.full((n_rows,), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tb)
        _snap(traj, record, t, x, eps, ab[t])
        mean = (x - betas[t] / torch.sqrt(1 - ab[t]) * eps) / torch.sqrt(alphas[t])
        if t > 0:
            var = betas[t] * (1 - ab_prev[t]) / (1 - ab[t])
            x = mean + torch.sqrt(var) * torch.randn(x.shape, device=device, generator=generator)
        else:
            x = mean
    return x if traj is None else (x, traj)


@torch.no_grad()
def mock_sample(eps_fn, n_rows, alphas_cumprod, steps=100, generator=None, device="cpu", record=None):
    """Vòng lấy mẫu CE-Loc gốc DÙNG THẬT (`inference.py` / `test_mul_box.py`): t = steps−1..0, `x -= eps/steps` —
    KHÔNG phải DDPM (model train với t trong [0, T)). x̂0 trong quỹ đạo suy theo ᾱ_t của lịch train (chỉ để xem)."""
    ab = alphas_cumprod.to(device)
    if steps > ab.shape[0]:
        raise ValueError(f"mock {steps} bước nhưng lịch train chỉ có T = {ab.shape[0]}")
    traj = [] if record is not None else None
    x = torch.randn((n_rows, 4), device=device, generator=generator)
    for t in reversed(range(steps)):
        tb = torch.full((n_rows,), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tb)
        _snap(traj, record, t, x, eps, ab[t])
        x = x - eps / steps
    return x if traj is None else (x, traj)


def _paper_spatial_softmax(feat):
    """`SpatialSoftmax` của CE-Loc gốc, y từng phép tính: meshgrid 'ij' -> toạ độ đầu là DỌC; xen kẽ (x_c, y_c)."""
    N, C, H, W = feat.shape
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, H, device=feat.device),
                                  torch.linspace(-1, 1, W, device=feat.device), indexing="ij")
    att = F.softmax(feat.reshape(N, C, -1), dim=-1)
    ex = torch.sum(pos_x.reshape(H * W) * att, dim=-1, keepdim=True)
    ey = torch.sum(pos_y.reshape(H * W) * att, dim=-1, keepdim=True)
    return torch.cat([ex, ey], dim=-1).reshape(N, -1)


def spatial_keypoints(feat, valid_hw, kind):
    """feat [B,C,H,W] (stride 32) -> [B, 2C] toạ độ SpatialSoftmax. `paper`: y bài (không mask, cả canvas, (dọc, ngang)
    trong [−1, 1]); `masked`: che ô đệm, (x, y) chia vùng thật (`models/memory.masked_spatial_softmax`)."""
    if kind == "paper":
        return _paper_spatial_softmax(feat)
    valid = valid_cells_mask(valid_hw, feat.shape[2], feat.shape[3], P5_STRIDE)
    return masked_spatial_softmax(feat, valid, valid_hw, P5_STRIDE).flatten(1)


class PaperVisionEncoder(nn.Module):
    """`SpatialVisualEncoder` của CE-Loc gốc: ResNet18 (BatchNorm) bỏ avgpool + fc -> SpatialSoftmax -> Linear(1024, D).
    Tên module (`backbone`, `projection`) trùng bản gốc để nạp checkpoint. in_channels 4: kênh density khởi tạo bằng
    TB weight RGB như bài (chỉ có ý nghĩa khi train từ đầu)."""

    def __init__(self, output_dim=128, in_channels=4, pretrained=False):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        if in_channels == 4:
            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv1.weight[:, :3] = r.conv1.weight
                conv1.weight[:, 3:] = r.conv1.weight.mean(dim=1, keepdim=True)
            r.conv1 = conv1
        self.in_channels = in_channels
        self.backbone = nn.Sequential(*list(r.children())[:-2])
        self.projection = nn.Linear(512 * 2, output_dim)

    def forward(self, x):
        return self.projection(_paper_spatial_softmax(self.backbone(x)))

    @torch.no_grad()
    def keypoints(self, x):
        """Soi SpatialSoftmax: -> (toạ độ [B,C,2] theo thứ tự (DỌC, NGANG) trong [−1, 1] — meshgrid 'ij' của bài —,
        attention softmax [B,C,H,W]). Toạ độ −1 / +1 = TÂM ô đầu / ô cuối của lưới H×W."""
        feat = self.backbone(x)
        N, C, H, W = feat.shape
        att = F.softmax(feat.reshape(N, C, -1), dim=-1)
        return _paper_spatial_softmax(feat).reshape(N, C, 2), att.reshape(N, C, H, W)


class BoxPolicy(nn.Module):
    def __init__(self, in_channels=3, pretrained_backbone=True, fpn_dim=256, vis_dim=128, text_in=512,
                 text_dim=128, step_embed_dim=256, down_dims=(64, 128, 256), kernel_size=3, n_groups=8,
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02, vision="r50_fpn", ss_source="c5",
                 backbone_norm="frozen", density_init="zero", ss_kind="masked", box_norm="valid"):
        super().__init__()
        if vision not in VISIONS:
            raise ValueError(f"vision {vision!r} không thuộc {VISIONS}")
        if ss_source not in SS_SOURCES or ss_kind not in SS_KINDS or box_norm not in BOX_NORMS:
            raise ValueError(f"ss_source {ss_source!r} / ss_kind {ss_kind!r} / box_norm {box_norm!r} không thuộc "
                             f"{SS_SOURCES} / {SS_KINDS} / {BOX_NORMS}")
        self.vision_kind, self.ss_source, self.ss_kind = vision, ss_source, ss_kind
        self.box_norm = "canvas" if vision == "r18_paper" else box_norm
        if vision == "r18_paper":
            self.vision = PaperVisionEncoder(vis_dim, in_channels, pretrained_backbone)
        else:
            self.backbone = ResNet50FPN(fpn_dim, pretrained=pretrained_backbone, in_channels=in_channels,
                                        norm=backbone_norm, density_init=density_init)
            self.vis_proj = nn.Linear(2 * (C5_CHANNELS if ss_source == "c5" else fpn_dim), vis_dim)
        self.text_proj = nn.Sequential(nn.Linear(text_in, text_dim), nn.Mish())
        self.noise_net = ConditionalUnet1D(4, vis_dim + text_dim, step_embed_dim, tuple(down_dims),
                                           kernel_size, n_groups)
        self.num_timesteps = num_timesteps
        self.register_buffer("alphas_cumprod", linear_alphas_cumprod(num_timesteps, beta_start, beta_end),
                             persistent=False)

    def text_emb(self, text_raw, null_text=False):
        """[B, text_dim]; `null_text=True` -> 0 (bỏ hẳn điều kiện text — chỉ để soi model, không dùng khi train)."""
        e = self.text_proj(text_raw)
        return torch.zeros_like(e) if null_text else e

    def condition(self, images, text_raw, valid_hw, null_text=False):
        """-> cond [B, vis_dim + text_dim]. Phần ảnh không phụ thuộc t: tính MỘT lần mỗi ảnh."""
        if self.vision_kind == "r18_paper":                                        # bài: không mask vùng thật
            return torch.cat([self.vision(images), self.text_emb(text_raw, null_text)], dim=-1)
        f = self.backbone.forward_c5(images) if self.ss_source == "c5" else self.backbone.forward_p5(images)
        return torch.cat([self.vis_proj(spatial_keypoints(f, valid_hw, self.ss_kind)), self.text_emb(text_raw, null_text)],
                         dim=-1)

    def forward(self, images, text_raw, valid_hw, x0, k=1, generator=None):
        """ε-MSE như `ObjectPlacementPolicy.compute_loss` của bài, k bộ (t, ε) mỗi ảnh.
        x0 [B,4] trong [−1,1]. -> loss vô hướng (trung bình trên B·k·4)."""
        cond = self.condition(images, text_raw, valid_hw).repeat_interleave(k, dim=0)
        x0 = x0.repeat_interleave(k, dim=0)
        dev = x0.device
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=dev, generator=generator)
        noise = torch.randn(x0.shape, device=dev, generator=generator)
        ab = self.alphas_cumprod[t].unsqueeze(-1)
        noisy = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred = self.noise_net(noisy.unsqueeze(1), t, cond).squeeze(1)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, n_samples, generator=None, sampler="ddpm", record=None,
               null_text=False):
        """n_samples mẫu ĐỘC LẬP mỗi ảnh (ảnh mã hoá một lần, các mẫu khử nhiễu song song).
        -> [B, n_samples, 4] trong [−1,1]; `record` (tập t / "all") -> (box, quỹ đạo: list {t, x_t, x0_hat} [B,n,4])."""
        B = images.shape[0]
        cond = self.condition(images, text_raw, valid_hw, null_text).repeat_interleave(n_samples, dim=0)
        fn = lambda x, t: self.noise_net(x.unsqueeze(1), t, cond).squeeze(1)  # noqa: E731
        run = ddpm_sample if sampler == "ddpm" else mock_sample
        out = run(fn, cond.shape[0], self.alphas_cumprod, generator=generator, device=cond.device, record=record)
        if record is None:
            return out.view(B, n_samples, 4)
        x, traj = out
        return x.view(B, n_samples, 4), [{**s, "x_t": s["x_t"].view(B, n_samples, 4),
                                          "x0_hat": s["x0_hat"].view(B, n_samples, 4)} for s in traj]

    @classmethod
    def load_celoc_paper(cls, path, map_location="cpu"):
        """Checkpoint CE-Loc gốc của bài (`model_state_dict` của `ObjectPlacementPolicy`) -> (model, state CLIP text
        của checkpoint, thông tin). Đổi tên: `vision_encoder.*` -> `vision.*`, `text_encoder.projection` ->
        `text_proj.0`, `noise_net.*` giữ; `text_encoder.backbone.*` (CLIP frozen) trả riêng để mã hoá tên lớp; số bước
        T và lịch β đọc từ buffer `alphas_cumprod` và PHẢI khớp lịch tuyến tính. Key thừa / thiếu nào khác ⇒ lỗi."""
        ck = path if isinstance(path, dict) else torch.load(path, map_location=map_location, weights_only=False)
        sd = ck.get("model_state_dict", ck)
        in_ch = int(sd["vision_encoder.backbone.0.weight"].shape[1])
        ab = sd.get("alphas_cumprod")
        T = int(ab.shape[0]) if ab is not None else 1000
        model = cls(in_channels=in_ch, pretrained_backbone=False, num_timesteps=T, vision="r18_paper")
        mine, clip, other = {}, {}, []
        for k, v in sd.items():
            if k.startswith("vision_encoder."):
                mine["vision." + k[len("vision_encoder."):]] = v
            elif k.startswith("text_encoder.projection."):
                mine["text_proj.0." + k[len("text_encoder.projection."):]] = v
            elif k.startswith("text_encoder.backbone."):
                clip[k[len("text_encoder.backbone."):]] = v
            elif k.startswith("noise_net."):
                mine[k] = v
            else:
                other.append(k)
        missing, unexpected = model.load_state_dict(mine, strict=False)
        bad = list(missing) + list(unexpected) + [k for k in other if k != "alphas_cumprod"]
        if bad:
            raise RuntimeError(f"checkpoint lệch kiến trúc CE-Loc gốc: {bad[:10]} ({len(bad)} key)")
        sched_err = None if ab is None else float((ab.float() - model.alphas_cumprod.float()).abs().max())
        if sched_err is not None and sched_err > 1e-5:
            raise RuntimeError(f"alphas_cumprod của checkpoint lệch lịch β tuyến tính 1e-4..0,02 (max {sched_err:.2e})")
        info = {"in_channels": in_ch, "num_timesteps": T, "schedule_max_err": sched_err, "n_clip_keys": len(clip),
                "epoch": ck.get("epoch"), "loss": ck.get("loss")}
        return model, clip, info
