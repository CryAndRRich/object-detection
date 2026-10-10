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

`vision="r18_paper"` (NẠP checkpoint của bài qua `BoxPolicy.load_celoc_paper`, hoặc TRAIN lại CE-Loc gốc — GAMMA2 pha 1,
`ss_mask: true` = SpatialSoftmax mask phần đệm: ô đệm −inf trước softmax, lưới toạ độ giữ hệ canvas của bài): phần ảnh đúng CE-Loc gốc
(`refs/repos/Count-Editing/CE-LocModel/models/{vision_encoder,spatial_softmax}.py`) — ResNet18 4 kênh, SpatialSoftmax
KHÔNG mask trên cả canvas (meshgrid 'ij': toạ độ đầu là DỌC), Linear(1024, 128); đầu vào `data.turns.paper_inputs`
(`to_tensor`, density `.convert("L")`), box chuẩn hoá theo CANVAS (whwh = T).

DELTA (docs/EXPERIMENT_DELTA.md) — bản CE-Loc cập nhật của tác giả gốc (`refs/CE-Loc-update/models/diffusion_module.py`):
- `refiner` (dict config, `enabled`): `ClipBoxRefiner` (`models/clip_refiner.py`) TRONG mạng khử nhiễu — mỗi lần gọi ε_θ:
  x_t -> refiner -> refined = x_t + Δ ; U-Net nhận `[x_t ; refined]` (`unet_input: concat`, 8 kênh, ra ε 4 kênh) hoặc chỉ refined
  (`replace`); loss = ε-MSE + `aux_loss_weight` · MSE(refined, x0). Cần token chữ CLIP B/16 (`text_tokens`, `models/text.TextTable`).
  CLIP ViT frozen nạp từ HF lúc dựng (không vào state_dict); buffer `clip_sha` = vân tay (vision, text) kiểm khớp khi nạp.
- `use_condition: false`: điều kiện ảnh + text = 0 (embedding t giữ), refiner bỏ qua (refined = x_t), aux tắt — như tác giả. Ở đây
  refiner không được dựng và `vision` / `text_proj` đóng băng (không ai đọc) ⇒ mọi tham số train vẫn có grad (DDP không find_unused).
- Sampler vòng mock nhận `mock_steps` (mặc định 100 như bài; 1000 = chạy đủ dải t).
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import ce_localization.models.clip_refiner as clip_refiner
from ce_localization.models.backbone import ResNet50FPN
from ce_localization.models.geo import pad_objects
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


def _paper_spatial_softmax(feat, valid=None):
    """`SpatialSoftmax` của CE-Loc gốc, y từng phép tính: meshgrid 'ij' -> toạ độ đầu là DỌC; xen kẽ (x_c, y_c).
    `valid` [N,H,W] bool (GAMMA2): ô False (phần đệm) nhận −inf trước softmax — cùng lưới toạ độ canvas của bài."""
    N, C, H, W = feat.shape
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, H, device=feat.device),
                                  torch.linspace(-1, 1, W, device=feat.device), indexing="ij")
    logits = feat.reshape(N, C, -1)
    if valid is not None:
        logits = logits.masked_fill(~valid.reshape(N, 1, -1), float("-inf"))
    att = F.softmax(logits, dim=-1)
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

    def __init__(self, output_dim=128, in_channels=4, pretrained=False, ss_mask=False):
        super().__init__()
        self.ss_mask = ss_mask
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

    def features(self, x):
        """-> (C2, C3, C4, C5) của ResNet18 (64 / 128 / 256 / 512 kênh, stride 4 / 8 / 16 / 32) — GAMMA2 dựng FPN trên đây."""
        feats = []
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i >= 4:                                       # 0..3 = conv1, bn1, relu, maxpool ; 4..7 = layer1..4
                feats.append(x)
        return tuple(feats)

    def valid(self, c5, valid_hw):
        """Ô C5 thuộc vùng ảnh thật [N,H,W] nếu `ss_mask`, không thì None (bài: cả canvas)."""
        return valid_cells_mask(valid_hw, c5.shape[2], c5.shape[3], P5_STRIDE) if self.ss_mask else None

    def keypoints_flat(self, c5, valid_hw=None):
        """C5 -> [N, 2C] toạ độ SpatialSoftmax (có mask nếu `ss_mask`)."""
        return _paper_spatial_softmax(c5, self.valid(c5, valid_hw))

    def forward(self, x, valid_hw=None):
        return self.projection(self.keypoints_flat(self.backbone(x), valid_hw))

    @torch.no_grad()
    def keypoints(self, x, valid_hw=None):
        """Soi SpatialSoftmax: -> (toạ độ [B,C,2] theo thứ tự (DỌC, NGANG) trong [−1, 1] — meshgrid 'ij' của bài —,
        attention softmax [B,C,H,W]). Toạ độ −1 / +1 = TÂM ô đầu / ô cuối của lưới H×W."""
        feat = self.backbone(x)
        N, C, H, W = feat.shape
        logits = feat.reshape(N, C, -1)
        valid = self.valid(feat, valid_hw)
        if valid is not None:
            logits = logits.masked_fill(~valid.reshape(N, 1, -1), float("-inf"))
        att = F.softmax(logits, dim=-1)
        return _paper_spatial_softmax(feat, valid).reshape(N, C, 2), att.reshape(N, C, H, W)


class BoxPolicy(nn.Module):
    def __init__(self, in_channels=3, pretrained_backbone=True, fpn_dim=256, vis_dim=128, text_in=512,
                 text_dim=128, step_embed_dim=256, down_dims=(64, 128, 256), kernel_size=3, n_groups=8,
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02, vision="r50_fpn", ss_source="c5",
                 backbone_norm="frozen", density_init="zero", ss_kind="masked", box_norm="valid", ss_mask=False,
                 obj_attn=False, obj_heads=4, obj_max=300, refiner=None, use_condition=True):
        super().__init__()
        if vision not in VISIONS:
            raise ValueError(f"vision {vision!r} không thuộc {VISIONS}")
        if ss_source not in SS_SOURCES or ss_kind not in SS_KINDS or box_norm not in BOX_NORMS:
            raise ValueError(f"ss_source {ss_source!r} / ss_kind {ss_kind!r} / box_norm {box_norm!r} không thuộc "
                             f"{SS_SOURCES} / {SS_KINDS} / {BOX_NORMS}")
        self.vision_kind, self.ss_source, self.ss_kind = vision, ss_source, ss_kind
        self.box_norm = "canvas" if vision == "r18_paper" else box_norm
        if vision == "r18_paper":
            self.vision = PaperVisionEncoder(vis_dim, in_channels, pretrained_backbone, ss_mask)
        else:
            self.backbone = ResNet50FPN(fpn_dim, pretrained=pretrained_backbone, in_channels=in_channels,
                                        norm=backbone_norm, density_init=density_init)
            self.vis_proj = nn.Linear(2 * (C5_CHANNELS if ss_source == "c5" else fpn_dim), vis_dim)
        self.text_proj = nn.Sequential(nn.Linear(text_in, text_dim), nn.Mish())
        self.cond_dim = vis_dim + text_dim
        # DELTA: refiner của tác giả (`ObjectPlacementPolicy`): concat => U-Net nhận [x_t ; refined] (8 kênh), vẫn ra ε 4 kênh
        rcfg = {**clip_refiner.REFINER_DEFAULTS, **(refiner or {})} if refiner and refiner.get("enabled") else None
        self.use_refiner, self.use_condition = rcfg is not None, bool(use_condition)
        self.refiner_unet_input = rcfg["unet_input"] if rcfg else None
        self.refiner_aux_weight = float(rcfg["aux_loss_weight"]) if rcfg else 0.0
        if rcfg:
            if self.refiner_unet_input not in ("concat", "replace"):
                raise ValueError(f"refiner.unet_input {self.refiner_unet_input!r} không thuộc ('concat', 'replace')")
            if self.box_norm != "canvas":
                raise ValueError("refiner (RoIAlign trên lưới CLIP của cả canvas) cần box chuẩn hoá theo canvas (vision r18_paper)")
            if obj_attn:
                raise ValueError("refiner + obj_attn chưa hỗ trợ (luồng vật của U-Net nhận box 4 số)")
            if rcfg["use_density"] and in_channels != 4:
                raise ValueError("refiner.use_density cần ảnh 4 kênh (RGB + density)")
        unet_in = 8 if self.refiner_unet_input == "concat" else 4
        self.noise_net = ConditionalUnet1D(unet_in, self.cond_dim, step_embed_dim, tuple(down_dims),
                                           kernel_size, n_groups, obj_attn=obj_attn, obj_heads=obj_heads, output_dim=4)
        self.obj_attn, self.obj_max = bool(obj_attn), obj_max
        self.num_timesteps = num_timesteps
        self.register_buffer("alphas_cumprod", linear_alphas_cumprod(num_timesteps, beta_start, beta_end),
                             persistent=False)
        # dựng SAU noise_net: khởi tạo ResNet / U-Net trùng bản không refiner (DELTA1 vs DELTA1.1 so cặp)
        self.refiner = None
        if rcfg and self.use_condition:
            vision_model, text_hidden = clip_refiner.load_clip(rcfg["clip_model_name"])
            self.refiner = clip_refiner.ClipBoxRefiner(rcfg, vision_model, text_hidden)
            sha = torch.zeros(2, 32, dtype=torch.uint8)
            sha[0] = torch.tensor(list(clip_refiner.weights_sha256(vision_model)), dtype=torch.uint8)
            self.register_buffer("clip_sha", sha)                    # [vân tay CLIP vision ; vân tay CLIP text] (text đặt sau)
        if not self.use_condition:                                   # không ai đọc ảnh / text: đóng băng (mọi tham số train có grad)
            for mod in (self.vision if vision == "r18_paper" else self.backbone, self.text_proj):
                mod.requires_grad_(False)
            if vision != "r18_paper":
                self.vis_proj.requires_grad_(False)
        self.track_refine = False
        self._stats, self._refine_track = {}, {}

    def text_emb(self, text_raw, null_text=False):
        """[B, text_dim]; `null_text=True` -> 0 (bỏ hẳn điều kiện text — chỉ để soi model, không dùng khi train)."""
        e = self.text_proj(text_raw)
        return torch.zeros_like(e) if null_text else e

    def condition(self, images, text_raw, valid_hw, null_text=False):
        """-> cond [B, vis_dim + text_dim]. Phần ảnh không phụ thuộc t: tính MỘT lần mỗi ảnh. `use_condition: false` -> 0, không chạy
        encoder (như `encode_condition` của tác giả)."""
        if not self.use_condition:
            return images.new_zeros(images.shape[0], self.cond_dim, dtype=torch.float32)
        if self.vision_kind == "r18_paper":                       # bài: không mask vùng thật; GAMMA2: ss_mask
            return torch.cat([self.vision(images, valid_hw), self.text_emb(text_raw, null_text)], dim=-1)
        f = self.backbone.forward_c5(images) if self.ss_source == "c5" else self.backbone.forward_p5(images)
        return torch.cat([self.vis_proj(spatial_keypoints(f, valid_hw, self.ss_kind)), self.text_emb(text_raw, null_text)],
                         dim=-1)

    def cond_from_keypoints(self, kp, text_raw, null_text=False):
        """ResNet18 của bài: toạ độ SpatialSoftmax [B, 2C] -> cond — để GAMMA2 dùng chung backbone với model refine."""
        return torch.cat([self.vision.projection(kp), self.text_emb(text_raw, null_text)], dim=-1)

    @property
    def needs_objects(self):
        """Train / suy luận cần `objects` (box các vật đang có) — GAMMA4 (`obj_attn`)."""
        return self.obj_attn

    @property
    def needs_text_tokens(self):
        """Train / suy luận cần `text_tokens` (token chữ CLIP B/16, `TextTable.tokens`) — DELTA (refiner bật, có điều kiện)."""
        return self.refiner is not None

    # ------------------------------------------------------------------ DELTA: vân tay CLIP frozen (không nằm trong state_dict)
    @staticmethod
    def _sha_row(sha):
        return torch.tensor(list(sha), dtype=torch.uint8)

    def set_text_fingerprint(self, sha):
        """Ghi vân tay CLIP text (bytes 32, `TextTable.token_sha`) vào `clip_sha[1]` — gọi TRƯỚC khi nạp checkpoint để nạp tự kiểm."""
        if self.refiner is not None:
            self.clip_sha[1] = self._sha_row(sha).to(self.clip_sha.device)

    def check_text_fingerprint(self, sha):
        """Sau khi nạp checkpoint (eval): CLIP text vừa tải phải đúng bản lúc train."""
        if self.refiner is not None and self.clip_sha[1].any() and \
                not torch.equal(self.clip_sha[1].cpu(), self._sha_row(sha)):
            raise RuntimeError("CLIP text (token chữ của refiner) KHÁC bản lúc train checkpoint — kiểm clip_model_name / HF cache")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """`clip_sha`: hàng (vision / text) đã có giá trị ở model phải TRÙNG checkpoint (CLIP tải từ HF đúng bản đã train); hàng còn
        0 (vd. text chưa đặt khi eval) nhận giá trị của checkpoint."""
        key = prefix + "clip_sha"
        cur = self.clip_sha.clone() if self.refiner is not None else None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if cur is None or key not in state_dict:
            return
        inc = state_dict[key].to(cur.device)
        for r, what in enumerate(("CLIP ViT (vision)", "CLIP text")):
            if cur[r].any() and inc[r].any() and not torch.equal(cur[r], inc[r]):
                error_msgs.append(f"{key}: vân tay {what} của model (tải từ HF) KHÁC checkpoint — CLIP frozen không đúng bản đã train")
            if cur[r].any() and not inc[r].any():
                self.clip_sha[r] = cur[r]

    # ------------------------------------------------------------------ DELTA: refiner trong mạng khử nhiễu
    def refiner_context(self, images, text_tokens):
        """Ngữ cảnh refiner (một lần mỗi ảnh): RGB = 3 kênh đầu, density = kênh 4 (cùng [0, 1] kiểu bài như tác giả)."""
        if self.refiner is None:
            return None
        if text_tokens is None:
            raise ValueError("model.refiner cần `text_tokens` (token chữ CLIP, TextTable.tokens) khi train / suy luận")
        dens = images[:, 3:4] if images.shape[1] == 4 else images.new_zeros(images.shape[0], 1, *images.shape[2:])
        return self.refiner.encode_context(images[:, :3], dens, *text_tokens)

    def predict_noise(self, x, t, cond, ctx=None, ctx_index=None, obj=None):
        """x [R,4] box nhiễu, t [R], cond [R, D], ctx (`refiner_context`, C hàng) + ctx_index [R] | None, obj (token vật, mask) | None
        -> (ε̂ [R,4], refined [R,4] | None) — `predict_noise` của tác giả. Không refiner: U-Net trên x như bài."""
        unet_in, refined = x, None
        if self.use_refiner:
            refined = self.refiner(x, t, ctx, ctx_index) if ctx is not None else x
            unet_in = torch.cat([x, refined.to(x.dtype)], dim=-1) if self.refiner_unet_input == "concat" else refined
        eps = self.noise_net(unet_in.unsqueeze(1), t, cond, *(obj or (None, None))).squeeze(1)
        if refined is not None and ctx is not None and self.track_refine:
            self._track_delta(t, (refined - x).abs().mean(-1))
        return eps, refined

    def _track_delta(self, t, d):
        """Suy luận (eval): cộng dồn TB |Δ| của refiner theo khoảng t (100 bước / khoảng)."""
        for b in torch.unique(t // 100).tolist():
            sel = (t // 100) == b
            s, n = self._refine_track.get(b, (0.0, 0))
            self._refine_track[b] = (s + float(d[sel].sum()), n + int(sel.sum()))

    def pop_refine_stats(self):
        """-> {"t<a>-<b>": TB |Δ|} (khoảng t) của các lần lấy mẫu từ lần gọi trước | None."""
        if not self._refine_track:
            return None
        out = {f"t{b * 100}-{b * 100 + 99}": s / max(n, 1) for b, (s, n) in sorted(self._refine_track.items())}
        self._refine_track = {}
        return out

    def pop_stats(self):
        """-> thống kê loss lần forward train gần nhất (loss_eps, loss_aux, delta_abs, aux_ratio — DELTA) rồi xoá."""
        st, self._stats = self._stats, {}
        return st

    def object_tokens(self, objects, valid_hw, canvas, cond, generator=None, cap=False):
        """objects: list B tensor [Mᵢ,4] xyxy pixel canvas -> (token 6 tầng, mask) cho `noise_net` (mã hoá gương, t = 0).
        Box vật chuẩn hoá GIỐNG đích (`boxes_to_unit` theo `box_norm`). `cap`: > obj_max vật thì lấy ngẫu nhiên obj_max (train)."""
        if objects is None:
            raise ValueError("model.obj_attn cần `objects` (box các vật đang có) khi train / suy luận")
        dev = cond.device
        hw = valid_hw.float().cpu()
        whwh = (torch.full((len(objects), 4), float(canvas)) if self.box_norm == "canvas" else
                torch.stack([hw[:, 1], hw[:, 0], hw[:, 1], hw[:, 0]], 1))
        units = []
        for i, o in enumerate(objects):
            o = torch.as_tensor(o, dtype=torch.float32).reshape(-1, 4).cpu()
            if cap and len(o) > self.obj_max:
                seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator, device=generator.device)) if generator \
                    is not None else 0
                o = o[torch.randperm(len(o), generator=torch.Generator().manual_seed(seed))[: self.obj_max]]
            units.append(boxes_to_unit(o, whwh[i]))
        objs, mask = pad_objects(units, dev, min_m=1)
        return self.noise_net.encode_objects(objs, mask, cond), mask

    def forward(self, images, text_raw, valid_hw, x0, k=1, generator=None, objects=None, text_tokens=None):
        """ε-MSE như `ObjectPlacementPolicy.compute_loss` của bài, k bộ (t, ε) mỗi ảnh.
        x0 [B,4] trong [−1,1]. `objects` (GAMMA4): list B box vật xyxy pixel canvas. `text_tokens` (DELTA): (token, mask) của
        `TextTable.tokens`. -> loss vô hướng (trung bình trên B·k·4; DELTA + aux)."""
        cond = self.condition(images, text_raw, valid_hw)
        obj = (self.object_tokens(objects, valid_hw, images.shape[-1], cond, generator, cap=True) if self.obj_attn
               else None)
        return self.eps_loss(cond, x0, k, generator, obj, self.refiner_context(images, text_tokens))

    @staticmethod
    def _repeat_obj(obj, n):
        return None if obj is None else ([t.repeat_interleave(n, 0) for t in obj[0]], obj[1].repeat_interleave(n, 0))

    def eps_loss(self, cond, x0, k=1, generator=None, obj=None, ctx=None):
        """ε-MSE từ cond [B, D] đã tính sẵn; obj = (token, mask) của `object_tokens` | None; ctx (DELTA) = `refiner_context` (B hàng)
        — k bộ (t, ε) / ảnh có t khác nhau nên ngữ cảnh nhân thành B·k hàng (đường tách đôi cần một t mỗi hàng ngữ cảnh).
        DELTA: + `aux_loss_weight` · MSE(refined, x0) như `compute_loss` của tác giả; thống kê đọc qua `pop_stats()`."""
        obj = self._repeat_obj(obj, k)
        B = cond.shape[0]
        cond = cond.repeat_interleave(k, dim=0)
        x0 = x0.repeat_interleave(k, dim=0)
        dev = x0.device
        if ctx is not None and k > 1:
            ctx = self.refiner.index_context(ctx, torch.arange(B, device=dev).repeat_interleave(k))
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=dev, generator=generator)
        noise = torch.randn(x0.shape, device=dev, generator=generator)
        ab = self.alphas_cumprod[t].unsqueeze(-1)
        noisy = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        pred, refined = self.predict_noise(noisy, t, cond, ctx, None, obj)
        loss = F.mse_loss(pred, noise)
        if refined is not None and ctx is not None:
            with torch.autocast(device_type=dev.type, enabled=False):          # refined fp32 (refiner tắt autocast)
                aux = F.mse_loss(refined, x0.to(refined.dtype))
                self._stats = {"loss_eps": loss.detach().float(), "loss_aux": aux.detach().float(),
                               "delta_abs": (refined - noisy.to(refined.dtype)).abs().mean().detach().float(),
                               "aux_ratio": (aux / F.mse_loss(noisy.to(refined.dtype), x0.to(refined.dtype)).clamp_min(1e-12))
                               .detach().float()}
            if self.refiner_aux_weight > 0:
                loss = loss + self.refiner_aux_weight * aux
        return loss

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, n_samples, generator=None, sampler="ddpm", record=None,
               null_text=False, objects=None, use_objects=True, text_tokens=None, use_refiner=True, mock_steps=100):
        """n_samples mẫu ĐỘC LẬP mỗi ảnh (ảnh mã hoá một lần, các mẫu khử nhiễu song song).
        -> [B, n_samples, 4] trong [−1,1]; `record` (tập t / "all") -> (box, quỹ đạo: list {t, x_t, x0_hat} [B,n,4]).
        GAMMA4: `objects` = box vật (token tính MỘT lần mỗi ảnh, dùng cho mọi bước); `use_objects=False` = tắt (`_noobj`).
        DELTA: `text_tokens` (token chữ CLIP); `use_refiner=False` = bỏ refiner (refined = x_t — `_norefine`, ngoài phân bố train).
        `mock_steps`: số bước vòng mock (t = mock_steps − 1 .. 0)."""
        cond = self.condition(images, text_raw, valid_hw, null_text)
        obj = (self.object_tokens(objects, valid_hw, images.shape[-1], cond) if self.obj_attn and use_objects else None)
        ctx = self.refiner_context(images, text_tokens) if self.refiner is not None and use_refiner else None
        return self.sample_from_cond(cond, n_samples, generator, sampler, record, obj=obj, ctx=ctx, mock_steps=mock_steps)

    @torch.no_grad()
    def sample_from_cond(self, cond, n_samples, generator=None, sampler="ddpm", record=None, obj=None, ctx=None,
                         mock_steps=100):
        """Như `sample` nhưng từ cond [B, D] đã tính sẵn (+ token vật của `object_tokens`, ngữ cảnh refiner `refiner_context`).
        Mọi hàng cùng t ở mỗi bước (mock / DDPM) ⇒ refiner tính luồng ngữ cảnh một lần mỗi ảnh (`ctx_index`)."""
        B = cond.shape[0]
        cond = cond.repeat_interleave(n_samples, dim=0)
        ot, om = self._repeat_obj(obj, n_samples) or (None, None)
        ci = None if ctx is None else torch.arange(B, device=cond.device).repeat_interleave(n_samples)
        fn = lambda x, t: self.predict_noise(x, t, cond, ctx, ci, None if ot is None else (ot, om))[0]  # noqa: E731
        run = ddpm_sample if sampler == "ddpm" else partial(mock_sample, steps=mock_steps)
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
