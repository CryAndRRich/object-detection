"""BoxPolicy — bộ sinh MỘT box kiểu CE-Loc gốc / Diffusion Policy cho bài ADD (docs/EXPERIMENT_GAMMA.md, EXPERIMENT_DELTA.md).

    ảnh [B,3|4,T,T] (kiểu bài: `to_tensor` /255 + density `.convert("L")`) -> backbone ảnh -> SpatialSoftmax kiểu bài
        (`paper_spatial_softmax`: meshgrid 'ij', toạ độ [−1, 1] trên cả canvas) -> Linear(2C, vis_dim)          ┐
    text: CLIP pooler 512 (bảng `models/text.TextTable`) -> Linear(512, text_dim) -> Mish                       ┤ cond
    box nhiễu v_t [B,1,4] + t -> U-Net 1D (`models/unet1d.py`), FiLM theo [emb t ; cond] -> ε̂

Box trong không gian khuếch tán: cxcywh chia CANVAS như bài, `·2 − 1` -> [−1, 1] (`norm_whwh`). Lịch β tuyến tính 1e-4..0,02,
ε-MSE như bài. Train: mỗi ảnh rút `noise_per_image` bộ (t, ε) độc lập — backbone chỉ tính MỘT lần mỗi ảnh, U-Net rất rẻ.

Backbone (`vision`):
- `r18_paper` (CE-Loc gốc — nạp checkpoint của bài qua `load_celoc_paper`, hoặc train lại: GAMMA2 pha 1 / GAMMA4 / DELTA): ResNet18
  4 kênh như `refs/repos/Count-Editing/CE-LocModel/models/{vision_encoder,spatial_softmax}.py`, SpatialSoftmax 512 kênh trên C5,
  Linear(1024, 128); `ss_mask: true` = ô đệm −inf trước softmax (lưới toạ độ giữ hệ canvas của bài).
- `r50_fpn` ("CE-Loc gốc + R-50", GAMMA0 / 0.1, người dùng chốt 2026-10-01): R-50 BN train, kênh density = TB weight RGB như bài,
  SpatialSoftmax 2048 kênh trên C5 (layer4) -> Linear(4096, vis_dim). FPN vẫn dựng (không dùng) để giữ key checkpoint GAMMA0.

DELTA — bản CE-Loc cập nhật của tác giả gốc (`refs/CE-Loc-update/models/diffusion_module.py`):
- `refiner` (dict config, `enabled`): `ClipBoxRefiner` (`models/clip_refiner.py`) TRONG mạng khử nhiễu — mỗi lần gọi ε_θ:
  x_t -> refiner -> refined = x_t + Δ. `unet_input: concat` (DELTA1 / 2.x): U-Net nhận `[x_t ; refined]` (8 kênh, ra ε 4 kênh),
  loss = ε-MSE + `aux_loss_weight` · MSE(refined, x0). `unet_input: none` (DELTA3): KHÔNG U-Net / backbone / FiLM — refiner là cả mạng
  khử nhiễu, ε̂ suy từ refined (= x̂0) cho sampler, loss = `aux_loss_weight` · MSE(refined, x0). `unet_input: post` (DELTA3.1 / 3.2):
  U-Net TRƯỚC (nhận x_t, 4 kênh) -> ε̂₁ -> x̂0₁ (kẹp [−1, 1], detach) -> refiner nhìn x̂0₁, head đọc là sửa trên ε: ε̂ = ε̂₁ + Δε;
  loss = MSE(ε̂, ε) + `aux_loss_weight` · MSE(ε̂₁, ε); bỏ refiner (`_norefine`) = ε̂₁. Cần token chữ CLIP text
  (`text_tokens`, `models/text.TextTable`). CLIP ViT frozen nạp từ HF lúc dựng (không vào state_dict); buffer `clip_sha` = vân tay
  (vision, text) kiểm khớp khi nạp.
- `unet_input: mid` (DELTA4.1, mục 10.6): refiner CHÈN giữa U-Net (4 kênh), ở vị trí mid sau cross-attn box vật: h [d_mid] ->
  b = x_t + `mid_in`(h) (Linear d_mid -> 4, khởi tạo 0 ⇒ lúc đầu b = x_t) -> refined = refiner(b) -> h ← h + `mid_out`(refined) (Linear
  4 -> d_mid, khởi tạo 0 ⇒ lúc đầu ≡ U-Net không refiner); loss = ε-MSE + `aux_loss_weight` · MSE(refined, x0) + `mid_aux_weight` ·
  MSE(b, x0) (b = U-Net đoán box giữa chừng). Bỏ refiner (`_norefine`) = bỏ nhánh chèn (ngoài phân bố train).
- refiner + `obj_attn` (DELTA4 / 3.2 / 4.1): như GAMMA4; U-Net 8 kênh (`concat`) thì box vật mã hoá gương dưới dạng `[o ; o]` (refined
  của box sạch = chính nó); `mid`: box vật KHÔNG qua nhánh chèn.
- DELTA5.x (mục 11, bản log-scale của tác giả `refs/CE-Loc-logScale/`): `box_coder` (`utils/box_coder.py`: linear | log — w, h trên
  thang log) áp cho box đích, box vật, RoIAlign của refiner, giải mã khi eval; loss phụ thêm `aux_giou_weight` · (1 − GIoU) trên box
  đã giải mã; sampler `ddim` của tác giả (`ddim_sample`: x̂0 kẹp [−1, 1], `eta`, `ddim_steps` bước cách đều trên T).
- `use_condition: false` (DELTA1.1): điều kiện ảnh + text = 0 (embedding t giữ), refiner bỏ qua (refined = x_t), aux tắt — như tác giả.
  Ở đây refiner không được dựng và `vision` / `text_proj` đóng băng (không ai đọc) ⇒ mọi tham số train vẫn có grad (DDP không find_unused).

GAMMA4 (`obj_attn`): box nhiễu cross-attend tới box các vật đang có ở 6 vị trí U-Net (`models/unet1d.py`). GAMMA4.0.1 / 4.0.2
(`obj_k`, docs/EXPERIMENT_GAMMA.md mục 16.8): chỉ `obj_k` vật vào attention — `obj_select: nearest` = `obj_k` vật có tâm gần tâm x_t
nhất, chọn lại ở MỖI lần gọi U-Net (mỗi bước, mỗi mẫu; mọi vật mã hoá một lần, chọn bằng mask `nearest_mask`); `random` = `obj_k` vật
ngẫu nhiên chọn TRƯỚC khi mã hoá — train rút lại mỗi lần, eval một bộ mỗi ảnh (seed = crc32 box vật) dùng chung cho mọi mẫu / bước.
"""

import zlib
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import ce_localization.models.clip_refiner as clip_refiner
from ce_localization.models.backbone import ResNet50FPN
from ce_localization.models.geo import pad_objects
from ce_localization.models.memory import valid_cells_mask
from ce_localization.models.unet1d import ConditionalUnet1D
from ce_localization.utils.box_coder import (box_coder_config, encode_rel, generalized_box_iou_cxcywh, is_log, to_linear_unit,
                                             unit_fraction)
from ce_localization.utils.diffusion_math import (linear_alphas_cumprod, predict_noise_from_start, predict_start_from_noise,
                                                  q_sample)

__all__ = ["BoxPolicy", "PaperVisionEncoder", "boxes_to_unit", "unit_to_boxes", "norm_whwh", "ddpm_sample", "mock_sample",
           "ddim_sample", "paper_spatial_softmax", "nearest_mask", "VISIONS", "UNET_INPUTS", "OBJ_SELECTS", "SAMPLERS"]

VISIONS = ("r50_fpn", "r18_paper")
UNET_INPUTS = ("concat", "none", "post", "mid")   # DELTA: U-Net nhận [x_t ; refined] | không U-Net | refiner sửa ε sau U-Net | chèn giữa
OBJ_SELECTS = ("nearest", "random")          # GAMMA4.0.x: cách chọn `obj_k` vật vào cross-attn
SAMPLERS = ("ddpm", "mock", "ddim")           # DDPM T bước | vòng mock của bài | DDIM / DDPM có kẹp của bản log-scale (DELTA5)

C5_STRIDE = 32
C5_CHANNELS = 2048


def norm_whwh(model, whwh, canvas):
    """whwh vùng thật [B,4] -> whwh chuẩn hoá box của `model` (`box_norm`): canvas T (CE-Loc: BoxPolicy / BoxRefiner) hoặc
    giữ vùng thật (refine kiểu DiffusionDet của ProposeRefine)."""
    return torch.full_like(whwh, float(canvas)) if getattr(model, "box_norm", "valid") == "canvas" else whwh


def boxes_to_unit(boxes_xyxy, whwh, coder=None):
    """xyxy pixel canvas [...,4] + whwh [...,4] (nw, nh, nw, nh) -> cxcywh/(nw,nh) ·2 − 1 (`coder` log — DELTA5: w, h trên thang log,
    `utils/box_coder.py`)."""
    b = boxes_xyxy / whwh
    c = torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                     b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], dim=-1)
    return encode_rel(c, coder)


def unit_to_boxes(u, whwh, coder=None):
    """Ngược `boxes_to_unit`: [...,4] mã hoá -> xyxy pixel canvas. KHÔNG kẹp (linear: w, h âm giữ nguyên để
    đo tỉ lệ box suy biến; IoU tự kẹp w, h về 0. log: w, h luôn > 0)."""
    c = (to_linear_unit(u, coder) + 1) / 2
    xyxy = torch.stack([c[..., 0] - c[..., 2] / 2, c[..., 1] - c[..., 3] / 2,
                        c[..., 0] + c[..., 2] / 2, c[..., 1] + c[..., 3] / 2], dim=-1)
    return xyxy * whwh


def nearest_mask(x, units, mask, k):
    """x [R,4] box nhiễu, units [R,M,4] box vật (cùng hệ cxcywh·2 − 1), mask [R,M] -> mask [R,M] chỉ giữ k vật có tâm gần tâm x nhất
    (khoảng cách Euclid, như `relation.gather` của GAMMA3.1; không chia (w, h) của x — ở t lớn vô nghĩa). < k vật: giữ hết."""
    d = ((units[..., :2].float() - x[:, None, :2].float()) ** 2).sum(-1).masked_fill(~mask, float("inf"))
    idx = d.topk(min(k, d.shape[1]), dim=1, largest=False).indices
    return torch.zeros_like(mask).scatter(1, idx, True) & mask


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
def ddim_sample(eps_fn, n_rows, alphas_cumprod, ddim_steps=None, eta=1.0, clip_sample=True, generator=None, device="cpu",
                record=None):
    """`ObjectPlacementPolicy.sample` của bản log-scale tác giả (`refs/CE-Loc-logScale/models/diffusion_module.py`, viết lại đúng phép
    tính — có test so bản sao nguyên văn): `ddim_steps` bước cách đều T−1 -> 0 (None = đủ T bước), mỗi bước x̂0 = (x − √(1−ᾱ)ε̂)/√ᾱ, kẹp
    [−1, 1] (`clip_sample`) rồi suy lại ε̂; σ = eta·√((1−ᾱ_prev)/(1−ᾱ)·(1−ᾱ/ᾱ_prev)); x ← √ᾱ_prev·x̂0 + √(1−ᾱ_prev−σ²)·ε̂ (+ σ·z trừ
    bước cuối). eta 1 ≈ DDPM, 0 = DDIM tất định. -> x0 [R,4]; `record` -> (x0, quỹ đạo)."""
    ab = alphas_cumprod.to(device)
    T = ab.shape[0]
    timesteps = torch.linspace(T - 1, 0, ddim_steps or T, device=device).round().long().unique_consecutive()
    traj = [] if record is not None else None
    x = torch.randn((n_rows, 4), device=device, generator=generator)
    for i, t in enumerate(timesteps):
        tb = t.expand(n_rows)
        eps = eps_fn(x, tb)
        ab_t = ab[t]
        _snap(traj, record, int(t), x, eps, ab_t)
        ab_prev = ab[timesteps[i + 1]] if i + 1 < len(timesteps) else x.new_tensor(1.0)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_sample:
            x0 = x0.clamp(-1, 1)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(timesteps):
            x = x + sigma * torch.randn(x.shape, device=device, generator=generator)
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


def paper_spatial_softmax(feat, valid=None, return_att=False):
    """`SpatialSoftmax` của CE-Loc gốc, y từng phép tính: meshgrid 'ij' -> toạ độ đầu là DỌC; xen kẽ (x_c, y_c).
    `valid` [N,H,W] bool (GAMMA2): ô False (phần đệm) nhận −inf trước softmax — cùng lưới toạ độ canvas của bài.
    -> [N, 2C] (+ attention softmax [N, C, H·W] nếu `return_att`)."""
    N, C, H, W = feat.shape
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, H, device=feat.device),
                                  torch.linspace(-1, 1, W, device=feat.device), indexing="ij")
    logits = feat.reshape(N, C, -1)
    if valid is not None:
        logits = logits.masked_fill(~valid.reshape(N, 1, -1), float("-inf"))
    att = F.softmax(logits, dim=-1)
    ex = torch.sum(pos_x.reshape(H * W) * att, dim=-1, keepdim=True)
    ey = torch.sum(pos_y.reshape(H * W) * att, dim=-1, keepdim=True)
    kp = torch.cat([ex, ey], dim=-1).reshape(N, -1)
    return (kp, att) if return_att else kp


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
        return valid_cells_mask(valid_hw, c5.shape[2], c5.shape[3], C5_STRIDE) if self.ss_mask else None

    def keypoints_flat(self, c5, valid_hw=None):
        """C5 -> [N, 2C] toạ độ SpatialSoftmax (có mask nếu `ss_mask`)."""
        return paper_spatial_softmax(c5, self.valid(c5, valid_hw))

    def forward(self, x, valid_hw=None):
        return self.projection(self.keypoints_flat(self.backbone(x), valid_hw))

    @torch.no_grad()
    def keypoints(self, x, valid_hw=None):
        """Soi SpatialSoftmax: -> (toạ độ [B,C,2] theo thứ tự (DỌC, NGANG) trong [−1, 1] — meshgrid 'ij' của bài —,
        attention softmax [B,C,H,W]). Toạ độ −1 / +1 = TÂM ô đầu / ô cuối của lưới H×W."""
        feat = self.backbone(x)
        N, C, H, W = feat.shape
        kp, att = paper_spatial_softmax(feat, self.valid(feat, valid_hw), return_att=True)
        return kp.reshape(N, C, 2), att.reshape(N, C, H, W)


class BoxPolicy(nn.Module):
    def __init__(self, in_channels=3, pretrained_backbone=True, fpn_dim=256, vis_dim=128, text_in=512,
                 text_dim=128, step_embed_dim=256, down_dims=(64, 128, 256), kernel_size=3, n_groups=8,
                 num_timesteps=1000, beta_start=1e-4, beta_end=0.02, vision="r50_fpn", ss_mask=False,
                 obj_attn=False, obj_heads=4, obj_max=300, obj_k=None, obj_select="nearest", refiner=None, use_condition=True,
                 box_coder=None):
        super().__init__()
        rcfg = clip_refiner.refiner_config(refiner)
        self.use_refiner, self.use_condition = rcfg is not None, bool(use_condition)
        self.refiner_aux_weight = float(rcfg["aux_loss_weight"]) if rcfg else 0.0
        self.mid_aux_weight = float(rcfg["mid_aux_weight"]) if rcfg else 0.0
        self.aux_giou_weight = float(rcfg["aux_giou_weight"]) if rcfg else 0.0
        self.box_coder = box_coder_config(box_coder)            # DELTA5: log; mọi config cũ: linear (trùng từng bit)
        unet_input = rcfg["unet_input"] if rcfg else None
        self.box_norm = "canvas"
        if rcfg:
            if unet_input not in UNET_INPUTS:
                raise ValueError(f"refiner.unet_input {unet_input!r} không thuộc {UNET_INPUTS}")
            if obj_attn and unet_input == "none":
                raise ValueError("refiner.unet_input none (không có U-Net) không đi với obj_attn")
            if rcfg["use_density"] and in_channels != 4:
                raise ValueError("refiner.use_density cần ảnh 4 kênh (RGB + density)")
            if unet_input == "none" and not self.use_condition:
                raise ValueError("refiner.unet_input none (refiner là cả mạng khử nhiễu) cần use_condition: true")
            if unet_input == "post" and not self.use_condition:
                raise ValueError("refiner.unet_input post cần use_condition: true (không thì không có refiner)")
            if unet_input == "post" and int(rcfg["num_stages"]) != 1:
                raise ValueError("refiner.unet_input post cần num_stages 1 (stage sau RoIAlign quanh box lẫn đơn vị ε)")
            if unet_input == "mid" and not self.use_condition:
                raise ValueError("refiner.unet_input mid cần use_condition: true (không thì không có refiner, mid_in / mid_out không grad)")
            if self.mid_aux_weight and unet_input != "mid":
                raise ValueError("refiner.mid_aux_weight chỉ dùng với unet_input mid")
            if self.aux_giou_weight and unet_input == "post":
                raise ValueError("refiner.aux_giou_weight cần box refined (concat / none / mid) — post: loss phụ là ε-MSE của U-Net")
        self.unet_input = unet_input
        self.unet_in_dim = 8 if unet_input == "concat" else 4     # DELTA1 / 2.x / 4: U-Net nhận [x_t ; refined]
        self.obj_attn, self.obj_max = bool(obj_attn), obj_max
        if obj_k is not None and (not obj_attn or int(obj_k) < 1):
            raise ValueError(f"obj_k {obj_k!r} cần obj_attn: true và >= 1")
        if obj_select not in OBJ_SELECTS:
            raise ValueError(f"obj_select {obj_select!r} không thuộc {OBJ_SELECTS}")
        # thuộc tính thường (không vào state_dict) ⇒ eval.py --obj-k / --obj-select đặt lại được trên checkpoint GAMMA4
        self.obj_k, self.obj_select = (None if obj_k is None else int(obj_k)), obj_select
        self.num_timesteps = num_timesteps
        self.register_buffer("alphas_cumprod", linear_alphas_cumprod(num_timesteps, beta_start, beta_end),
                             persistent=False)
        self.vision_kind, self.noise_net = None, None
        if unet_input == "none":                                     # DELTA3: không backbone / FiLM / U-Net
            self.cond_dim = 0
        else:
            if vision not in VISIONS:
                raise ValueError(f"vision {vision!r} không thuộc {VISIONS}")
            self.vision_kind = vision
            if vision == "r18_paper":
                self.vision = PaperVisionEncoder(vis_dim, in_channels, pretrained_backbone, ss_mask)
            else:
                self.backbone = ResNet50FPN(fpn_dim, pretrained=pretrained_backbone, in_channels=in_channels, norm="bn",
                                            density_init="rgb_mean")
                self.vis_proj = nn.Linear(2 * C5_CHANNELS, vis_dim)
            self.text_proj = nn.Sequential(nn.Linear(text_in, text_dim), nn.Mish())
            self.cond_dim = vis_dim + text_dim
            # DELTA1 (`ObjectPlacementPolicy` của tác giả, concat): U-Net nhận [x_t ; refined] (8 kênh), vẫn ra ε 4 kênh
            self.noise_net = ConditionalUnet1D(self.unet_in_dim, self.cond_dim, step_embed_dim,
                                               tuple(down_dims), kernel_size, n_groups, obj_attn=obj_attn,
                                               obj_heads=obj_heads, output_dim=4)
        # dựng SAU noise_net: khởi tạo ResNet / U-Net trùng bản không refiner (DELTA1 vs DELTA1.1 so cặp)
        self.refiner = None
        if rcfg and self.use_condition:
            vision_model, text_hidden = clip_refiner.load_clip(rcfg["clip_model_name"], clip_refiner.text_model_name(rcfg))
            self.refiner = clip_refiner.ClipBoxRefiner(rcfg, vision_model, text_hidden)
            sha = torch.zeros(2, 32, dtype=torch.uint8)
            sha[0] = torch.tensor(list(clip_refiner.weights_sha256(vision_model)), dtype=torch.uint8)
            self.register_buffer("clip_sha", sha)                    # [vân tay CLIP vision ; vân tay CLIP text] (text đặt sau)
            self.refiner.box_coder = self.box_coder if is_log(self.box_coder) else None   # RoIAlign giải mã box log (DELTA5)
        if unet_input == "mid":                                      # DELTA4.1: h -> box b -> refiner -> h (cả hai khởi tạo 0)
            self.mid_in, self.mid_out = nn.Linear(self.noise_net.mid_dim, 4), nn.Linear(4, self.noise_net.mid_dim)
            for lin in (self.mid_in, self.mid_out):
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
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
        encoder (như `encode_condition` của tác giả); DELTA3 (không U-Net) -> [B, 0] (sampler chỉ đọc B / thiết bị)."""
        if not self.use_condition or self.noise_net is None:
            return images.new_zeros(images.shape[0], self.cond_dim, dtype=torch.float32)
        if self.vision_kind == "r18_paper":                       # bài: không mask vùng thật; GAMMA2: ss_mask
            return torch.cat([self.vision(images, valid_hw), self.text_emb(text_raw, null_text)], dim=-1)
        kp = paper_spatial_softmax(self.backbone.forward_c5(images))
        return torch.cat([self.vis_proj(kp), self.text_emb(text_raw, null_text)], dim=-1)

    def cond_from_keypoints(self, kp, text_raw):
        """ResNet18 của bài: toạ độ SpatialSoftmax [B, 2C] -> cond — để GAMMA2 dùng chung backbone với model refine."""
        return torch.cat([self.vision.projection(kp), self.text_emb(text_raw)], dim=-1)

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
            raise RuntimeError("CLIP text (token chữ của refiner) KHÁC bản lúc train checkpoint — kiểm refiner.text_model_name / "
                               "HF cache")

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
        """x [R,4] box nhiễu, t [R], cond [R, D], ctx (`refiner_context`, C hàng) + ctx_index [R] | None, obj (token vật, mask, box vật)
        của `object_tokens` | None -> (ε̂ [R,4], refined [R,4] | None) — `predict_noise` của tác giả. Không refiner: U-Net trên x như
        bài. DELTA3 (không U-Net): refined = x̂0 ⇒ ε̂ = (x_t − √ᾱ_t · refined) / √(1 − ᾱ_t) (fp32). DELTA3.x (`post`): refined None.
        GAMMA4.0.1 (`obj_select: nearest`): mask chỉ giữ `obj_k` vật gần x (x_t noisy khi train — không theo đích)."""
        return self._denoise(x, t, cond, ctx, ctx_index, obj)[:2]

    def _denoise(self, x, t, cond, ctx=None, ctx_index=None, obj=None):
        """Lõi của `predict_noise` -> (ε̂, refined | None, phụ | None): phụ = ε̂₁ của U-Net trước refiner (`post`) | box b giữa U-Net
        mà refiner nhận (`mid`)."""
        if obj is not None:
            tok, mask, units = obj
            if self.obj_k is not None and self.obj_select == "nearest":
                mask = nearest_mask(x, units, mask, self.obj_k)
            obj = (tok, mask)
        if self.unet_input == "post":                          # DELTA3.x: U-Net trước, refiner nhìn x̂0 của U-Net rồi sửa ε
            eps1 = self.noise_net(x.unsqueeze(1), t, cond, *(obj or (None, None))).squeeze(1)
            if ctx is None:                                    # bỏ refiner (`_norefine`): U-Net trần, đúng phân bố train
                return eps1, None, eps1
            with torch.autocast(device_type=x.device.type, enabled=False):
                x0 = predict_start_from_noise(x.float(), t, eps1.float(), self.alphas_cumprod).clamp(-1, 1).detach()
                delta = self.refiner(x0, t, ctx, ctx_index) - x0          # head của tác giả, đọc là sửa trên ε (khởi tạo 0)
                eps = eps1.float() + delta
            if self.track_refine:
                self._track_delta(t, delta.abs().mean(-1))
            return eps, None, eps1
        if self.unet_input == "mid":                           # DELTA4.1: refiner chèn ở vị trí mid của U-Net
            mid = {}

            def inject(h):
                with torch.autocast(device_type=h.device.type, enabled=False):
                    b = x.float() + self.mid_in(h[..., 0].float())
                    mid["b"], mid["refined"] = b, self.refiner(b, t, ctx, ctx_index)
                    return h + self.mid_out(mid["refined"])[..., None].to(h.dtype)
            eps = self.noise_net(x.unsqueeze(1), t, cond, *(obj or (None, None)),
                                 inject=None if ctx is None else inject).squeeze(1)
            if ctx is not None and self.track_refine:
                self._track_delta(t, (mid["refined"] - mid["b"]).abs().mean(-1))
            return eps, mid.get("refined"), mid.get("b")
        unet_in, refined = x, None
        if self.use_refiner:
            refined = self.refiner(x, t, ctx, ctx_index) if ctx is not None else x
            unet_in = torch.cat([x, refined.to(x.dtype)], dim=-1)
        if self.noise_net is not None:
            eps = self.noise_net(unet_in.unsqueeze(1), t, cond, *(obj or (None, None))).squeeze(1)
        elif ctx is None:
            raise ValueError("refiner.unet_input none: refiner là cả mạng khử nhiễu, không chạy được khi bỏ refiner")
        else:
            eps = predict_noise_from_start(x.float(), t, refined, self.alphas_cumprod)
        if refined is not None and ctx is not None and self.track_refine:
            self._track_delta(t, (refined - x).abs().mean(-1))
        return eps, refined, None

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
        """-> thống kê loss lần forward train gần nhất (loss_eps, loss_aux, delta_abs, aux_ratio — DELTA; + loss_eps1, eps_ratio — post;
        + loss_mid, mid_ratio — mid) rồi xoá."""
        st, self._stats = self._stats, {}
        return st

    def object_tokens(self, objects, canvas, cond, generator=None, cap=False):
        """objects: list B tensor [Mᵢ,4] xyxy pixel canvas -> (token 6 tầng, mask, box vật [B,M,4] chuẩn hoá) cho `noise_net` (mã hoá
        gương, t = 0). Box vật chuẩn hoá GIỐNG đích (chia canvas). `cap` (train): > obj_max vật thì lấy ngẫu nhiên obj_max.
        GAMMA4.0.2 (`obj_k` + `random`): giữ `obj_k` vật ngẫu nhiên — train (`cap`) rút theo `generator`, eval seed = crc32 box vật
        (cố định theo mẫu, không phụ thuộc batch, không tiêu RNG eval). `nearest`: giữ hết, chọn ở `predict_noise`."""
        if objects is None:
            raise ValueError("model.obj_attn cần `objects` (box các vật đang có) khi train / suy luận")
        dev = cond.device
        whwh = torch.full((4,), float(canvas))
        units = []

        def draw():
            return int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator, device=generator.device)) if generator \
                is not None else 0
        for o in objects:
            o = torch.as_tensor(o, dtype=torch.float32).reshape(-1, 4).cpu()
            if self.obj_k is not None and self.obj_select == "random" and len(o) > self.obj_k:
                seed = draw() if cap else zlib.crc32(o.contiguous().numpy().tobytes())
                o = o[torch.randperm(len(o), generator=torch.Generator().manual_seed(seed))[: self.obj_k]]
            elif cap and len(o) > self.obj_max:
                o = o[torch.randperm(len(o), generator=torch.Generator().manual_seed(draw()))[: self.obj_max]]
            units.append(boxes_to_unit(o, whwh, self.box_coder))
        objs, mask = pad_objects(units, dev, min_m=1)
        enc = torch.cat([objs, objs], -1) if self.unet_in_dim == 8 else objs      # DELTA4: U-Net 8 kênh nhận [o ; o]
        return self.noise_net.encode_objects(enc, mask, cond), mask, objs

    def forward(self, images, text_raw, valid_hw, x0, k=1, generator=None, objects=None, text_tokens=None):
        """ε-MSE như `ObjectPlacementPolicy.compute_loss` của bài, k bộ (t, ε) mỗi ảnh.
        x0 [B,4] trong [−1,1]. `objects` (GAMMA4): list B box vật xyxy pixel canvas. `text_tokens` (DELTA): (token, mask) của
        `TextTable.tokens`. -> loss vô hướng (trung bình trên B·k·4; DELTA + aux)."""
        cond = self.condition(images, text_raw, valid_hw)
        obj = (self.object_tokens(objects, images.shape[-1], cond, generator, cap=True) if self.obj_attn
               else None)
        return self.eps_loss(cond, x0, k, generator, obj, self.refiner_context(images, text_tokens))

    @staticmethod
    def _repeat_obj(obj, n):
        return None if obj is None else ([t.repeat_interleave(n, 0) for t in obj[0]], obj[1].repeat_interleave(n, 0),
                                         obj[2].repeat_interleave(n, 0))

    def eps_loss(self, cond, x0, k=1, generator=None, obj=None, ctx=None):
        """ε-MSE từ cond [B, D] đã tính sẵn; obj = (token, mask, box vật) của `object_tokens` | None; ctx (DELTA) = `refiner_context` (B hàng)
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
        noisy = q_sample(x0, t, noise, self.alphas_cumprod)
        pred, refined, extra = self._denoise(noisy, t, cond, ctx, None, obj)
        if self.unet_input == "post":                          # DELTA3.x: ε cuối + ε của U-Net trước refiner
            eps1 = extra
            with torch.autocast(device_type=dev.type, enabled=False):
                loss, loss1 = F.mse_loss(pred.float(), noise.float()), F.mse_loss(eps1.float(), noise.float())
                self._stats = {"loss_eps": loss.detach(), "loss_eps1": loss1.detach(),
                               "delta_abs": (pred.float() - eps1.float()).abs().mean().detach(),
                               "eps_ratio": (loss / loss1.clamp_min(1e-12)).detach()}
            return loss + self.refiner_aux_weight * loss1
        loss = F.mse_loss(pred, noise) if self.noise_net is not None else None      # DELTA3: không có số hạng ε
        if refined is not None and ctx is not None:
            src = extra if self.unet_input == "mid" else noisy                  # box refiner nhận: b (DELTA4.1) | x_t
            with torch.autocast(device_type=dev.type, enabled=False):          # refined fp32 (refiner tắt autocast)
                aux = F.mse_loss(refined, x0.to(refined.dtype))
                giou = None
                if self.aux_giou_weight > 0:                                   # DELTA5.x: GIoU trên box giải mã (tỉ lệ canvas)
                    giou = (1 - generalized_box_iou_cxcywh(unit_fraction(refined, self.box_coder),
                                                           unit_fraction(x0.to(refined.dtype), self.box_coder))).mean()
                st = {} if loss is None else {"loss_eps": loss.detach().float()}
                self._stats = {**st, "loss_aux": aux.detach().float(),
                               "delta_abs": (refined - src.to(refined.dtype)).abs().mean().detach().float(),
                               "aux_ratio": (aux / F.mse_loss(noisy.to(refined.dtype), x0.to(refined.dtype)).clamp_min(1e-12))
                               .detach().float()}
                if giou is not None:
                    self._stats["loss_giou"] = giou.detach()
                if self.unet_input == "mid":                                   # MSE(b, x0): b = U-Net đoán box giữa chừng
                    lmid = F.mse_loss(extra, x0.float())
                    self._stats.update({"loss_mid": lmid.detach(), "mid_ratio": (aux / lmid.clamp_min(1e-12)).detach()})
            if giou is not None:                                               # aux = MSE + w_giou · (1 − GIoU) (tác giả)
                aux = aux + self.aux_giou_weight * giou
            if loss is None:
                loss = self.refiner_aux_weight * aux
            elif self.refiner_aux_weight > 0:
                loss = loss + self.refiner_aux_weight * aux
            if self.unet_input == "mid" and self.mid_aux_weight > 0:
                loss = loss + self.mid_aux_weight * lmid
        return loss

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, n_samples, generator=None, sampler="ddpm", record=None,
               null_text=False, objects=None, use_objects=True, text_tokens=None, use_refiner=True, ddim_steps=None, eta=1.0,
               clip_sample=True):
        """n_samples mẫu ĐỘC LẬP mỗi ảnh (ảnh mã hoá một lần, các mẫu khử nhiễu song song).
        -> [B, n_samples, 4] trong [−1,1]; `record` (tập t / "all") -> (box, quỹ đạo: list {t, x_t, x0_hat} [B,n,4]).
        GAMMA4: `objects` = box vật (token tính MỘT lần mỗi ảnh, dùng cho mọi bước; 4.0.1 chọn `obj_k` gần nhất mỗi bước); `use_objects=False`
        = tắt (`_noobj`).
        DELTA: `text_tokens` (token chữ CLIP); `use_refiner=False` = bỏ refiner (refined = x_t — `_norefine`, ngoài phân bố train).
        `sampler` ∈ SAMPLERS; `ddim` (DELTA5): `ddim_steps` / `eta` / `clip_sample` của `ddim_sample`."""
        cond = self.condition(images, text_raw, valid_hw, null_text)
        obj = (self.object_tokens(objects, images.shape[-1], cond) if self.obj_attn and use_objects else None)
        ctx = self.refiner_context(images, text_tokens) if use_refiner else None
        return self.sample_from_cond(cond, n_samples, generator, sampler, record, obj=obj, ctx=ctx, ddim_steps=ddim_steps, eta=eta,
                                     clip_sample=clip_sample)

    @torch.no_grad()
    def sample_from_cond(self, cond, n_samples, generator=None, sampler="ddpm", record=None, obj=None, ctx=None, ddim_steps=None,
                         eta=1.0, clip_sample=True):
        """Như `sample` nhưng từ cond [B, D] đã tính sẵn (+ token vật của `object_tokens`, ngữ cảnh refiner `refiner_context`).
        Mọi hàng cùng t ở mỗi bước (mock / DDPM) ⇒ refiner tính luồng ngữ cảnh một lần mỗi ảnh (`ctx_index`).
        DELTA + `record`: mỗi bước quỹ đạo thêm `refined` [B,n,4] (box refiner đề xuất ở bước đó; `x0_hat` = x̂0 suy từ ε̂ của U-Net)."""
        B = cond.shape[0]
        cond = cond.repeat_interleave(n_samples, dim=0)
        obj = self._repeat_obj(obj, n_samples)
        ci = None if ctx is None else torch.arange(B, device=cond.device).repeat_interleave(n_samples)
        refs = {} if record is not None and ctx is not None else None              # t -> refined (chỉ ghi, không đổi phép tính)

        def fn(x, t):
            eps, refined = self.predict_noise(x, t, cond, ctx, ci, obj)
            if refs is not None and refined is not None:
                refs[int(t[0])] = refined.detach().float().cpu()
            return eps
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler {sampler!r} không thuộc {SAMPLERS}")
        run = {"ddpm": ddpm_sample, "mock": mock_sample,
               "ddim": partial(ddim_sample, ddim_steps=ddim_steps, eta=eta, clip_sample=clip_sample)}[sampler]
        out = run(fn, cond.shape[0], self.alphas_cumprod, generator=generator, device=cond.device, record=record)
        if record is None:
            return out.view(B, n_samples, 4)
        x, traj = out
        return x.view(B, n_samples, 4), [{**s, "x_t": s["x_t"].view(B, n_samples, 4), "x0_hat": s["x0_hat"].view(B, n_samples, 4),
                                          **({"refined": refs[s["t"]].view(B, n_samples, 4)} if refs else {})}
                                         for s in traj]

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
