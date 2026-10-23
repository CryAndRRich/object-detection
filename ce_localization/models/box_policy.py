"""BoxPolicy — CE-Loc từ DELTA4 trở đi (docs/EXPERIMENT_DELTA.md): sinh MỘT box bằng khuếch tán, điều kiện ảnh + text + box vật.

    ảnh [B,4,T,T] (RGB + density, kiểu bài: /255 và `.convert("L")`) -> ResNet18 của bài -> SpatialSoftmax C5 (ô đệm −inf)
        -> Linear(1024, vis_dim)                                                             ┐ cond (FiLM của U-Net)
    text: CLIP pooler 512 (`models/text.TextTable`) -> Linear -> Mish                         ┘
    box vật đang có -> token gương qua chính U-Net (t = 0) -> cross-attn ở 6 vị trí U-Net (GAMMA4)
    x_t -> BoxRefiner của tác giả (`models/clip_refiner.py`) -> refined ; U-Net 1D nhận [x_t ; refined] -> ε̂

Box trong không gian khuếch tán: cxcywh chia CANVAS, mã hoá `model.box_coder` (linear: ·2 − 1; log — DELTA5: w, h thang log).
Loss = MSE(ε̂, ε) + `aux_loss_weight` · [MSE(refined, x0) + `aux_giou_weight` · (1 − GIoU(refined, x0))] (công thức tác giả).
Sampler: `mock` (vòng `x -= ε/100` của CE-Loc gốc) | `ddim` (sampler bản log-scale tác giả: x̂0 kẹp [−1, 1], `eta`, `ddim_steps`).
CLIP ViT frozen nạp từ HF lúc dựng (không vào state_dict); buffer `clip_sha` = vân tay (vision, text) kiểm khớp khi nạp.
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import ce_localization.models.clip_refiner as clip_refiner
from ce_localization.models.unet1d import ConditionalUnet1D
from ce_localization.utils.box_coder import (box_coder_config, encode_rel, generalized_box_iou_cxcywh, is_log, to_linear_unit,
                                             unit_fraction)
from ce_localization.utils.diffusion_math import linear_alphas_cumprod, q_sample

__all__ = ["BoxPolicy", "PaperVisionEncoder", "boxes_to_unit", "unit_to_boxes", "mock_sample", "ddim_sample", "paper_spatial_softmax",
           "valid_cells_mask", "pad_objects", "build_model", "load_checkpoint", "SAMPLERS"]

SAMPLERS = ("mock", "ddim")
C5_STRIDE = 32


def boxes_to_unit(boxes_xyxy, whwh, coder=None):
    """xyxy pixel canvas [...,4] + whwh [...,4] -> cxcywh/whwh mã hoá theo `coder` (None / linear: ·2 − 1)."""
    b = boxes_xyxy / whwh
    c = torch.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2,
                     b[..., 2] - b[..., 0], b[..., 3] - b[..., 1]], dim=-1)
    return encode_rel(c, coder)


def unit_to_boxes(u, whwh, coder=None):
    """Ngược `boxes_to_unit` -> xyxy pixel canvas. KHÔNG kẹp (linear: w, h âm giữ nguyên để đo box suy biến)."""
    c = (to_linear_unit(u, coder) + 1) / 2
    xyxy = torch.stack([c[..., 0] - c[..., 2] / 2, c[..., 1] - c[..., 3] / 2,
                        c[..., 0] + c[..., 2] / 2, c[..., 1] + c[..., 3] / 2], dim=-1)
    return xyxy * whwh


@torch.no_grad()
def ddim_sample(eps_fn, n_rows, alphas_cumprod, ddim_steps=None, eta=1.0, clip_sample=True, generator=None, device="cpu"):
    """`ObjectPlacementPolicy.sample` của bản log-scale tác giả (có test so bản sao nguyên văn): `ddim_steps` bước cách đều T−1 -> 0
    (None = đủ T), x̂0 = (x − √(1−ᾱ)ε̂)/√ᾱ kẹp [−1, 1] (`clip_sample`) rồi suy lại ε̂; σ = eta·√((1−ᾱ_prev)/(1−ᾱ)·(1−ᾱ/ᾱ_prev));
    x ← √ᾱ_prev·x̂0 + √(1−ᾱ_prev−σ²)·ε̂ (+ σ·z trừ bước cuối). eps_fn(x [R,4], t [R] long) -> ε̂ [R,4]."""
    ab = alphas_cumprod.to(device)
    T = ab.shape[0]
    # t dạng int (một lần đọc về CPU): chỉ số tensor 0-d trên GPU ép đồng bộ ở MỖI bước
    timesteps = torch.linspace(T - 1, 0, ddim_steps or T, device=device).round().long().unique_consecutive().tolist()
    x = torch.randn((n_rows, 4), device=device, generator=generator)
    for i, t in enumerate(timesteps):
        tb = torch.full((n_rows,), t, device=device, dtype=torch.long)
        eps = eps_fn(x, tb)
        ab_t = ab[t]
        ab_prev = ab[timesteps[i + 1]] if i + 1 < len(timesteps) else x.new_tensor(1.0)
        x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_sample:
            x0 = x0.clamp(-1, 1)
            eps = (x - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
        x = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(timesteps):
            x = x + sigma * torch.randn(x.shape, device=device, generator=generator)
    return x


@torch.no_grad()
def mock_sample(eps_fn, n_rows, alphas_cumprod, steps=100, generator=None, device="cpu"):
    """Vòng lấy mẫu CE-Loc gốc DÙNG THẬT (`inference.py`): t = steps−1..0, `x -= eps/steps` — không phải DDPM."""
    if steps > alphas_cumprod.shape[0]:
        raise ValueError(f"mock {steps} bước nhưng lịch train chỉ có T = {alphas_cumprod.shape[0]}")
    x = torch.randn((n_rows, 4), device=device, generator=generator)
    for t in reversed(range(steps)):
        x = x - eps_fn(x, torch.full((n_rows,), t, device=device, dtype=torch.long)) / steps
    return x


def valid_cells_mask(valid_hw, H, W, stride):
    """[B,2] (nh, nw) pixel -> [B,H,W] bool: ô thuộc vùng ảnh thật (mép trên / trái của ô nằm trong vùng thật)."""
    rows = torch.arange(H, device=valid_hw.device)[None, :] * stride < valid_hw[:, :1]   # [B,H]
    cols = torch.arange(W, device=valid_hw.device)[None, :] * stride < valid_hw[:, 1:2]  # [B,W]
    return rows[:, :, None] & cols[:, None, :]


def paper_spatial_softmax(feat, valid=None):
    """`SpatialSoftmax` của CE-Loc gốc (meshgrid 'ij', xen kẽ toạ độ); `valid` [N,H,W]: ô False nhận −inf trước softmax -> [N, 2C]."""
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


class PaperVisionEncoder(nn.Module):
    """`SpatialVisualEncoder` của CE-Loc gốc: ResNet18 (BN train) bỏ avgpool + fc -> SpatialSoftmax -> Linear(1024, D). Kênh density
    của conv1 khởi tạo = TB weight RGB như bài. `ss_mask`: ô đệm −inf trước softmax."""

    def __init__(self, output_dim=128, in_channels=4, pretrained=False, ss_mask=True):
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

    def forward(self, x, valid_hw=None):
        c5 = self.backbone(x)
        valid = valid_cells_mask(valid_hw, c5.shape[2], c5.shape[3], C5_STRIDE) if self.ss_mask else None
        return self.projection(paper_spatial_softmax(c5, valid))


def pad_objects(objects, dev, min_m=1):
    """list B tensor [Mᵢ,C] -> (objs [B,M,C] float32, mask [B,M] bool), M = max(max Mᵢ, min_m)."""
    M = max([len(o) for o in objects] + [min_m])
    C = objects[0].shape[-1] if objects else 4
    objs = torch.zeros(len(objects), M, C)
    mask = torch.zeros(len(objects), M, dtype=torch.bool)
    for i, o in enumerate(objects):
        o = torch.as_tensor(o, dtype=torch.float32).reshape(-1, C)
        objs[i, :len(o)] = o
        mask[i, :len(o)] = True
    return objs.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)


class BoxPolicy(nn.Module):
    def __init__(self, in_channels=4, pretrained_backbone=True, vis_dim=128, text_in=512, text_dim=128, step_embed_dim=256,
                 down_dims=(64, 128, 256), kernel_size=3, n_groups=8, num_timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 ss_mask=True, obj_heads=4, obj_max=300, refiner=None, box_coder=None):
        super().__init__()
        rcfg = clip_refiner.refiner_config(refiner)
        if rcfg is None or rcfg["unet_input"] != "concat":
            raise ValueError("BoxPolicy cần model.refiner bật với unet_input concat (DELTA4+)")
        if rcfg["use_density"] and in_channels != 4:
            raise ValueError("refiner.use_density cần ảnh 4 kênh (RGB + density)")
        self.refiner_aux_weight = float(rcfg["aux_loss_weight"])
        self.aux_giou_weight = float(rcfg["aux_giou_weight"])
        self.box_coder = box_coder_config(box_coder)
        self.obj_max = obj_max
        self.num_timesteps = num_timesteps
        self.register_buffer("alphas_cumprod", linear_alphas_cumprod(num_timesteps, beta_start, beta_end), persistent=False)
        self.vision = PaperVisionEncoder(vis_dim, in_channels, pretrained_backbone, ss_mask)
        self.text_proj = nn.Sequential(nn.Linear(text_in, text_dim), nn.Mish())
        self.cond_dim = vis_dim + text_dim
        self.noise_net = ConditionalUnet1D(8, self.cond_dim, step_embed_dim, tuple(down_dims), kernel_size, n_groups,
                                           obj_heads=obj_heads, output_dim=4)
        # dựng SAU noise_net (thứ tự khởi tạo ngẫu nhiên như mọi checkpoint đã train)
        vision_model, text_hidden = clip_refiner.load_clip(rcfg["clip_model_name"], clip_refiner.text_model_name(rcfg))
        self.refiner = clip_refiner.ClipBoxRefiner(rcfg, vision_model, text_hidden)
        sha = torch.zeros(2, 32, dtype=torch.uint8)
        sha[0] = torch.tensor(list(clip_refiner.weights_sha256(vision_model)), dtype=torch.uint8)
        self.register_buffer("clip_sha", sha)                        # [vân tay CLIP vision ; vân tay CLIP text] (text đặt sau)
        self.refiner.box_coder = self.box_coder if is_log(self.box_coder) else None   # RoIAlign giải mã box log (DELTA5)
        self._stats = {}

    def condition(self, images, text_raw, valid_hw):
        """-> cond [B, vis_dim + text_dim] (một lần mỗi ảnh)."""
        return torch.cat([self.vision(images, valid_hw), self.text_proj(text_raw)], dim=-1)

    # ------------------------------------------------------------------ vân tay CLIP frozen (không nằm trong state_dict)
    @staticmethod
    def _sha_row(sha):
        return torch.tensor(list(sha), dtype=torch.uint8)

    def set_text_fingerprint(self, sha):
        """Ghi vân tay CLIP text (`TextTable.token_sha`) vào `clip_sha[1]` — gọi TRƯỚC khi nạp checkpoint để nạp tự kiểm."""
        self.clip_sha[1] = self._sha_row(sha).to(self.clip_sha.device)

    def check_text_fingerprint(self, sha):
        """Sau khi nạp checkpoint (eval): CLIP text vừa tải phải đúng bản lúc train."""
        if self.clip_sha[1].any() and not torch.equal(self.clip_sha[1].cpu(), self._sha_row(sha)):
            raise RuntimeError("CLIP text (token chữ của refiner) KHÁC bản lúc train checkpoint — kiểm refiner.text_model_name / "
                               "HF cache")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """`clip_sha`: hàng đã có giá trị ở model phải TRÙNG checkpoint; hàng còn 0 (text chưa đặt khi eval) nhận giá trị checkpoint."""
        key = prefix + "clip_sha"
        cur = self.clip_sha.clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if key not in state_dict:
            return
        inc = state_dict[key].to(cur.device)
        for r, what in enumerate(("CLIP ViT (vision)", "CLIP text")):
            if cur[r].any() and inc[r].any() and not torch.equal(cur[r], inc[r]):
                error_msgs.append(f"{key}: vân tay {what} của model (tải từ HF) KHÁC checkpoint — CLIP frozen không đúng bản đã train")
            if cur[r].any() and not inc[r].any():
                self.clip_sha[r] = cur[r]

    # ------------------------------------------------------------------ khử nhiễu
    def refiner_context(self, images, text_tokens):
        """Ngữ cảnh refiner (một lần mỗi ảnh): RGB = 3 kênh đầu, density = kênh 4."""
        if text_tokens is None:
            raise ValueError("refiner cần `text_tokens` (token chữ CLIP, TextTable.tokens)")
        return self.refiner.encode_context(images[:, :3], images[:, 3:4], *text_tokens)

    def predict_noise(self, x, t, cond, ctx=None, ctx_index=None, obj=None):
        """x [R,4], t [R], cond [R,D], ctx (`refiner_context`, C hàng) + ctx_index [R] | None (eval `_norefine`: refined = x_t),
        obj (token, mask) của `object_tokens` | None (eval `_noobj`) -> (ε̂ [R,4], refined [R,4])."""
        refined = self.refiner(x, t, ctx, ctx_index) if ctx is not None else x
        unet_in = torch.cat([x, refined.to(x.dtype)], dim=-1)
        eps = self.noise_net(unet_in.unsqueeze(1), t, cond, *(obj or (None, None))).squeeze(1)
        return eps, refined

    def pop_stats(self):
        """-> thống kê loss lần forward train gần nhất (loss_eps, loss_aux, loss_giou, delta_abs, aux_ratio) rồi xoá."""
        st, self._stats = self._stats, {}
        return st

    def object_tokens(self, objects, canvas, cond, generator=None, cap=False):
        """objects: list B tensor [Mᵢ,4] xyxy pixel canvas -> (token 6 tầng, mask). Box vật mã hoá như đích, vào U-Net dạng gương
        [o ; o] (refined của box sạch = chính nó). `cap` (train): > obj_max vật thì lấy ngẫu nhiên obj_max theo `generator`."""
        if objects is None:
            raise ValueError("BoxPolicy cần `objects` (box các vật đang có)")
        whwh = torch.full((4,), float(canvas))
        units = []
        for o in objects:
            o = torch.as_tensor(o, dtype=torch.float32).reshape(-1, 4).cpu()
            if cap and len(o) > self.obj_max:
                seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=generator, device=generator.device)) if generator \
                    is not None else 0
                o = o[torch.randperm(len(o), generator=torch.Generator().manual_seed(seed))[: self.obj_max]]
            units.append(boxes_to_unit(o, whwh, self.box_coder))
        objs, mask = pad_objects(units, cond.device)
        return self.noise_net.encode_objects(torch.cat([objs, objs], -1), mask, cond), mask

    def encode(self, images, text_raw, valid_hw, objects, text_tokens, generator=None, cap=False):
        """Phần không phụ thuộc t, một lần mỗi ảnh -> (cond, obj, ctx)."""
        cond = self.condition(images, text_raw, valid_hw)
        return cond, self.object_tokens(objects, images.shape[-1], cond, generator, cap), self.refiner_context(images, text_tokens)

    def forward(self, images, text_raw, valid_hw, x0, k=1, generator=None, objects=None, text_tokens=None):
        """Loss train, k bộ (t, ε) mỗi ảnh. x0 [B,4] mã hoá; `objects` list B box vật xyxy pixel canvas; `text_tokens` (token, mask)."""
        cond, obj, ctx = self.encode(images, text_raw, valid_hw, objects, text_tokens, generator, cap=True)
        return self.eps_loss(cond, x0, k, generator, obj, ctx)

    @staticmethod
    def _repeat_obj(obj, n):
        return None if obj is None else ([t.repeat_interleave(n, 0) for t in obj[0]], obj[1].repeat_interleave(n, 0))

    def eps_loss(self, cond, x0, k=1, generator=None, obj=None, ctx=None):
        """Loss từ phần đã mã hoá (B ảnh): k bộ (t, ε) / ảnh có t khác nhau nên ngữ cảnh refiner nhân thành B·k hàng."""
        obj = self._repeat_obj(obj, k)
        B = cond.shape[0]
        cond = cond.repeat_interleave(k, dim=0)
        x0 = x0.repeat_interleave(k, dim=0)
        dev = x0.device
        if k > 1:
            ctx = self.refiner.index_context(ctx, torch.arange(B, device=dev).repeat_interleave(k))
        t = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=dev, generator=generator)
        noise = torch.randn(x0.shape, device=dev, generator=generator)
        noisy = q_sample(x0, t, noise, self.alphas_cumprod)
        pred, refined = self.predict_noise(noisy, t, cond, ctx, None, obj)
        loss = F.mse_loss(pred, noise)
        with torch.autocast(device_type=dev.type, enabled=False):          # refined fp32 (refiner tắt autocast)
            aux = F.mse_loss(refined, x0.to(refined.dtype))
            giou = None
            if self.aux_giou_weight > 0:                                   # DELTA5.x: GIoU trên box giải mã (tỉ lệ canvas)
                giou = (1 - generalized_box_iou_cxcywh(unit_fraction(refined, self.box_coder),
                                                       unit_fraction(x0.to(refined.dtype), self.box_coder))).mean()
            self._stats = {"loss_eps": loss.detach().float(), "loss_aux": aux.detach().float(),
                           "delta_abs": (refined - noisy.to(refined.dtype)).abs().mean().detach().float(),
                           "aux_ratio": (aux / F.mse_loss(noisy.to(refined.dtype), x0.to(refined.dtype)).clamp_min(1e-12))
                           .detach().float()}
            if giou is not None:
                self._stats["loss_giou"] = giou.detach()
        if giou is not None:
            aux = aux + self.aux_giou_weight * giou
        if self.refiner_aux_weight > 0:
            loss = loss + self.refiner_aux_weight * aux
        return loss

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, n_samples, generator=None, sampler="mock", objects=None, text_tokens=None,
               use_objects=True, use_refiner=True, ddim_steps=None, eta=1.0, clip_sample=True):
        """n_samples mẫu độc lập mỗi ảnh -> [B, n_samples, 4] mã hoá. `use_objects` / `use_refiner` False = eval `_noobj` / `_norefine`."""
        cond, obj, ctx = self.encode(images, text_raw, valid_hw, objects, text_tokens)
        return self.sample_from_cond(cond, n_samples, generator, sampler, obj if use_objects else None, ctx if use_refiner else None,
                                     ddim_steps=ddim_steps, eta=eta, clip_sample=clip_sample)

    @torch.no_grad()
    def sample_from_cond(self, cond, n_samples, generator=None, sampler="mock", obj=None, ctx=None, ddim_steps=None, eta=1.0,
                         clip_sample=True):
        """Như `sample` từ phần đã mã hoá (`encode`). Mọi hàng cùng t ở mỗi bước ⇒ refiner tính luồng ngữ cảnh một lần mỗi ảnh."""
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler {sampler!r} không thuộc {SAMPLERS}")
        B = cond.shape[0]
        cond = cond.repeat_interleave(n_samples, dim=0)
        obj = self._repeat_obj(obj, n_samples)
        ci = None if ctx is None else torch.arange(B, device=cond.device).repeat_interleave(n_samples)

        def fn(x, t):
            return self.predict_noise(x, t, cond, ctx, ci, obj)[0]
        run = mock_sample if sampler == "mock" else partial(ddim_sample, ddim_steps=ddim_steps, eta=eta, clip_sample=clip_sample)
        return run(fn, cond.shape[0], self.alphas_cumprod, generator=generator, device=cond.device).view(B, n_samples, 4)


# ----------------------------------------------------------------------------- dựng / nạp từ config

def _check_supported(m):
    """Branch `delta` chỉ giữ kiến trúc DELTA4+; checkpoint / config cũ hơn: branch `new`."""
    bad = {k: m.get(k) for k, want in (("arch", "box_policy"), ("vision", "r18_paper"), ("obj_attn", True)) if m.get(k) != want}
    bad.update({k: m[k] for k in ("obj_k",) if m.get(k) is not None})
    if m.get("use_condition", True) is not True:
        bad["use_condition"] = m["use_condition"]
    r = clip_refiner.refiner_config(m.get("refiner"))
    if r is None or r["unet_input"] != "concat":
        bad["refiner"] = None if r is None else r["unet_input"]
    if bad:
        raise ValueError(f"config không phải DELTA4+ ({bad}) — kiến trúc cũ ở branch `new`")


def build_model(cfg, pretrained_backbone=None):
    m, d = cfg["model"], cfg["diffusion"]
    _check_supported(m)
    return BoxPolicy(in_channels=m.get("in_channels", 4),
                     pretrained_backbone=m.get("pretrained_backbone", True) if pretrained_backbone is None else pretrained_backbone,
                     vis_dim=m["vis_dim"], text_in=m.get("text_dim", 512), text_dim=m["text_proj_dim"], step_embed_dim=m["step_embed_dim"],
                     down_dims=m["down_dims"], kernel_size=m["kernel_size"], n_groups=m["n_groups"], num_timesteps=d["num_timesteps"],
                     beta_start=d["beta_start"], beta_end=d["beta_end"], ss_mask=m.get("ss_mask", True), obj_heads=m.get("obj_heads", 4),
                     obj_max=m.get("obj_max", 300), refiner=m["refiner"], box_coder=m.get("box_coder"))


def load_checkpoint(ck, cfg=None):
    """Checkpoint của train.py (đường dẫn | dict) -> (model CPU, cfg, iter); cfg mặc định = config lưu trong checkpoint."""
    if not isinstance(ck, dict):
        ck = torch.load(ck, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:
        raise ValueError("checkpoint CE-Loc gốc của bài — eval ở branch `new`")
    cfg = cfg or ck["config"]
    model = build_model(cfg, pretrained_backbone=False)
    model.load_state_dict(ck["model"])
    return model, cfg, ck.get("iter")
