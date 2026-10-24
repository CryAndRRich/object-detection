"""Model: đối chiếu bản sao NGUYÊN VĂN code tác giả (BoxRefiner `refs/CE-Loc-update/`, BoxCoder / GIoU / sampler `refs/CE-Loc-logScale/`
— chép vào đây, không import refs), công thức loss, U-Net box vật, vân tay CLIP, eval dùng chung lượt mã hoá. CLIP giả, không tải gì."""

import math
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from tests.ce_localization.helpers import (FAKE_TEXT_SHA, SMALL_REFINER, _fake_load_clip, _fake_text_table, _patch_clip, small_batch,
                                           small_policy)

LOG = {"type": "log", "min_size_ratio": 1 / 128}


@pytest.fixture(autouse=True)
def _fake_clip(monkeypatch):
    _patch_clip(monkeypatch)


# ----------------------------------------------------------------------------- bản sao nguyên văn code tác giả

def _author_rope(x, pos, theta):
    dh = x.shape[-1]
    quarter = dh // 4
    freqs = theta ** (-torch.arange(quarter, device=x.device, dtype=torch.float32) / quarter)
    angles = (pos.float()[..., None] * freqs).flatten(-2)
    cos, sin = angles.cos()[:, None].to(x.dtype), angles.sin()[:, None].to(x.dtype)
    x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _author_block(self, x, pos, attn_mask, t_emb):
    shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(t_emb)[:, None].chunk(6, dim=-1)
    B, N, D = x.shape
    h = self.norm1(x) * (1 + scale1) + shift1
    q, k, v = self.qkv(h).view(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k = _author_rope(q, pos, self.rope_theta), _author_rope(k, pos, self.rope_theta)
    h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
    x = x + gate1 * self.proj(h.transpose(1, 2).reshape(B, N, D))
    h = self.norm2(x) * (1 + scale2) + shift2
    return x + gate2 * self.mlp(h)


def _author_sample_tokens(self, box, img_map):
    B, g, k = box.shape[0], self.grid, self.roi_size
    cx = ((box[:, 0] + 1) / 2 * g).clamp(0, g)
    cy = ((box[:, 1] + 1) / 2 * g).clamp(0, g)
    w = ((box[:, 2] + 1) / 2 * g).clamp(0.5, g)
    h = ((box[:, 3] + 1) / 2 * g).clamp(0.5, g)
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    rois = torch.stack([torch.arange(B, device=box.device, dtype=box.dtype), x1, y1, x2, y2], dim=1)
    feats = roi_align(img_map.contiguous(), rois, output_size=k, spatial_scale=1.0, sampling_ratio=2, aligned=True)
    tokens = feats.flatten(2).transpose(1, 2)
    steps = (torch.arange(k, device=box.device, dtype=box.dtype) + 0.5) / k
    xs = x1[:, None] + steps * (x2 - x1)[:, None]
    ys = y1[:, None] + steps * (y2 - y1)[:, None]
    pos = torch.stack([xs[:, None, :].expand(B, k, k), ys[:, :, None].expand(B, k, k)], dim=-1)
    return tokens, pos.reshape(B, k * k, 2)


def _author_attn_mask(self, text_mask, n_img, n_samp):
    B, L = text_mask.shape
    gid = torch.cat([torch.full((L,), 0), torch.full((n_img,), 1), torch.full((n_samp,), 2)])
    flow = self.group_allow[gid][:, gid]
    key_valid = torch.cat([text_mask, text_mask.new_ones(B, n_img + n_samp)], dim=1)
    return (flow[None] & key_valid[:, None, :])[:, None]


def _author_forward(self, sample, timestep, ctx):
    """`BoxRefiner.forward` của tác giả (ctx một hàng / mẫu — `expand_context`)."""
    B, g = sample.shape[0], self.grid
    t_emb = self.time_emb(timestep)
    text_tok, img_tok = ctx['text_tokens'] + self.type_emb[0], ctx['image_tokens'] + self.type_emb[1]
    L, n_img, n_samp = text_tok.shape[1], img_tok.shape[1], self.roi_size ** 2
    coords = torch.arange(g, dtype=sample.dtype) + 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    img_pos = torch.stack([xx, yy], dim=-1).reshape(1, n_img, 2).expand(B, -1, -1)
    text_pos = torch.full((B, L, 2), g / 2, dtype=sample.dtype)
    attn_mask = _author_attn_mask(self, ctx['text_mask'], n_img, n_samp)
    box = sample
    for blocks, head in zip(self.stages, self.heads):
        box_in = box.detach()
        samp_tok, samp_pos = _author_sample_tokens(self, box_in, ctx['image_map'])
        samp_tok = samp_tok + self.box_emb(box_in)[:, None] + self.type_emb[2]
        x = torch.cat([text_tok, img_tok, samp_tok], dim=1)
        pos = torch.cat([text_pos, img_pos, samp_pos], dim=1)
        for blk in blocks:
            x = _author_block(blk, x, pos, attn_mask, t_emb)
        box = box + head(x[:, -n_samp:].flatten(1))
    return box


class _AuthorBoxCoder:
    """`BoxCoder` của `refs/CE-Loc-logScale/utils/normalization.py` (encode / decode)."""
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.type, self.min_size_ratio = cfg.get('type', 'linear'), cfg.get('min_size_ratio', 1 / 128)
        self.log_min = math.log(self.min_size_ratio)

    def encode(self, box, canvas_size):
        rel = box / box.new_tensor([float(canvas_size)] * 4)
        out = rel * 2 - 1
        if self.type == 'log':
            size = rel[..., 2:].clamp(min=self.min_size_ratio).log()
            out = torch.cat([out[..., :2], (size - self.log_min) / -self.log_min * 2 - 1], dim=-1)
        return out

    def decode(self, box, canvas_size=1.0):
        rel = (box + 1) / 2
        if self.type == 'log':
            size = ((box[..., 2:].clamp(-5, 5) + 1) / 2 * -self.log_min + self.log_min).exp()
            rel = torch.cat([rel[..., :2], size], dim=-1)
        return rel * box.new_tensor([float(canvas_size)] * 4)


def _author_xyxy(box):
    w, h = box[..., 2].clamp(min=0), box[..., 3].clamp(min=0)
    return torch.stack([box[..., 0] - w / 2, box[..., 1] - h / 2, box[..., 0] + w / 2, box[..., 1] + h / 2], dim=-1)


def _author_giou(box1, box2):
    a, b = _author_xyxy(box1), _author_xyxy(box2)
    iw = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    ih = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    inter = iw * ih
    union = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]) + (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1]) - inter
    iou = inter / (union + 1e-6)
    enclose = (torch.maximum(a[..., 2], b[..., 2]) - torch.minimum(a[..., 0], b[..., 0])) * \
              (torch.maximum(a[..., 3], b[..., 3]) - torch.minimum(a[..., 1], b[..., 1]))
    return iou - (enclose - union) / (enclose + 1e-6)


def _author_sample(predict, alphas_cumprod, num_timesteps, B, num_inference_steps=None, eta=1.0, clip_sample=True, generator=None):
    """Vòng `ObjectPlacementPolicy.sample` của bản log-scale."""
    steps = num_inference_steps or num_timesteps
    timesteps = torch.linspace(num_timesteps - 1, 0, steps).round().long().unique_consecutive()
    box = torch.randn((B, 4), generator=generator)
    for i, t in enumerate(timesteps):
        eps = predict(box, t.expand(B))
        ab_t = alphas_cumprod[t]
        ab_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else box.new_tensor(1.0)
        x0 = (box - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_sample:
            x0 = x0.clamp(-1, 1)
            eps = (box - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
        box = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(timesteps):
            box = box + sigma * torch.randn(box.shape, generator=generator)
    return box


# ----------------------------------------------------------------------------- refiner

def _refiner(dtype=torch.float64):
    """ClipBoxRefiner nhỏ, 2 stage; AdaLN cuối + head khác 0 (khởi tạo của tác giả = identity, không kiểm gì)."""
    from ce_localization.models.clip_refiner import ClipBoxRefiner
    torch.manual_seed(0)
    vis, th = _fake_load_clip("openai/clip-vit-base-patch16")
    r = ClipBoxRefiner({"d_model": 32, "num_heads": 2, "num_layers": 3, "num_stages": 2}, vis, th)
    for blocks in r.stages:
        for blk in blocks:
            nn.init.normal_(blk.ada[-1].weight, std=0.05)
            nn.init.normal_(blk.ada[-1].bias, std=0.05)
    for h in r.heads:
        nn.init.normal_(h[-1].weight, std=0.05)
    r.time_emb[1].register_forward_pre_hook(lambda mod, a: (a[0].to(mod.weight.dtype),))   # SinusoidalPosEmb luôn ra fp32
    return r.to(dtype)


def _ctx(r, B=2, dtype=torch.float64):
    g = torch.Generator().manual_seed(1)
    rgb, den = torch.rand(B, 3, 128, 128, generator=g, dtype=dtype), torch.rand(B, 1, 128, 128, generator=g, dtype=dtype)
    lens = [3, 5][:B]
    tok = torch.randn(B, max(lens), 512, generator=g, dtype=dtype)
    mask = torch.zeros(B, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
        tok[i, n:] = 0.0
    return r.encode_context(rgb, den, tok, mask)


@pytest.mark.parametrize("shared", [False, True])
def test_refiner_split_equals_author_forward(shared):
    """Luồng ngữ cảnh tính một lần mỗi (ảnh, t) == `forward` nguyên văn của tác giả: đầu ra VÀ grad mọi tham số (float64, 2 stage,
    chữ dài khác nhau). `shared`: 3 mẫu / ảnh cùng t (vòng lấy mẫu); fp32 cũng sát."""
    r = _refiner()
    idx, t = (torch.arange(2).repeat_interleave(3), torch.tensor([5, 5, 5, 12, 12, 12])) if shared else (None, torch.tensor([3, 17]))
    x = torch.randn(len(t), 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.7
    out = r(x, t, _ctx(r), idx)
    out.pow(2).sum().backward()
    g_ours = {n: p.grad.clone() for n, p in r.named_parameters() if p.grad is not None}
    r.zero_grad()
    ctx2 = _ctx(r)
    ref = _author_forward(r, x, t, ctx2 if idx is None else r.index_context(ctx2, idx))
    assert torch.allclose(out, ref, atol=1e-10, rtol=0) and not torch.allclose(ref, x)
    ref.pow(2).sum().backward()
    g_ref = {n: p.grad.clone() for n, p in r.named_parameters() if p.grad is not None}
    assert set(g_ours) == set(g_ref) and len(g_ref) > 10
    for n in g_ref:
        assert torch.allclose(g_ours[n], g_ref[n], atol=1e-9, rtol=1e-7), n
    r32 = _refiner(torch.float32)
    ctx = _ctx(r32, dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(r32(x.float(), t, ctx, idx), _author_forward(r32, x.float(), t, ctx if idx is None else
                                                                          r32.index_context(ctx, idx)), atol=1e-5)


def test_refiner_state_dict_names_match_author():
    """Tên + shape key = state_dict `BoxRefiner` của tác giả trừ CLIP frozen (không lưu, không là tham số)."""
    from ce_localization.models.clip_refiner import REFINER_DEFAULTS, ClipBoxRefiner
    vis, th = _fake_load_clip("openai/clip-vit-base-patch16")
    r = ClipBoxRefiner(REFINER_DEFAULTS, vis, th)
    want = {"image_proj.0.weight", "image_proj.0.bias", "image_proj.1.weight", "image_proj.1.bias", "text_proj.0.weight",
            "text_proj.0.bias", "text_proj.1.weight", "text_proj.1.bias", "density_proj.weight", "density_proj.bias", "type_emb",
            "box_emb.proj.0.weight", "box_emb.proj.0.bias", "box_emb.proj.2.weight", "box_emb.proj.2.bias", "time_emb.1.weight",
            "time_emb.1.bias", "time_emb.3.weight", "time_emb.3.bias"}
    for li in range(4):
        want |= {f"stages.0.{li}.{n}" for n in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias", "mlp.0.weight", "mlp.0.bias",
                                                "mlp.3.weight", "mlp.3.bias", "ada.1.weight", "ada.1.bias")}
    want |= {f"heads.0.{n}" for n in ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias")}
    assert set(r.state_dict()) == want and r.state_dict()["heads.0.1.weight"].shape == (256, 256 * 9)
    assert {id(p) for p in vis.parameters()}.isdisjoint({id(p) for p in r.parameters()})


# ----------------------------------------------------------------------------- box coder, sampler

def test_box_coder_giou_match_author():
    """boxes_to_unit / unit_to_boxes (log) == `BoxCoder` của tác giả; GIoU == `generalized_box_iou` của tác giả; linear trả chính tensor;
    RoIAlign của refiner trên box log == trên box đã giải mã."""
    from ce_localization.models.box_policy import boxes_to_unit, unit_to_boxes
    from ce_localization.utils.box_coder import box_coder_config, generalized_box_iou_cxcywh, to_linear_unit, unit_fraction
    g = torch.Generator().manual_seed(0)
    cxcy = torch.rand(50, 2, generator=g) * 512
    wh = torch.exp(torch.empty(50, 2).uniform_(np.log(2), np.log(500), generator=g))
    cxcywh = torch.cat([cxcy, wh], -1).double()
    xyxy = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], -1)
    canvas = torch.full((4,), 512.0, dtype=torch.float64)
    coder, ref = box_coder_config(LOG), _AuthorBoxCoder(LOG)
    u = boxes_to_unit(xyxy, canvas, coder)
    assert torch.allclose(u, ref.encode(cxcywh, 512), atol=1e-12)
    noisy = u + torch.randn(u.shape, generator=g, dtype=torch.float64) * 3
    assert torch.allclose(unit_to_boxes(noisy, canvas, coder), _author_xyxy(ref.decode(noisy, 512)), atol=1e-9)
    assert (unit_fraction(noisy, coder)[:, 2:] > 0).all()
    a, b = torch.rand(9, 4, generator=g) * 2 - 1, torch.rand(9, 4, generator=g) * 2 - 1
    assert torch.allclose(generalized_box_iou_cxcywh(unit_fraction(a, coder), unit_fraction(b, coder)),
                          _author_giou(ref.decode(a), ref.decode(b)), atol=1e-6)
    lin = box_coder_config(None)
    assert to_linear_unit(u, lin) is u and torch.allclose(boxes_to_unit(xyxy, canvas, lin), _AuthorBoxCoder().encode(cxcywh, 512))
    r = _refiner()
    ctx = _ctx(r)
    box = torch.rand(2, 4, generator=g, dtype=torch.float64) * 2 - 1
    tok_lin, _ = r._sample_tokens(to_linear_unit(box, coder), ctx["image_map"])
    r.box_coder = coder
    assert torch.equal(r._sample_tokens(box, ctx["image_map"])[0], tok_lin)
    with pytest.raises(ValueError, match="box_coder"):
        box_coder_config({"type": "sqrt"})


@pytest.mark.parametrize("steps,eta,clip", [(None, 1.0, True), (7, 0.0, True), (5, 0.5, False)])
def test_samplers_match_author_and_paper(steps, eta, clip):
    """`ddim_sample` == vòng `sample` của tác giả (cùng generator); `mock_sample` == vòng của CE-Loc gốc (`x -= eps / 100`)."""
    from ce_localization.models.box_policy import ddim_sample, mock_sample
    from ce_localization.utils.diffusion_math import linear_alphas_cumprod
    ab = linear_alphas_cumprod(20)
    W = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))

    def fn(x, t):
        return torch.tanh(x @ W) * (1 + t[:, None].float() / 20)
    ours = ddim_sample(fn, 6, ab, ddim_steps=steps, eta=eta, clip_sample=clip, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(ours, _author_sample(fn, ab, 20, 6, steps, eta, clip, generator=torch.Generator().manual_seed(5)), atol=1e-6)
    x = torch.randn(6, 4, generator=torch.Generator().manual_seed(5))
    for t in reversed(range(20)):
        x = x - fn(x, torch.full((6,), t)) / 20
    assert torch.allclose(mock_sample(fn, 6, ab, steps=20, generator=torch.Generator().manual_seed(5)), x)
    with pytest.raises(ValueError, match="mock"):
        mock_sample(fn, 6, ab, steps=21)


# ----------------------------------------------------------------------------- BoxPolicy

def test_box_policy_loss_formula_grads_and_identity():
    """Khởi tạo như tác giả: refined == x_t, ε̂ = U-Net([x_t ; x_t]); loss = ε-MSE + 0,1·[MSE(refined, x0) + w·(1 − GIoU)]; mọi tham số có
    grad (DDP không find_unused); box vật mã hoá cùng coder log; cap obj_max khi train; thiếu token chữ -> lỗi."""
    m = small_policy(refiner={"aux_giou_weight": 0.5}, box_coder=LOG, obj_max=1).train()
    x, text, vhw, x0, tok, objs = small_batch()
    with torch.no_grad():
        cond, obj, ctx = m.encode(x, text, vhw, objs, tok)
        xt, t = torch.randn(6, 4), torch.full((6,), 7)
        eps, refined = m.predict_noise(xt, t, cond.repeat_interleave(3, 0), ctx, torch.arange(2).repeat_interleave(3))
        assert torch.equal(refined, xt)
        assert torch.allclose(eps, m.noise_net(torch.cat([xt, xt], -1).unsqueeze(1), t, cond.repeat_interleave(3, 0)).squeeze(1))
    for blk in m.refiner.stages[0]:
        nn.init.normal_(blk.ada[-1].weight, std=0.02)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=0.02)
    for xa in m.noise_net.obj_xattn:
        nn.init.normal_(xa.attn.out_proj.weight, std=0.02)
    loss = m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), objects=objs, text_tokens=tok)
    st = m.pop_stats()
    assert set(st) == {"loss_eps", "loss_aux", "delta_abs", "aux_ratio", "loss_giou"} and float(st["delta_abs"]) > 0
    assert torch.allclose(loss, st["loss_eps"] + 0.1 * (st["loss_aux"] + 0.5 * st["loss_giou"]), atol=1e-6)
    loss.backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []
    with pytest.raises(ValueError, match="text_tokens"):
        m(x, text, vhw, x0, objects=objs)


def test_object_cap_random_cpu_generator():
    """Train (`cap`): > obj_max vật -> lấy NGẪU NHIÊN obj_max theo `obj_generator` CPU (cùng seed = cùng tập, khác seed = khác tập);
    eval giữ hết."""
    m = small_policy(obj_max=3)
    objs = [torch.tensor([[i * 5.0, 0.0, i * 5.0 + 4, 4.0] for i in range(10)]), torch.tensor([[0.0, 0.0, 4.0, 4.0]])]
    cond = torch.randn(2, m.cond_dim)
    tok = lambda seed, cap=True: m.object_tokens(objs, 64, cond, torch.Generator().manual_seed(seed), cap)  # noqa: E731
    (t1, m1), (t2, _), (t3, _) = tok(1), tok(1), tok(2)
    assert m1.sum(1).tolist() == [3, 1] and torch.equal(t1[0], t2[0]) and not torch.equal(t1[0], t3[0])
    assert tok(1, cap=False)[1].sum(1).tolist() == [10, 1]


def test_unet_object_tokens_zero_init_bucket_and_kv():
    """Cross-attn box vật out_proj = 0 ⇒ lúc khởi tạo = U-Net không box vật; token vật không đổi khi đệm hàng (`obj_bucket`) hay khi
    truyền sẵn chỉ số nonzero; ảnh 0 vật cộng 0; `ObjKV` (lấy mẫu: K / V mỗi ảnh một lần) == token nhân theo số mẫu."""
    from ce_localization.models.unet1d import ConditionalUnet1D
    torch.manual_seed(0)
    u = ConditionalUnet1D(8, 16, 32, (16, 32, 64), 3, 4, obj_bucket=1)
    objs, mask = torch.randn(3, 5, 8), torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    cond = torch.randn(3, 16)
    tok1 = u.encode_objects(objs, mask, cond)
    u.obj_bucket = 512
    tok2 = u.encode_objects(objs, mask, cond)
    assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(tok1, tok2)) and float(tok1[0][2].abs().sum()) == 0
    x, t = torch.randn(3, 1, 8), torch.tensor([3, 50, 99])
    assert torch.allclose(u(x, t, cond, tok1, mask), u(x, t, cond))
    for xa in u.obj_xattn:
        nn.init.normal_(xa.attn.out_proj.weight, std=0.1)
    out = u(x, t, cond, tok1, mask)
    assert not torch.allclose(out[:2], u(x, t, cond)[:2]) and torch.allclose(out[2], u(x, t, cond)[2])
    tok3 = u.encode_objects(objs, mask, cond, mask.nonzero(as_tuple=True))
    assert all(torch.equal(a, b) for a, b in zip(tok2, tok3))
    n = 4
    x4, t4, c4 = torch.randn(3 * n, 1, 8), torch.randint(0, 100, (3 * n,)), cond.repeat_interleave(n, 0)
    rep = u(x4, t4, c4, [tk.repeat_interleave(n, 0) for tk in tok1], mask.repeat_interleave(n, 0))
    assert torch.allclose(u(x4, t4, c4, u.object_kv(tok1, mask, n)), rep, atol=1e-5)


def test_clip_out_of_state_dict_and_fingerprint(monkeypatch):
    """CLIP frozen không vào state_dict / parameters, theo `.to()`; `clip_sha`: CLIP vision khác bản lúc train ⇒ nạp lỗi; hàng text còn 0
    nhận của checkpoint, khác thì `check_text_fingerprint` báo lỗi."""
    import ce_localization.models.clip_refiner as cr
    a = small_policy()
    clip = a.refiner.vision_backbone
    assert not any("vision_backbone" in k for k in a.state_dict()) and "clip_sha" in a.state_dict()
    assert {id(p) for p in clip.parameters()}.isdisjoint({id(p) for p in a.parameters()})
    a.double()
    assert next(clip.parameters()).dtype == torch.float64
    a.float()
    a.set_text_fingerprint(FAKE_TEXT_SHA)
    sd = a.state_dict()
    monkeypatch.setattr(cr, "load_clip", lambda name, text_name: _fake_load_clip(name, seed=1))
    with pytest.raises(RuntimeError, match="vân tay CLIP ViT"):
        small_policy().load_state_dict(sd)
    monkeypatch.setattr(cr, "load_clip", lambda name, text_name: _fake_load_clip(name, seed=0))
    c = small_policy()
    c.load_state_dict(sd)
    assert torch.equal(c.clip_sha, a.clip_sha)
    c.check_text_fingerprint(FAKE_TEXT_SHA)
    with pytest.raises(RuntimeError, match="CLIP text"):
        c.check_text_fingerprint(b"\x01" * 32)


def test_clip_cache_lazy_memmap_and_check(tmp_path):
    """`ClipCache`: đọc lười memmap (pickle sang worker KHÔNG kèm mảng), đúng hàng theo khoá; `check` từ chối cache khác CLIP / lớp ViT /
    AMP hoặc thiếu khoá."""
    import json
    import pickle
    from ce_localization.data.clip_cache import ClipCache
    m = small_policy()
    feats = np.random.default_rng(0).standard_normal((3, 49, 32)).astype(np.float32)
    np.save(tmp_path / "feats.npy", feats)
    cfg = {"data": {"image_size": 128}, "training": {"amp": True}}
    meta = {"clip_sha": bytes(m.clip_sha[0].tolist()).hex(), "vision_layer": m.refiner.vision_layer, "image_size": 128, "amp": False,
            "rows": {"a": 0, "b": 1, "c": 2}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    c = ClipCache(str(tmp_path))
    assert torch.equal(c.get("b"), torch.from_numpy(feats[1])) and c._arr is not None
    assert len(pickle.dumps(c)) < feats.nbytes / 4 and pickle.loads(pickle.dumps(c))._arr is None
    c.check(m, cfg, ["a", "c"])                                          # CPU: AMP không bật
    with pytest.raises(ValueError, match="thiếu"):
        c.check(m, cfg, ["a", "z"])
    m.refiner.vision_layer = -1
    with pytest.raises(ValueError, match="vision_layer"):
        c.check(m, cfg, ["a"])


def test_predict_add_shared_encoding_equals_separate_runs():
    """Eval nhiều biến thể (mock / ddim / _noobj / _norefine) dùng chung một lượt mã hoá ảnh / batch == từng biến thể chạy riêng (cùng
    seed) — trùng từng bit."""
    from ce_localization.data.turns import collate_add
    from ce_localization.engine.add_eval import predict_add
    m = small_policy().eval()
    for blk in m.refiner.stages[0]:
        nn.init.normal_(blk.ada[-1].weight, std=0.05)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=0.05)
    for xa in m.noise_net.obj_xattn:                                 # out_proj = 0 lúc khởi tạo ⇒ `_noobj` sẽ trùng bản gốc
        nn.init.normal_(xa.attn.out_proj.weight, std=0.05)
    x, _, vhw, _, _, objs = small_batch()
    items = [{"image": (x[i] * 255).to(torch.uint8), "target": torch.zeros(4), "holes": torch.tensor([[1.0, 1, 9, 9]]),
              "objects": objs[i], "valid_hw": tuple(vhw[i].tolist()), "text": ["apple", "cup"][i], "image_id": f"s{i}", "t": 1}
             for i in range(2)]

    class Loader(list):
        dataset = items
    loader = Loader([collate_add(items[:1]), collate_add(items[1:])])
    var = [("_mock", {"sampler": "mock"}), ("_ddim", {"sampler": "ddim", "ddim_steps": 7, "eta": 1.0}),
           ("_mock_noobj", {"sampler": "mock", "use_objects": False}), ("_ddim_norefine", {"sampler": "ddim", "ddim_steps": 7,
                                                                                          "use_refiner": False})]
    tt = _fake_text_table(["apple", "cup"])
    allv = predict_add(m, loader, tt, 3, seed=0, variants=var)
    for s, kw in var:
        one = predict_add(m, loader, tt, 3, seed=0, variants=[(s, kw)])[s]
        assert all(np.array_equal(a["boxes"], b["boxes"]) for a, b in zip(allv[s], one)), s
    assert not np.array_equal(allv["_mock"][0]["boxes"], allv["_mock_noobj"][0]["boxes"])


def test_build_model_only_delta4_plus():
    from ce_localization.models.box_policy import _check_supported
    from ce_localization.utils.config import load_config
    from tests.ce_localization.helpers import CFG_DIR, RUNS
    for run in RUNS:
        _check_supported(load_config(os.path.join(CFG_DIR, f"{run}.yaml"))["model"])
    m = load_config(os.path.join(CFG_DIR, "delta4.yaml"))["model"]
    for bad in ({"vision": "r50_fpn"}, {"obj_attn": False}, {"refiner": {**m["refiner"], "unet_input": "post"}}, {"use_condition": False},
                {"obj_k": 3}):
        with pytest.raises(ValueError, match="DELTA4"):
            _check_supported({**m, **bad})


def test_text_tokens_causal_padding_equivalence():
    """CLIP text causal + đệm phải: token thật của một tên tính riêng == tính trong batch đệm (cách tác giả) ⇒ bảng tính sẵn đúng;
    `TextTable.tokens` đệm 0 bên phải + mask."""
    from transformers import CLIPTextConfig, CLIPTextModel
    from ce_localization.models.text import TextTable
    torch.manual_seed(0)
    m = CLIPTextModel(CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                     vocab_size=100, max_position_embeddings=16)).eval()
    seqs = [[1, 5, 7, 2], [1, 9, 2], [1, 3, 4, 6, 8, 2]]
    ids, am = torch.zeros(3, 6, dtype=torch.long), torch.zeros(3, 6, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)], am[i, :len(s)] = torch.tensor(s), 1
    with torch.no_grad():
        batch = m(input_ids=ids, attention_mask=am).last_hidden_state
        for i, s in enumerate(seqs):
            one = m(input_ids=torch.tensor([s]), attention_mask=torch.ones(1, len(s), dtype=torch.long)).last_hidden_state[0]
            assert torch.allclose(one, batch[i, :len(s)], atol=1e-5)
    tt = TextTable({"a": torch.zeros(512), "bb": torch.zeros(512)}, {"a": torch.ones(2, 8), "bb": torch.full((4, 8), 2.0)}, b"x" * 32)
    pad, mask = tt.tokens(["bb", "a"], "cpu")
    assert pad.shape == (2, 4, 8) and mask.tolist() == [[True] * 4, [True, True, False, False]] and float(pad[1, 2:].abs().sum()) == 0


@pytest.mark.skipif(os.environ.get("CE_HF_TESTS") != "1", reason="G1-HF: CLIP thật từ HF — server / Kaggle với CE_HF_TESTS=1")
def test_hf_clip_real():
    """G1-HF: CLIP ViT thật của run (`CE_HF_CLIP`, mặc định B/32) — lưới = ảnh / patch, `encode_context` dùng đúng
    `hidden_states[-2][:, 1:]`; token chữ CLIP text B/16 = tokenize theo batch của tác giả; in vân tay."""
    import importlib
    import ce_localization.models.clip_refiner as cr
    importlib.reload(cr)                                             # bỏ CLIP giả của fixture
    from transformers import CLIPTextModel, CLIPTokenizer
    from ce_localization.models.text import encode_class_tokens
    b16 = "openai/clip-vit-base-patch16"
    name = os.environ.get("CE_HF_CLIP", "openai/clip-vit-base-patch32")
    vis, th = cr.load_clip(name, b16)
    res, g = vis.config.image_size, vis.config.image_size // vis.config.patch_size
    r = cr.ClipBoxRefiner({**SMALL_REFINER, "d_model": 256, "num_heads": 8, "clip_model_name": name}, vis, th).eval()
    assert th == 512 and r.grid == g and r.clip_res == res
    rgb = torch.rand(2, 3, 512, 512)
    with torch.no_grad():
        ctx = r.encode_context(rgb, torch.zeros(2, 1, 512, 512), torch.zeros(2, 3, 512), torch.ones(2, 3, dtype=torch.bool))
        px = F.interpolate(rgb, size=(res, res), mode="bicubic", align_corners=False, antialias=True)
        hid = vis(pixel_values=(px.clamp(0, 1) - r.clip_mean) / r.clip_std, output_hidden_states=True).hidden_states[-2][:, 1:]
        assert torch.allclose(ctx["image_tokens"], r.image_proj(hid) + r.density_proj(torch.zeros(2, g * g, 1)), atol=1e-5)
    names = ["apple", "hot air balloon", "candy piece", "sea shell"]
    table, sha = encode_class_tokens(names, b16)
    tok, model = CLIPTokenizer.from_pretrained(b16), CLIPTextModel.from_pretrained(b16).eval()
    inp = tok(names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**inp).last_hidden_state
    for i, n in enumerate(names):
        assert torch.allclose(table[n], ref[i][inp["attention_mask"][i].bool()], atol=1e-5), n
    print(f"\n[G1-HF] {name}: lưới {g}×{g}, ảnh {res} | vân tay CLIP ViT {cr.weights_sha256(vis).hex()} | CLIP text B/16 {sha.hex()}")
