"""DELTA (docs/EXPERIMENT_DELTA.md): BoxRefiner của tác giả gốc trong CE-Loc (`models/clip_refiner.py`, `models/box_policy.py`) —
DELTA1 / 1.1 (mục 4–8), DELTA2.x (đổi CLIP-ViT của refiner, mục 9.1), DELTA3 (refiner không U-Net, mục 9.2), DELTA4 (GAMMA4 + refiner,
mục 10.1), DELTA3.1 / 3.2 (refiner sau U-Net, sửa ε, mục 10.2), DELTA4.1 (refiner chèn giữa U-Net, mục 10.6), DELTA5.x (box mã hoá
log + GIoU + sampler của bản log-scale tác giả, mục 11 — đối chiếu bản sao nguyên văn `refs/CE-Loc-logScale/`).

Tham chiếu: bản sao NGUYÊN VĂN `BoxRefiner.forward` / `_sample_tokens` / `_attn_mask` / `RefinerBlock.forward` / `apply_rope_2d` của
`refs/CE-Loc-update/models/box_refiner.py` (chép vào đây — không import refs), chạy trên CÙNG tham số của `ClipBoxRefiner`. CLIP ViT thay
bằng CLIP tí hon dựng từ config (`helpers._fake_load_clip`), không tải gì.
"""

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision.ops import roi_align

from tests.ce_localization.helpers import (CFG_G, FAKE_TEXT_SHA, FAKE_VIT, _fake_load_clip, _fake_text_table, _fake_turn_index,
                                           _gamma2_cfg, _patch_clip, _run_train)

AUTHOR_REFINER = {"enabled": True, "clip_model_name": "openai/clip-vit-base-patch16", "vision_layer": -2, "d_model": 256,
                  "num_layers": 4, "num_heads": 8, "mlp_ratio": 4.0, "dropout": 0.0, "rope_theta": 100.0, "roi_size": 3,
                  "num_stages": 1, "use_density": True, "unet_input": "concat", "aux_loss_weight": 0.1,
                  "attention_flow": {"sample": ["sample", "image", "text"], "image": ["image", "text"], "text": ["text", "image"]}}


@pytest.fixture(autouse=True)
def _fake_clip(monkeypatch):
    """Mọi test ở đây: CLIP ViT giả (seed 0) — KHÔNG để `load_clip` thật đọc HF cache / tải mạng. Test nào cần CLIP khác thì patch lại."""
    _patch_clip(monkeypatch)


# ----------------------------------------------------------------------------- bản sao nguyên văn code tác giả (tham chiếu)

def _author_rope(x, pos, theta):
    dh = x.shape[-1]
    quarter = dh // 4
    freqs = theta ** (-torch.arange(quarter, device=x.device, dtype=torch.float32) / quarter)
    angles = (pos.float()[..., None] * freqs).flatten(-2)  # [B, N, Dh/2]
    cos = angles.cos()[:, None].to(x.dtype)
    sin = angles.sin()[:, None].to(x.dtype)
    x1, x2 = x[..., :dh // 2], x[..., dh // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _author_block(self, x, pos, attn_mask, t_emb):
    shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(t_emb)[:, None].chunk(6, dim=-1)

    B, N, D = x.shape
    h = self.norm1(x) * (1 + scale1) + shift1
    q, k, v = self.qkv(h).view(B, N, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
    q = _author_rope(q, pos, self.rope_theta)
    k = _author_rope(k, pos, self.rope_theta)
    h = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0)
    x = x + gate1 * self.proj(h.transpose(1, 2).reshape(B, N, D))

    h = self.norm2(x) * (1 + scale2) + shift2
    return x + gate2 * self.mlp(h)


def _author_sample_tokens(self, box, img_map):
    B, g, k = box.shape[0], self.grid, self.roi_size
    # [-1, 1] -> patch-grid units; clamp because noisy samples can be anywhere
    cx = ((box[:, 0] + 1) / 2 * g).clamp(0, g)
    cy = ((box[:, 1] + 1) / 2 * g).clamp(0, g)
    w = ((box[:, 2] + 1) / 2 * g).clamp(0.5, g)
    h = ((box[:, 3] + 1) / 2 * g).clamp(0.5, g)
    x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    rois = torch.stack([torch.arange(B, device=box.device, dtype=box.dtype), x1, y1, x2, y2], dim=1)
    feats = roi_align(img_map.contiguous(), rois, output_size=k, spatial_scale=1.0,
                      sampling_ratio=2, aligned=True)                 # [B, D, k, k]
    tokens = feats.flatten(2).transpose(1, 2)                        # [B, k*k, D] (y-major)

    # Continuous positions of each RoI cell center, same order as tokens
    steps = (torch.arange(k, device=box.device, dtype=box.dtype) + 0.5) / k
    xs = x1[:, None] + steps * (x2 - x1)[:, None]
    ys = y1[:, None] + steps * (y2 - y1)[:, None]
    pos = torch.stack([xs[:, None, :].expand(B, k, k), ys[:, :, None].expand(B, k, k)], dim=-1)
    return tokens, pos.reshape(B, k * k, 2)


def _author_attn_mask(self, text_mask, n_img, n_samp):
    B, L = text_mask.shape
    device = text_mask.device
    gid = torch.cat([
        torch.full((L,), 0, device=device),
        torch.full((n_img,), 1, device=device),
        torch.full((n_samp,), 2, device=device),
    ])
    flow = self.group_allow[gid][:, gid]                               # [N, N]
    key_valid = torch.cat([text_mask, text_mask.new_ones(B, n_img + n_samp)], dim=1)
    return (flow[None] & key_valid[:, None, :])[:, None]               # [B, 1, N, N]


def _author_forward(self, sample, timestep, ctx):
    """`BoxRefiner.forward` của tác giả (ctx đã một hàng / mẫu — `expand_context`)."""
    B, g = sample.shape[0], self.grid
    t_emb = self.time_emb(timestep)

    text_tok = ctx['text_tokens'] + self.type_emb[0]
    img_tok = ctx['image_tokens'] + self.type_emb[1]
    L, n_img, n_samp = text_tok.shape[1], img_tok.shape[1], self.roi_size ** 2

    # Image patch centers in grid units; text has no location, so place it at the canvas center
    coords = torch.arange(g, device=sample.device, dtype=sample.dtype) + 0.5
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    img_pos = torch.stack([xx, yy], dim=-1).reshape(1, n_img, 2).expand(B, -1, -1)
    text_pos = torch.full((B, L, 2), g / 2, device=sample.device, dtype=sample.dtype)
    attn_mask = _author_attn_mask(self, ctx['text_mask'], n_img, n_samp)

    box = sample
    for blocks, head in zip(self.stages, self.heads):
        # Detach the box used to build tokens (DETR-style iterative refinement)
        box_in = box.detach()
        samp_tok, samp_pos = _author_sample_tokens(self, box_in, ctx['image_map'])
        samp_tok = samp_tok + self.box_emb(box_in)[:, None] + self.type_emb[2]

        x = torch.cat([text_tok, img_tok, samp_tok], dim=1)
        pos = torch.cat([text_pos, img_pos, samp_pos], dim=1)
        for blk in blocks:
            x = _author_block(blk, x, pos, attn_mask, t_emb)

        # Keep only the refined sample tokens and map back to box space
        box = box + head(x[:, -n_samp:].flatten(1))
    return box


# ----------------------------------------------------------------------------- tiện ích

def _refiner(cfg=None, seed=0, randomize=True, dtype=torch.float64):
    """ClipBoxRefiner nhỏ (CLIP giả); `randomize`: lớp cuối AdaLN + head khác 0 (khởi tạo của tác giả = 0 ⇒ identity, không kiểm gì)."""
    from ce_localization.models.clip_refiner import ClipBoxRefiner
    torch.manual_seed(seed)
    vis, th = _fake_load_clip("openai/clip-vit-base-patch16")
    r = ClipBoxRefiner({"d_model": 32, "num_heads": 2, "num_layers": 3, "num_stages": 2, **(cfg or {})}, vis, th)
    if randomize:
        for blocks in r.stages:
            for blk in blocks:
                nn.init.normal_(blk.ada[-1].weight, std=0.05)
                nn.init.normal_(blk.ada[-1].bias, std=0.05)
        for h in r.heads:
            nn.init.normal_(h[-1].weight, std=0.05)
    # SinusoidalPosEmb (cả bản tác giả) luôn ra float32: ép về dtype tham số khi test float64 (fp32: no-op)
    r.time_emb[1].register_forward_pre_hook(lambda mod, a: (a[0].to(mod.weight.dtype),))
    return r.to(dtype)


def _ctx(r, B=2, seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    rgb = torch.rand(B, 3, 128, 128, generator=g, dtype=dtype)
    den = torch.rand(B, 1, 128, 128, generator=g, dtype=dtype)
    lens = [3, 5, 4, 2][:B]
    tok = torch.randn(B, max(lens), 512, generator=g, dtype=dtype)
    mask = torch.zeros(B, max(lens), dtype=torch.bool)
    for i, n in enumerate(lens):
        mask[i, :n] = True
        tok[i, n:] = 0.0
    return r.encode_context(rgb, den, tok, mask)


def _grads(m):
    return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}


def _policy(refiner=AUTHOR_REFINER, use_condition=True, seed=0, num_timesteps=20, **kw):
    from ce_localization.models.box_policy import BoxPolicy
    torch.manual_seed(seed)
    return BoxPolicy(in_channels=4, pretrained_backbone=False, num_timesteps=num_timesteps, vision="r18_paper", ss_mask=True,
                     refiner=refiner, use_condition=use_condition, **kw)


def _batch(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, 4, 128, 128, generator=g)
    vhw = torch.tensor([[96, 128], [128, 112]])[:B]
    tok = (torch.randn(B, 5, 512, generator=g), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)[:B])
    return x, torch.randn(B, 512, generator=g), vhw, torch.rand(B, 4, generator=g) * 2 - 1, tok


# ----------------------------------------------------------------------------- port đúng tác giả

@pytest.mark.parametrize("case", ["per_row", "shared", "flow_no_text"])
def test_refiner_split_equals_author_forward(monkeypatch, case):
    """Đường tách đôi (luồng ngữ cảnh một lần mỗi (ảnh, t)) == `forward` nguyên văn của tác giả: đầu ra VÀ grad mọi tham số (float64,
    AdaLN + head khác 0, 2 stage, chữ dài khác nhau). `shared`: 3 mẫu / ảnh cùng t (vòng mock / DDPM)."""
    flow = {"flow_no_text": {"sample": ["sample", "image"], "image": ["image", "text"], "text": ["text", "image"]}}.get(case)
    r = _refiner({"attention_flow": flow} if flow else None)
    ctx = _ctx(r)
    if case == "per_row":
        idx, t = None, torch.tensor([3, 17])
    else:
        idx, t = torch.arange(2).repeat_interleave(3), torch.tensor([5, 5, 5, 12, 12, 12])
    x = torch.randn(len(t), 4, generator=torch.Generator().manual_seed(2), dtype=torch.float64) * 0.7
    out = r(x, t, ctx, idx)
    out.pow(2).sum().backward()
    g_ours = _grads(r)
    r.zero_grad()
    ctx2 = _ctx(r)                                                   # đồ thị mới cho lần backward thứ hai
    ctx2 = ctx2 if idx is None else r.index_context(ctx2, idx)
    ref = _author_forward(r, x, t, ctx2)
    assert torch.allclose(out, ref, atol=1e-10, rtol=0), (out - ref).abs().max()
    assert not torch.allclose(ref, x)                                # khác identity: phép thử có nghĩa
    ref.pow(2).sum().backward()
    g_ref = _grads(r)
    assert set(g_ours) == set(g_ref) and len(g_ref) > 10
    for n in g_ref:
        assert torch.allclose(g_ours[n], g_ref[n], atol=1e-9, rtol=1e-7), n


def test_refiner_split_fp32_close_to_author():
    r = _refiner(dtype=torch.float32)
    ctx = _ctx(r, dtype=torch.float32)
    idx, t = torch.arange(2).repeat_interleave(4), torch.tensor([9] * 4 + [1] * 4)
    x = torch.randn(8, 4, generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a = r(x, t, ctx, idx)
        b = _author_forward(r, x, t, r.index_context(ctx, idx))
    assert torch.allclose(a, b, atol=1e-5)


def test_refiner_state_dict_names_match_author():
    """Tên + shape key = state_dict `BoxRefiner` của tác giả trừ CLIP frozen (`vision_backbone.*`, `text_backbone.*`, không lưu)."""
    from ce_localization.models.clip_refiner import ClipBoxRefiner
    vis, th = _fake_load_clip("openai/clip-vit-base-patch16")
    r = ClipBoxRefiner(AUTHOR_REFINER, vis, th)
    want = {"image_proj.0.weight", "image_proj.0.bias", "image_proj.1.weight", "image_proj.1.bias", "text_proj.0.weight",
            "text_proj.0.bias", "text_proj.1.weight", "text_proj.1.bias", "density_proj.weight", "density_proj.bias", "type_emb",
            "box_emb.proj.0.weight", "box_emb.proj.0.bias", "box_emb.proj.2.weight", "box_emb.proj.2.bias", "time_emb.1.weight",
            "time_emb.1.bias", "time_emb.3.weight", "time_emb.3.bias"}
    for li in range(4):
        want |= {f"stages.0.{li}.{n}" for n in ("qkv.weight", "qkv.bias", "proj.weight", "proj.bias", "mlp.0.weight", "mlp.0.bias",
                                                "mlp.3.weight", "mlp.3.bias", "ada.1.weight", "ada.1.bias")}
    want |= {f"heads.0.{n}" for n in ("0.weight", "0.bias", "1.weight", "1.bias", "3.weight", "3.bias")}
    sd = r.state_dict()
    assert set(sd) == want
    assert sd["stages.0.0.qkv.weight"].shape == (768, 256) and sd["heads.0.1.weight"].shape == (256, 256 * 9)
    assert sd["box_emb.proj.0.weight"].shape == (256, 4 * 2 * 8 + 4) and sd["type_emb"].shape == (3, 256)
    clip_ids = {id(p) for p in vis.parameters()}
    assert not any(id(p) in clip_ids for p in r.parameters())


def test_refiner_identity_at_init_and_unet_concat():
    """Khởi tạo như tác giả: refined == x_t; ε̂ = U-Net trên [x_t ; x_t] (8 kênh vào, 4 ra)."""
    m = _policy().eval()
    x_img, text, vhw, _, tok = _batch()
    assert m.noise_net.final_conv[1].out_channels == 4 and m.noise_net.down_modules[0][0].blocks[0].block[0].in_channels == 8
    with torch.no_grad():
        cond = m.condition(x_img, text, vhw)
        ctx = m.refiner_context(x_img, tok)
        x, t = torch.randn(6, 4), torch.full((6,), 7)
        ci = torch.arange(2).repeat_interleave(3)
        eps, refined = m.predict_noise(x, t, cond.repeat_interleave(3, 0), ctx, ci)
        assert torch.equal(refined, x)
        assert torch.allclose(eps, m.noise_net(torch.cat([x, x], -1).unsqueeze(1), t, cond.repeat_interleave(3, 0)).squeeze(1))


def test_clip_holder_moves_and_stays_out_of_params():
    m = _policy()
    clip = m.refiner.vision_backbone
    n_clip = sum(p.numel() for p in clip.parameters())
    assert not any("vision_backbone" in k for k in m.state_dict()) and "clip_sha" in m.state_dict()
    assert n_clip > 0
    assert {id(p) for p in clip.parameters()}.isdisjoint({id(p) for p in m.parameters()})
    m.double()
    assert next(clip.parameters()).dtype == torch.float64 and not next(clip.parameters()).requires_grad
    m.float().to(memory_format=torch.channels_last)
    w = clip.vision_model.embeddings.patch_embedding.weight
    assert w.dtype == torch.float32 and w.is_contiguous(memory_format=torch.channels_last)
    m.train()
    assert not clip.training


def test_clip_fingerprint_mismatch_refuses_load(monkeypatch):
    """`clip_sha`: CLIP vision tải về khác bản lúc train ⇒ nạp strict báo lỗi; vân tay text tương tự; hàng text còn 0 thì nhận của ckpt."""
    import ce_localization.models.clip_refiner as cr
    a = _policy()
    a.set_text_fingerprint(FAKE_TEXT_SHA)
    sd = a.state_dict()
    monkeypatch.setattr(cr, "load_clip", lambda name, text_name: _fake_load_clip(name, seed=1))
    b = _policy()
    with pytest.raises(RuntimeError, match="vân tay CLIP ViT"):
        b.load_state_dict(sd)
    monkeypatch.setattr(cr, "load_clip", lambda name, text_name: _fake_load_clip(name, seed=0))
    c = _policy()                                                    # hàng text = 0 -> nhận của checkpoint
    c.load_state_dict(sd)
    assert torch.equal(c.clip_sha, a.clip_sha)
    c.check_text_fingerprint(FAKE_TEXT_SHA)
    with pytest.raises(RuntimeError, match="CLIP text"):
        c.check_text_fingerprint(b"\x01" * 32)
    d = _policy()
    d.set_text_fingerprint(b"\x02" * 32)
    with pytest.raises(RuntimeError, match="vân tay CLIP text"):
        d.load_state_dict(sd)


# ----------------------------------------------------------------------------- BoxPolicy + refiner

def test_box_policy_refiner_all_params_grad_and_aux():
    """Mọi tham số train nhận grad (DDP không find_unused); loss = ε + 0,1·aux; aux 0 thì refiner vẫn có grad (refined không detach)."""
    m = _policy().train()
    x, text, vhw, x0, tok = _batch()
    for blk in m.refiner.stages[0]:                                  # khác identity để grad lan ngược đủ
        nn.init.normal_(blk.ada[-1].weight, std=0.02)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=0.02)
    loss = m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), text_tokens=tok)
    st = m.pop_stats()
    assert set(st) == {"loss_eps", "loss_aux", "delta_abs", "aux_ratio"} and float(st["delta_abs"]) > 0
    assert torch.allclose(loss, st["loss_eps"] + 0.1 * st["loss_aux"], atol=1e-6)
    loss.backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []
    assert float(m.refiner.heads[0][-1].weight.grad.abs().sum()) > 0
    m.zero_grad()
    m.refiner_aux_weight = 0.0
    loss0 = m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), text_tokens=tok)
    assert torch.allclose(loss0, st["loss_eps"], atol=1e-6)
    loss0.backward()
    assert float(m.refiner.stages[0][0].qkv.weight.grad.abs().sum()) > 0
    with pytest.raises(ValueError, match="text_tokens"):
        m(x, text, vhw, x0)


def test_box_policy_refiner_at_init_all_params_in_graph():
    """Khởi tạo y tác giả (head / AdaLN = 0): mọi tham số train vẫn có grad (tensor 0) — DDP find_unused=False chạy được từ iter 0."""
    m = _policy().train()
    x, text, vhw, x0, tok = _batch()
    m(x, text, vhw, x0, k=1, text_tokens=tok).backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []


def test_no_condition_freezes_and_bypasses(monkeypatch):
    """`use_condition: false` (DELTA1.1): không dựng refiner / không nạp CLIP; vision + text_proj đóng băng; mọi tham số train có grad;
    U-Net vào [x_t ; x_t]; box KHÔNG phụ thuộc ảnh / text."""
    import ce_localization.models.clip_refiner as cr
    monkeypatch.setattr(cr, "load_clip", lambda *a: (_ for _ in ()).throw(AssertionError("không được nạp CLIP")))
    m = _policy(use_condition=False).train()
    assert m.refiner is None and not m.needs_text_tokens and "clip_sha" not in m.state_dict()
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert frozen == {n for n, _ in m.named_parameters() if n.startswith(("vision.", "text_proj."))} and frozen
    x, text, vhw, x0, _ = _batch()
    m(x, text, vhw, x0, k=2).backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []
    m.eval()
    with torch.no_grad():
        a = m.sample(x, text, vhw, 3, generator=torch.Generator().manual_seed(1), sampler="ddpm")
        b = m.sample(torch.rand_like(x), torch.randn_like(text), vhw, 3, generator=torch.Generator().manual_seed(1), sampler="ddpm")
    assert torch.equal(a, b)


def test_refiner_fp32_under_autocast():
    """AMP (CPU: bf16 thay fp16): ResNet / U-Net theo autocast, refiner (token, transformer, head) và aux fp32."""
    m = _policy().train()
    x, text, vhw, x0, tok = _batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cond = m.condition(x, text, vhw)
        ctx = m.refiner_context(x, tok)
        assert ctx["image_tokens"].dtype == torch.float32 and ctx["text_tokens"].dtype == torch.float32
        eps, refined = m.predict_noise(torch.randn(2, 4), torch.tensor([3, 9]), cond, ctx)
        assert refined.dtype == torch.float32 and eps.dtype == torch.bfloat16
        loss = m(x, text, vhw, x0, k=1, text_tokens=tok)
    assert torch.isfinite(loss) and m.pop_stats()["loss_aux"].dtype == torch.float32
    loss.backward()
    assert m.refiner.heads[0][-1].weight.grad.dtype == torch.float32


def test_text_tokens_causal_padding_equivalence():
    """CLIP text causal + đệm phải: token thật của một tên tính riêng == tính trong batch đệm (cách tác giả) ⇒ bảng tính sẵn đúng.
    `TextTable.tokens` đệm 0 bên phải + mask."""
    from transformers import CLIPTextConfig, CLIPTextModel
    from ce_localization.models.text import TextTable
    torch.manual_seed(0)
    m = CLIPTextModel(CLIPTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                     vocab_size=100, max_position_embeddings=16)).eval()
    seqs = [[1, 5, 7, 2], [1, 9, 2], [1, 3, 4, 6, 8, 2]]
    L = max(map(len, seqs))
    ids = torch.zeros(3, L, dtype=torch.long)
    am = torch.zeros(3, L, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)], am[i, :len(s)] = torch.tensor(s), 1
    with torch.no_grad():
        batch = m(input_ids=ids, attention_mask=am).last_hidden_state
        for i, s in enumerate(seqs):
            one = m(input_ids=torch.tensor([s]), attention_mask=torch.ones(1, len(s), dtype=torch.long)).last_hidden_state[0]
            assert torch.allclose(one, batch[i, :len(s)], atol=1e-5)
    tt = TextTable({"a": torch.zeros(512), "bb": torch.zeros(512)}, {"a": torch.ones(2, 8), "bb": torch.full((4, 8), 2.0)}, b"x" * 32)
    pad, mask = tt.tokens(["bb", "a", "a"], "cpu")
    assert pad.shape == (3, 4, 8) and mask.tolist() == [[True] * 4, [True, True, False, False], [True, True, False, False]]
    assert float(pad[1, 2:].abs().sum()) == 0 and torch.equal(pad[0], torch.full((4, 8), 2.0))
    with pytest.raises(ValueError):
        TextTable({"a": torch.zeros(512)}).tokens(["a"], "cpu")


def test_sample_shared_context_equals_expanded():
    """`sample_from_cond` (ngữ cảnh một lần / ảnh) == ngữ cảnh nhân sẵn cho từng mẫu; vòng mock gọi ε_θ đúng 100 lần."""
    m = _policy(num_timesteps=100).eval()
    for blk in m.refiner.stages[0]:
        nn.init.normal_(blk.ada[-1].weight, std=0.05)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=0.05)
    x, text, vhw, _, tok = _batch()
    calls = []
    h = m.noise_net.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad():
        cond = m.condition(x, text, vhw)
        ctx = m.refiner_context(x, tok)
        a = m.sample_from_cond(cond, 3, torch.Generator().manual_seed(5), "mock", ctx=ctx)
        assert len(calls) == 100
        ci = torch.arange(2).repeat_interleave(3)
        ctx_e, cond_e = m.refiner.index_context(ctx, ci), cond.repeat_interleave(3, 0)
        from ce_localization.models.box_policy import mock_sample
        b = mock_sample(lambda xx, t: m.predict_noise(xx, t, cond_e, ctx_e)[0], 6, m.alphas_cumprod,
                        generator=torch.Generator().manual_seed(5))
    h.remove()
    assert torch.allclose(a.view(6, 4), b, atol=1e-5)


def test_sample_record_refined():
    """`record` (tool soi refiner vs U-Net): quỹ đạo thêm `refined` của từng bước = refiner trên đúng x_t / t của bước đó; box ra
    TRÙNG TỪNG BIT bản không ghi; không refiner / bỏ refiner thì không có khoá `refined`."""
    m = _policy(num_timesteps=100).eval()
    for blk in m.refiner.stages[0]:
        nn.init.normal_(blk.ada[-1].weight, std=0.05)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=0.05)
    x, text, vhw, _, tok = _batch()
    with torch.no_grad():
        cond, ctx = m.condition(x, text, vhw), m.refiner_context(x, tok)
        plain = m.sample_from_cond(cond, 3, torch.Generator().manual_seed(5), "mock", ctx=ctx)
        out, traj = m.sample_from_cond(cond, 3, torch.Generator().manual_seed(5), "mock", record="all", ctx=ctx)
        assert torch.equal(plain, out) and [s["t"] for s in traj] == list(range(99, -1, -1))
        assert all(s["refined"].shape == (2, 3, 4) for s in traj)
        ci = torch.arange(2).repeat_interleave(3)
        for s in (traj[0], traj[50], traj[-1]):
            tb = torch.full((6,), s["t"], dtype=torch.long)
            _, ref = m.predict_noise(s["x_t"].view(6, 4), tb, cond.repeat_interleave(3, 0), ctx, ci)
            assert torch.allclose(s["refined"].view(6, 4), ref, atol=1e-6)
        assert not torch.allclose(traj[0]["refined"], traj[0]["x0_hat"])        # refiner ≠ x̂0 của U-Net (hai box khác nhau)
        _, t2 = m.sample_from_cond(cond, 3, torch.Generator().manual_seed(5), "mock", record={99, 0})
        assert [s["t"] for s in t2] == [99, 0] and all("refined" not in s for s in t2)   # bỏ refiner (ctx None)


def test_mock_default_unchanged_without_refiner():
    """Hồi quy: CE-Loc không refiner, vòng mock 100 bước trùng TỪNG BIT cách tính cũ (U-Net gọi thẳng)."""
    from ce_localization.models.box_policy import BoxPolicy, mock_sample
    torch.manual_seed(0)
    m = BoxPolicy(in_channels=4, pretrained_backbone=False, num_timesteps=200, vision="r18_paper", ss_mask=True).eval()
    x, text, vhw, _, _ = _batch()
    with torch.no_grad():
        cond = m.condition(x, text, vhw)
        a = m.sample_from_cond(cond, 3, torch.Generator().manual_seed(2), "mock")
        c = cond.repeat_interleave(3, 0)
        b = mock_sample(lambda xx, t: m.noise_net(xx.unsqueeze(1), t, c).squeeze(1), 6, m.alphas_cumprod,
                        generator=torch.Generator().manual_seed(2))
    assert torch.equal(a.view(6, 4), b)


def test_mock_sample_steps():
    from ce_localization.models.box_policy import mock_sample
    ab = torch.linspace(0.99, 0.01, 300)
    seen = []
    x0 = torch.randn(5, 4, generator=torch.Generator().manual_seed(0))
    out = mock_sample(lambda x, t: (seen.append(int(t[0])), torch.ones_like(x))[1], 5, ab, steps=200,
                      generator=torch.Generator().manual_seed(0))
    assert seen == list(range(199, -1, -1)) and torch.allclose(out, x0 - 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        mock_sample(lambda x, t: x, 2, ab, steps=301)


def test_refiner_rejects_bad_combos():
    from ce_localization.models.box_policy import BoxPolicy
    _policy(obj_attn=True)                                                     # DELTA4: refiner (concat) + box vật hợp lệ
    with pytest.raises(ValueError, match="obj_attn"):
        _policy(refiner={**AUTHOR_REFINER, "unet_input": "none"}, obj_attn=True)
    with pytest.raises(ValueError, match="num_stages"):
        _policy(refiner={**POST, "num_stages": 2})
    with pytest.raises(ValueError, match="use_condition"):
        _policy(refiner=POST, use_condition=False)
    with pytest.raises(ValueError, match="use_condition"):
        _policy(refiner=MID, use_condition=False)
    with pytest.raises(ValueError, match="mid_aux_weight"):
        _policy(refiner={**AUTHOR_REFINER, "mid_aux_weight": 0.1})
    with pytest.raises(ValueError, match="4 kênh"):
        BoxPolicy(in_channels=3, pretrained_backbone=False, num_timesteps=20, vision="r18_paper", refiner=AUTHOR_REFINER)
    with pytest.raises(ValueError, match="unet_input"):
        _policy(refiner={**AUTHOR_REFINER, "unet_input": "replace"})
    with pytest.raises(ValueError, match="use_condition"):
        _policy(refiner={**AUTHOR_REFINER, "unet_input": "none"}, use_condition=False)
    with pytest.raises(ValueError, match="attention_flow"):                    # ảnh nhìn box: không có đường tách đôi
        _refiner({"attention_flow": {"sample": ["sample", "image", "text"], "image": ["image", "text", "sample"],
                                     "text": ["text"]}})


# ----------------------------------------------------------------------------- DELTA2.x / DELTA3

VITS = {"delta2_1": "openai/clip-vit-base-patch32", "delta2_2": "openai/clip-vit-large-patch14",
        "delta2_3": "openai/clip-vit-large-patch14-336"}
B16 = "openai/clip-vit-base-patch16"
NO_UNET = ("vision", "ss_mask", "pretrained_backbone", "vis_dim", "text_dim", "text_proj_dim", "step_embed_dim", "down_dims",
           "kernel_size", "n_groups")


@pytest.mark.parametrize("kind", list(VITS))
def test_delta2_vit_grid_and_text_b16(monkeypatch, kind):
    """DELTA2.x: ViT của refiner theo `clip_model_name` (lưới 7 / 16 / 24, hidden khác), token chữ vẫn từ CLIP text B/16
    (`text_model_name`) — `load_clip` nhận đúng hai tên; ngữ cảnh ảnh đủ g² token; mọi tham số train nhận grad."""
    import ce_localization.models.clip_refiner as cr
    seen = []
    monkeypatch.setattr(cr, "load_clip", lambda name, text_name: (seen.append((name, text_name)), _fake_load_clip(name))[1])
    rcfg = yaml.safe_load(open(CFG_G[kind]))["model"]["refiner"]
    m = _policy(refiner=rcfg).train()
    size, patch, hid = FAKE_VIT[VITS[kind]]
    g = size // patch
    assert seen == [(VITS[kind], B16)] and m.refiner.grid == g and m.refiner.clip_res == size
    assert m.refiner.image_proj[1].in_features == hid and cr.text_model_name(cr.refiner_config(rcfg)) == B16
    x, text, vhw, x0, tok = _batch()
    assert m.refiner_context(x, tok)["image_tokens"].shape == (2, g * g, 256)
    m(x, text, vhw, x0, k=2, text_tokens=tok).backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []


def test_text_model_name_defaults_to_clip_model_name():
    """Config / checkpoint DELTA1 (không có `text_model_name`): token chữ từ cùng tên `clip_model_name` như tác giả."""
    from ce_localization.models.clip_refiner import refiner_config, text_model_name
    assert text_model_name(refiner_config(AUTHOR_REFINER)) == B16
    assert refiner_config({"enabled": False}) is None and refiner_config(None) is None


def _delta3(seed=0, num_timesteps=20):
    rcfg = yaml.safe_load(open(CFG_G["delta3"]))["model"]["refiner"]
    from ce_localization.models.box_policy import BoxPolicy
    torch.manual_seed(seed)
    m = BoxPolicy(in_channels=4, num_timesteps=num_timesteps, refiner=rcfg)
    for blk in m.refiner.stages[0]:                                  # khác identity
        nn.init.normal_(blk.ada[-1].weight, std=0.05)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=0.05)
    return m


def test_delta3_refiner_only_structure_and_loss():
    """DELTA3: chỉ refiner (không U-Net / ResNet / FiLM, cond [B, 0]); loss = 1,0 · MSE(refined, x0), không có ε; mọi tham số train
    nhận grad (DDP không find_unused)."""
    m = _delta3().train()
    assert m.noise_net is None and m.cond_dim == 0 and not hasattr(m, "vision") and not hasattr(m, "text_proj")
    assert {n.split(".")[0] for n, _ in m.named_parameters()} == {"refiner"}
    x, text, vhw, x0, tok = _batch()
    assert m.condition(x, text, vhw).shape == (2, 0)
    loss = m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), text_tokens=tok)
    st = m.pop_stats()
    assert set(st) == {"loss_aux", "delta_abs", "aux_ratio"} and torch.allclose(loss, st["loss_aux"], atol=1e-7)
    loss.backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []


def test_delta3_eps_from_refined_and_mock_lands_on_refined():
    """ε̂ = (x_t − √ᾱ_t·refined) / √(1 − ᾱ_t) với t theo hàng; vòng mock: bước cuối (t = 0) rơi đúng √ᾱ₀·refined; bỏ refiner ⇒ lỗi rõ."""
    m = _delta3(num_timesteps=100).eval()
    x_img, text, vhw, _, tok = _batch()
    with torch.no_grad():
        cond = m.condition(x_img, text, vhw)
        ctx = m.refiner_context(x_img, tok)
        x, t = torch.randn(4, 4), torch.tensor([0, 7, 50, 99])
        ci = torch.tensor([0, 0, 1, 1])
        eps, refined = m.predict_noise(x, t, cond[ci], ctx, ci)
        ab = m.alphas_cumprod[t].unsqueeze(-1)
        assert torch.allclose(eps, (x - ab.sqrt() * refined) / (1 - ab).sqrt(), rtol=1e-4, atol=1e-3)   # t = 0: chia 0,01
        z = torch.randn(2, 4)
        e0, r0 = m.predict_noise(z, torch.zeros(2, dtype=torch.long), cond, ctx)
        assert torch.allclose(z - e0 / 100, m.alphas_cumprod[0].sqrt() * r0, atol=1e-4)
        with pytest.raises(ValueError, match="none"):
            m.sample(x_img, text, vhw, 2, sampler="mock", text_tokens=tok, use_refiner=False)


def test_delta2_delta3_configs():
    """delta2_x = delta1 chỉ khác `refiner.clip_model_name` + thêm `text_model_name` B/16; delta3 = delta1 bỏ khoá ResNet / FiLM / U-Net,
    `unet_input: none`, `aux_loss_weight: 1.0` — data / diffusion / training / eval y hệt."""
    load = lambda k: yaml.safe_load(open(CFG_G[k]))  # noqa: E731
    d1 = load("delta1")
    for kind, vit in VITS.items():
        d = load(kind)
        r = d["model"]["refiner"]
        assert r.pop("clip_model_name") == vit and r.pop("text_model_name") == B16
        ref = load("delta1")
        ref["model"]["refiner"].pop("clip_model_name")
        for x in (d, ref):
            x.pop("experiment"), x.pop("description")
        assert d == ref
    d3 = load("delta3")
    r = d3["model"].pop("refiner")
    assert r.pop("unet_input") == "none" and r.pop("aux_loss_weight") == 1.0
    r1 = d1["model"].pop("refiner")
    r1.pop("unet_input"), r1.pop("aux_loss_weight")
    assert r == r1
    for k in NO_UNET:
        d1["model"].pop(k)
    for x in (d1, d3):
        x.pop("experiment"), x.pop("description")
    assert d1 == d3


# ----------------------------------------------------------------------------- DELTA4 / DELTA3.x (mục 10)

POST = {**AUTHOR_REFINER, "unet_input": "post", "aux_loss_weight": 1.0}
MID = {**AUTHOR_REFINER, "unet_input": "mid", "mid_aux_weight": 0.1}


def _off_identity(m, std=0.05):
    for blk in m.refiner.stages[0]:
        nn.init.normal_(blk.ada[-1].weight, std=std)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=std)
    return m


OBJS = [torch.tensor([[10.0, 10.0, 30.0, 40.0], [50.0, 20.0, 70.0, 60.0], [80.0, 80.0, 100.0, 110.0]]),
        torch.tensor([[5.0, 5.0, 25.0, 25.0]])]


def test_delta4_objects_mirror_encoded_as_pair():
    """DELTA4: refiner trước U-Net (8 kênh) + box vật: token vật = `encode_objects` trên [o ; o]; box vật trả về giữ 4 số (cho
    `nearest_mask`, GAMMA4.0.1 vẫn chạy); mọi tham số train nhận grad (DDP không find_unused)."""
    m = _off_identity(_policy(obj_attn=True)).train()
    x, text, vhw, x0, tok = _batch()
    with torch.no_grad():
        cond = m.condition(x, text, vhw)
        toks, mask, units = m.object_tokens(OBJS, 128, cond)
        ref = m.noise_net.encode_objects(torch.cat([units, units], -1), mask, cond)
    assert m.unet_in_dim == 8 and units.shape[-1] == 4 and all(torch.equal(a, b) for a, b in zip(toks, ref))
    m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), objects=OBJS, text_tokens=tok).backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []
    m.eval().obj_k = 1                                                         # chọn vật gần nhất theo box 4 số
    with torch.no_grad():
        assert m.sample(x, text, vhw, 2, sampler="mock" if m.num_timesteps >= 100 else "ddpm", objects=OBJS,
                        text_tokens=tok).shape == (2, 2, 4)


@pytest.mark.parametrize("obj_attn", [False, True])
def test_post_refiner_after_unet(obj_attn):
    """DELTA3.1 / 3.2 (`unet_input: post`): U-Net 4 kênh; lúc khởi tạo ε̂ == ε̂₁ TỪNG BIT; bỏ refiner == U-Net trần; Δε không chảy grad
    về U-Net qua x̂0₁; loss = MSE(ε̂, ε) + 1·MSE(ε̂₁, ε), stats ε / ε₁ / |Δε| / ε/ε₁; mọi tham số train nhận grad; DELTA3.2 mã hoá
    box vật 4 số như GAMMA4."""
    m = _policy(refiner=POST, obj_attn=obj_attn).train()
    assert m.unet_in_dim == 4 and m.noise_net.final_conv[1].out_channels == 4
    x, text, vhw, x0, tok = _batch()
    obj = None
    with torch.no_grad():
        cond, ctx = m.condition(x, text, vhw), m.refiner_context(x, tok)
        if obj_attn:
            obj = m.object_tokens(OBJS, 128, cond)
            assert obj[0][0].shape[-1] == m.noise_net.obj_xattn[0].ln_q.normalized_shape[0]
        xt, t = torch.randn(2, 4), torch.tensor([3, 17])
        eps, refined, eps1 = m._denoise(xt, t, cond, ctx, None, obj)
        assert refined is None and torch.equal(eps, eps1.float())              # head khởi tạo 0 ⇒ Δε = 0
        e_nr, _ = m.predict_noise(xt, t, cond, None, None, obj)
        direct = m.noise_net(xt.unsqueeze(1), t, cond, *((obj[0], obj[1]) if obj else (None, None))).squeeze(1)
        assert torch.equal(e_nr, direct)
    _off_identity(m)
    eps, _, eps1 = m._denoise(xt, t, cond, ctx, None, obj)
    unet = [p for p in m.noise_net.parameters() if p.requires_grad]
    g = torch.autograd.grad((eps - eps1.float()).sum(), unet, allow_unused=True)
    assert all(gi is None or float(gi.abs().sum()) == 0 for gi in g)          # x̂0₁ detach
    kw = {"objects": OBJS} if obj_attn else {}
    loss = m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), text_tokens=tok, **kw)
    st = m.pop_stats()
    assert set(st) == {"loss_eps", "loss_eps1", "delta_abs", "eps_ratio"} and float(st["delta_abs"]) > 0
    assert torch.allclose(loss, st["loss_eps"] + st["loss_eps1"], atol=1e-6)
    loss.backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []


@pytest.mark.parametrize("obj_attn", [False, True])
def test_mid_refiner_inside_unet(obj_attn):
    """DELTA4.1 (`unet_input: mid`): U-Net 4 kênh; lúc khởi tạo b = x_t, refined = b, ε̂ == U-Net không refiner == bản không refiner cùng
    seed (GAMMA4) TỪNG BIT; nhánh chèn gọi ĐÚNG MỘT lần, ở vị trí mid (d 256), sau cross-attn box vật; box vật không qua nhánh chèn; bỏ
    refiner == U-Net trần; loss = ε + 0,1·MSE(refined, x0) + 0,1·MSE(b, x0); mọi tham số train nhận grad (kể cả khi Linear khởi tạo 0)."""
    m = _policy(refiner=MID, obj_attn=obj_attn).train()
    ref = _policy(refiner=None, obj_attn=obj_attn).train()                    # GAMMA4 / gamma2_celoc cùng seed
    assert m.unet_in_dim == 4 and m.noise_net.mid_site == 3 and m.noise_net.mid_dim == 256
    assert m.mid_in.weight.shape == (4, 256) and m.mid_out.weight.shape == (256, 4)
    sd = ref.state_dict()
    assert all(torch.equal(v, sd[k]) for k, v in m.state_dict().items() if k in sd) and set(sd) <= set(m.state_dict())
    x, text, vhw, x0, tok = _batch()
    xt, t = torch.randn(2, 4), torch.tensor([3, 17])
    with torch.no_grad():
        cond, ctx = m.condition(x, text, vhw), m.refiner_context(x, tok)
        obj = m.object_tokens(OBJS, 128, cond) if obj_attn else None
        eps, refined, b = m._denoise(xt, t, cond, ctx, None, obj)
        assert torch.equal(b, xt) and torch.equal(refined, b)
        assert torch.equal(eps, ref.predict_noise(xt, t, cond, None, None, obj)[0])
        assert torch.equal(m.predict_noise(xt, t, cond, None, None, obj)[0], eps)          # bỏ refiner
    _off_identity(m)
    for lin in (m.mid_in, m.mid_out):
        nn.init.normal_(lin.weight, std=0.05)
    calls = []
    orig = m.refiner.forward
    m.refiner.forward = lambda bx, *a: calls.append(bx.shape) or orig(bx, *a)  # noqa: E731
    seen = []
    if obj_attn:
        m.noise_net.obj_xattn[3].register_forward_hook(lambda mod, a, out: seen.append("xattn"))
    m.mid_out.register_forward_pre_hook(lambda mod, a: seen.append("mid"))
    with torch.no_grad():
        eps2, refined2, b2 = m._denoise(xt, t, cond, ctx, None, obj)
        e_nr = m.predict_noise(xt, t, cond, None, None, obj)[0]
        direct = m.noise_net(xt.unsqueeze(1), t, cond, *((obj[0], obj[1]) if obj else (None, None))).squeeze(1)
        assert torch.equal(e_nr, direct) and not torch.equal(eps2, direct) and not torch.equal(b2, xt)
        assert not torch.equal(refined2, b2)
        if obj_attn:                                                           # box vật không qua nhánh chèn
            n0 = len(calls)
            again = m.object_tokens(OBJS, 128, cond)
            assert len(calls) == n0 and all(torch.equal(a, c) for a, c in zip(again[0], obj[0]))
    assert len(calls) == 1 and seen[:2] == (["xattn", "mid"] if obj_attn else ["mid"])       # chỉ lần có ctx
    m.refiner.forward = orig
    kw = {"objects": OBJS} if obj_attn else {}
    loss = m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), text_tokens=tok, **kw)
    st = m.pop_stats()
    assert set(st) == {"loss_eps", "loss_aux", "loss_mid", "delta_abs", "aux_ratio", "mid_ratio"}
    assert torch.allclose(loss, st["loss_eps"] + 0.1 * st["loss_aux"] + 0.1 * st["loss_mid"], atol=1e-6)
    assert torch.allclose(st["mid_ratio"], st["loss_aux"] / st["loss_mid"])
    loss.backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []
    m0 = _policy(refiner=MID, obj_attn=obj_attn).train()                       # khởi tạo thật (mọi khối zero): vẫn đủ grad
    m0(x, text, vhw, x0, k=1, text_tokens=tok, **kw).backward()
    assert [n for n, p in m0.named_parameters() if p.requires_grad and p.grad is None] == []
    assert float(m0.mid_in.weight.grad.abs().sum()) > 0 and float(m0.mid_out.weight.grad.abs().sum()) > 0


def test_delta4_delta3x_configs():
    """delta4 = gamma4 + `use_condition: true` + khối refiner của delta2_1; delta3_1 / 3_2 = delta2_1 / delta4 chỉ khác
    `refiner.unet_input: post` + `aux_loss_weight: 1.0`; delta4_1 = delta4 chỉ khác `unet_input: mid` + `mid_aux_weight: 0.1`."""
    load = lambda k: yaml.safe_load(open(CFG_G[k]))  # noqa: E731
    g4, d4, d21 = load("obj"), load("delta4"), load("delta2_1")
    assert d4["model"].pop("refiner") == d21["model"]["refiner"] and d4["model"].pop("use_condition") is True
    for x in (g4, d4):
        x.pop("experiment"), x.pop("description")
    assert d4 == g4
    for kind, base in (("delta3_1", "delta2_1"), ("delta3_2", "delta4"), ("delta4_1", "delta4")):
        d, b = load(kind), load(base)
        r, rb = d["model"]["refiner"], b["model"]["refiner"]
        if kind == "delta4_1":
            assert r.pop("unet_input") == "mid" and r.pop("mid_aux_weight") == 0.1 and "mid_aux_weight" not in rb
            rb.pop("unet_input")
        else:
            assert r.pop("unet_input") == "post" and r.pop("aux_loss_weight") == 1.0
            rb.pop("unet_input"), rb.pop("aux_loss_weight")
        for x in (d, b):
            x.pop("experiment"), x.pop("description")
        assert d == b


# ----------------------------------------------------------------------------- DELTA5.x: bản log-scale của tác giả

class _AuthorBoxCoder:
    """`BoxCoder` nguyên văn của `refs/CE-Loc-logScale/utils/normalization.py` (chỉ phần encode / decode)."""
    def __init__(self, cfg=None):
        import math
        cfg = cfg or {}
        self.type = cfg.get('type', 'linear')
        self.min_size_ratio = cfg.get('min_size_ratio', 1 / 128)
        self.log_min = math.log(self.min_size_ratio)

    def encode(self, box, canvas_size):
        cw, ch = float(canvas_size), float(canvas_size)
        scale = box.new_tensor([cw, ch, cw, ch])
        rel = box / scale
        out = rel * 2 - 1
        if self.type == 'log':
            size = rel[..., 2:].clamp(min=self.min_size_ratio).log()
            out = torch.cat([out[..., :2], (size - self.log_min) / -self.log_min * 2 - 1], dim=-1)
        return out

    def decode(self, box, canvas_size=1.0):
        cw, ch = float(canvas_size), float(canvas_size)
        scale = box.new_tensor([cw, ch, cw, ch])
        rel = (box + 1) / 2
        if self.type == 'log':
            size = ((box[..., 2:].clamp(-5, 5) + 1) / 2 * -self.log_min + self.log_min).exp()
            rel = torch.cat([rel[..., :2], size], dim=-1)
        return rel * scale


def _author_cxcywh_to_xyxy(box):
    cx, cy = box[..., 0], box[..., 1]
    w, h = box[..., 2].clamp(min=0), box[..., 3].clamp(min=0)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _author_giou(box1, box2):
    a, b = _author_cxcywh_to_xyxy(box1), _author_cxcywh_to_xyxy(box2)
    iw = (torch.minimum(a[..., 2], b[..., 2]) - torch.maximum(a[..., 0], b[..., 0])).clamp(min=0)
    ih = (torch.minimum(a[..., 3], b[..., 3]) - torch.maximum(a[..., 1], b[..., 1])).clamp(min=0)
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    iou = inter / (union + 1e-6)
    ew = torch.maximum(a[..., 2], b[..., 2]) - torch.minimum(a[..., 0], b[..., 0])
    eh = torch.maximum(a[..., 3], b[..., 3]) - torch.minimum(a[..., 1], b[..., 1])
    enclose = ew * eh
    return iou - (enclose - union) / (enclose + 1e-6)


def _author_sample(predict, alphas_cumprod, num_timesteps, B, device, num_inference_steps=None, eta=1.0, clip_sample=True,
                   generator=None):
    """Vòng `ObjectPlacementPolicy.sample` nguyên văn (predict(box, t_batch) thay `self.predict_noise(box, t_batch, cond)[0]`)."""
    steps = num_inference_steps or num_timesteps
    timesteps = torch.linspace(num_timesteps - 1, 0, steps, device=device).round().long().unique_consecutive()
    box = torch.randn((B, 4), device=device, generator=generator)
    for i, t in enumerate(timesteps):
        t_batch = t.expand(B)
        eps = predict(box, t_batch)
        ab_t = alphas_cumprod[t]
        ab_prev = alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else box.new_tensor(1.0)
        x0 = (box - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
        if clip_sample:
            x0 = x0.clamp(-1, 1)
            eps = (box - ab_t.sqrt() * x0) / (1 - ab_t).sqrt()
        sigma = eta * ((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev)).sqrt()
        box = ab_prev.sqrt() * x0 + (1 - ab_prev - sigma ** 2).clamp(min=0).sqrt() * eps
        if i + 1 < len(timesteps):
            box = box + sigma * torch.randn(box.shape, device=device, generator=generator)
    return box


LOG = {"type": "log", "min_size_ratio": 1 / 128}


def test_box_coder_matches_author():
    """`boxes_to_unit` / `unit_to_boxes` với coder log == `BoxCoder` của tác giả; ví dụ README (5 / 10 / 50 % canvas -> −0,23 / 0,05 /
    0,71); giải mã luôn cạnh > 0 kể cả nhiễu lớn; linear (None / dict) trùng hàm cũ TỪNG BIT, `to_linear_unit` trả chính tensor."""
    from ce_localization.models.box_policy import boxes_to_unit, unit_to_boxes
    from ce_localization.utils.box_coder import box_coder_config, to_linear_unit
    g = torch.Generator().manual_seed(0)
    cxcy = torch.rand(50, 2, generator=g) * 512
    wh = torch.exp(torch.empty(50, 2).uniform_(np.log(2), np.log(500), generator=g))           # 2 px .. 500 px (có cạnh < 4 px)
    cxcywh = torch.cat([cxcy, wh], -1).double()
    xyxy = torch.cat([cxcywh[:, :2] - cxcywh[:, 2:] / 2, cxcywh[:, :2] + cxcywh[:, 2:] / 2], -1)
    canvas = torch.full((4,), 512.0, dtype=torch.float64)
    for cfg in (LOG, {"type": "linear"}):
        coder, ref = box_coder_config(cfg), _AuthorBoxCoder(cfg)
        u = boxes_to_unit(xyxy, canvas, coder)
        assert torch.allclose(u, ref.encode(cxcywh, 512), atol=1e-12)
        noisy = u + torch.randn(u.shape, generator=g, dtype=torch.float64) * 3
        assert torch.allclose(unit_to_boxes(noisy, canvas, coder), _author_cxcywh_to_xyxy(ref.decode(noisy, 512)) if cfg is LOG
                              else unit_to_boxes(noisy, canvas), atol=1e-9)
    ex = torch.tensor([[256.0, 256.0, 0.05 * 512, 0.10 * 512], [256.0, 256.0, 0.5 * 512, 0.5 * 512]], dtype=torch.float64)
    u = boxes_to_unit(torch.cat([ex[:, :2] - ex[:, 2:] / 2, ex[:, :2] + ex[:, 2:] / 2], -1), canvas, box_coder_config(LOG))
    assert np.allclose(u[:, 2:].numpy(), [[-0.23, 0.05], [0.71, 0.71]], atol=0.01)
    from ce_localization.utils.box_coder import unit_fraction
    assert (unit_fraction(torch.randn(1000, 4, generator=g) * 10, box_coder_config(LOG))[:, 2:] > 0).all()   # cạnh luôn > 0
    lin = box_coder_config(None)
    assert torch.equal(boxes_to_unit(xyxy, canvas, lin), boxes_to_unit(xyxy, canvas)) and to_linear_unit(u, lin) is u
    with pytest.raises(ValueError, match="box_coder"):
        box_coder_config({"type": "sqrt"})


def test_refiner_roialign_decodes_log_box():
    """Refiner với coder log: RoIAlign trên box log == refiner linear trên box đã giải mã (y `BoxCoder.decode(box, g)` của tác giả);
    embedding Fourier vẫn đọc box mã hoá (tác giả không đổi chỗ này)."""
    from ce_localization.utils.box_coder import box_coder_config, to_linear_unit
    r = _refiner(dtype=torch.float64)
    ctx = _ctx(r)
    g = torch.Generator().manual_seed(3)
    box = torch.rand(2, 4, generator=g, dtype=torch.float64) * 2 - 1
    tok_lin, pos_lin = r._sample_tokens(to_linear_unit(box, box_coder_config(LOG)), ctx["image_map"])
    r.box_coder = box_coder_config(LOG)
    tok_log, pos_log = r._sample_tokens(box, ctx["image_map"])
    assert torch.equal(tok_lin, tok_log) and torch.equal(pos_lin, pos_log)
    grid = _AuthorBoxCoder(LOG).decode(box, r.grid)
    x1 = grid[:, 0].clamp(0, r.grid) - grid[:, 2].clamp(0.5, r.grid) / 2
    assert torch.allclose(pos_log[:, 0, 0], x1 + (grid[:, 2].clamp(0.5, r.grid) / r.roi_size) / 2, atol=1e-9)


@pytest.mark.parametrize("steps,eta,clip", [(None, 1.0, True), (7, 1.0, True), (7, 0.0, True), (5, 0.5, False)])
def test_ddim_sample_matches_author(steps, eta, clip):
    """`ddim_sample` == vòng `sample` của tác giả (cùng generator): đủ T bước / cách đều, eta 1 / 0 / 0,5, có / không kẹp x̂0."""
    from ce_localization.models.box_policy import ddim_sample
    from ce_localization.utils.diffusion_math import linear_alphas_cumprod
    ab = linear_alphas_cumprod(20)
    W = torch.randn(4, 4, generator=torch.Generator().manual_seed(1))

    def fn(x, t):
        return torch.tanh(x @ W) * (1 + t[:, None].float() / 20)
    ours = ddim_sample(fn, 6, ab, ddim_steps=steps, eta=eta, clip_sample=clip, generator=torch.Generator().manual_seed(5))
    ref = _author_sample(fn, ab, 20, 6, "cpu", steps, eta, clip, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(ours, ref, atol=1e-6)
    x, traj = ddim_sample(fn, 6, ab, ddim_steps=steps, eta=eta, clip_sample=clip, generator=torch.Generator().manual_seed(5),
                          record="all")
    assert torch.equal(x, ours) and len(traj) == (steps or 20) and traj[0]["t"] == 19 and traj[-1]["t"] == 0


def test_delta5_aux_giou_and_log_targets():
    """DELTA5.x: loss = ε + 0,1·[MSE(refined, x0) + w·(1 − GIoU)], GIoU == `generalized_box_iou` của tác giả trên box giải mã;
    stats có `loss_giou`; mọi tham số có grad; box vật mã hoá cùng coder log; sampler ddim chạy; post + GIoU bị chặn."""
    m = _off_identity(_policy(refiner={**AUTHOR_REFINER, "aux_giou_weight": 0.5}, obj_attn=True, box_coder=LOG)).train()
    assert m.refiner.box_coder == m.box_coder and m.box_coder["type"] == "log"
    x, text, vhw, x0, tok = _batch()
    loss = m(x, text, vhw, x0, k=2, generator=torch.Generator().manual_seed(4), objects=OBJS, text_tokens=tok)
    st = m.pop_stats()
    assert torch.allclose(loss, st["loss_eps"] + 0.1 * (st["loss_aux"] + 0.5 * st["loss_giou"]), atol=1e-6)
    loss.backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []
    from ce_localization.models.box_policy import boxes_to_unit
    from ce_localization.utils.box_coder import generalized_box_iou_cxcywh, unit_fraction
    with torch.no_grad():
        cond = m.condition(x, text, vhw)
        _, _, units = m.object_tokens(OBJS, 128, cond)
        assert torch.allclose(units[1, :1], boxes_to_unit(OBJS[1], torch.full((4,), 128.0), m.box_coder))
        a, b = torch.rand(9, 4) * 2 - 1, torch.rand(9, 4) * 2 - 1
        dec = _AuthorBoxCoder(LOG)
        assert torch.allclose(generalized_box_iou_cxcywh(unit_fraction(a, m.box_coder), unit_fraction(b, m.box_coder)),
                              _author_giou(dec.decode(a), dec.decode(b)), atol=1e-6)
        m.eval()
        out = m.sample(x, text, vhw, 3, generator=torch.Generator().manual_seed(0), sampler="ddim", ddim_steps=5, objects=OBJS,
                       text_tokens=tok)
        assert out.shape == (2, 3, 4) and torch.isfinite(out).all()
        with pytest.raises(ValueError, match="sampler"):
            m.sample(x, text, vhw, 1, sampler="dpm", objects=OBJS, text_tokens=tok)
    with pytest.raises(ValueError, match="aux_giou_weight"):
        _policy(refiner={**POST, "aux_giou_weight": 0.1})


def test_delta5_configs():
    """delta5 = delta4 + `model.box_coder` log (1/128) + `aux_giou_weight` 0 + val bằng sampler ddim của tác giả (100 bước, eta 1);
    delta5_1 / 5_2 chỉ khác `aux_giou_weight` 0,1 / 0,5."""
    load = lambda k: yaml.safe_load(open(CFG_G[k]))  # noqa: E731
    d4, d5 = load("delta4"), load("delta5")
    assert d5["model"].pop("box_coder") == {"type": "log", "min_size_ratio": 1 / 128}
    assert d5["model"]["refiner"].pop("aux_giou_weight") == 0.0
    assert d5["eval"].pop("sample_kw") == {"sampler": "ddim", "ddim_steps": 100, "eta": 1.0} and d4["eval"].pop("sample_kw")
    for x in (d4, d5):
        x.pop("experiment"), x.pop("description")
    assert d4 == d5
    for kind, w in (("delta5_1", 0.1), ("delta5_2", 0.5)):
        d, b = load(kind), load("delta5")
        assert d["model"]["refiner"].pop("aux_giou_weight") == w and b["model"]["refiner"].pop("aux_giou_weight") == 0.0
        for x in (d, b):
            x.pop("experiment"), x.pop("description")
        assert d == b


# ----------------------------------------------------------------------------- config, eval, train trọn luồng

def test_delta_configs_only_add_refiner():
    """delta1 = gamma2_celoc + `use_condition: true` + khối `refiner` = mặc định tác giả NGUYÊN VĂN; delta1_1 = delta1 chỉ đổi
    `use_condition` ⇒ gamma2_celoc là đối chứng của DELTA1, DELTA1.1 là đối chứng "không nhìn ảnh"."""
    load = lambda k: yaml.safe_load(open(CFG_G[k]))  # noqa: E731
    a, b, c = load("celoc2"), load("delta1"), load("delta1_1")
    assert b["model"].pop("refiner") == AUTHOR_REFINER and b["model"].pop("use_condition") is True
    assert c["model"].pop("refiner") == AUTHOR_REFINER and c["model"].pop("use_condition") is False
    for d in (a, b, c):
        d.pop("experiment"), d.pop("description")
    assert a == b == c


def test_eval_add_variants_naming():
    from ce_localization.eval import add_variants
    keys = lambda v: [s for s, _, _ in v]  # noqa: E731
    assert keys(add_variants(["mock"])) == [""] and keys(add_variants(["ddpm"])) == [""]
    v = add_variants(["mock"], norefine=True)
    assert keys(v) == ["", "_norefine"] and v[1][2] == {"sampler": "mock", "sample_kw": {"use_refiner": False}}
    assert keys(add_variants(["mock", "ddpm"], norefine=True)) == ["_mock", "_ddpm", "_mock_norefine"]
    assert keys(add_variants(["mock"], obj_attn=True, norefine=True)) == ["", "_noobj", "_norefine"]     # DELTA4 / 3.2
    v = add_variants(["ddim"], obj_attn=True, norefine=True, ddim_steps=100, eta=1.0)                    # DELTA5
    assert keys(v) == ["", "_noobj", "_norefine"] and v[0][2] == {"sampler": "ddim", "sample_kw": {"ddim_steps": 100, "eta": 1.0}}
    assert v[2][2]["sample_kw"] == {"ddim_steps": 100, "eta": 1.0, "use_refiner": False}
    assert keys(add_variants(["mock", "ddim"], norefine=True)) == ["_mock", "_ddim", "_mock_norefine", "_ddim_norefine"]
    o = add_variants(["mock", "ddpm"], obj_attn=True)
    assert keys(o) == ["_mock", "_ddpm", "_mock_noobj", "_ddpm_noobj"] and o[3][2] == {"sampler": "ddpm",
                                                                                         "sample_kw": {"use_objects": False}}


def test_grad_groups_refiner():
    from ce_localization.utils.grad_monitor import group_of
    assert group_of("refiner.stages.0.1.qkv.weight") == "refiner.blocks"
    assert group_of("refiner.heads.0.3.weight") == "refiner.head"
    assert group_of("refiner.image_proj.1.weight") == "refiner.embed"
    assert group_of("noise_net.final_conv.1.weight") == "unet1d"
    assert group_of("mid_in.weight") == group_of("mid_out.bias") == "refiner.mid"


def _eval(monkeypatch, ck, out, extra=(), clip_seed=0):
    import ce_localization.eval as ea
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    _patch_clip(monkeypatch, clip_seed)
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", ck, "--split", "test", "--n-samples", "3", "--num-workers", "0",
                                      "--out", out, "--device", "cpu", *extra])
    ea.main()
    with open(out) as f:
        return json.load(f)


@pytest.mark.parametrize("kind", ["delta4", "delta3", "delta3_2", "delta4_1", "delta5_2"])   # refiner trước / không U-Net / sau / giữa / log
def test_full_flow_delta_train_resume_eval(tmp_path, monkeypatch, kind):
    """DELTA4 / 3 / 3.2 / 4.1 thu nhỏ (refiner kích thước thật, CLIP giả, T = 100 cho vòng mock) — bốn đường mã của refiner (`concat` +
    box vật, `none`, `post` + box vật, `mid` + box vật): DELTA4 train 2 + --resume tới 4 == train liền 4 (mọi tensor kể cả `clip_sha`), các
    bản kia train một lượt;
    history có thống kê refiner; eval mock 100 cho cả 2 loại ảnh (+ `_norefine` khi có U-Net, `_noobj` khi có box vật); CLIP khác bản
    lúc train ⇒ eval từ chối."""
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = _gamma2_cfg(tmp_path, base, kind)
    assert cfg["model"]["refiner"]["d_model"] == 256 and cfg["diffusion"]["num_timesteps"] == 100
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    if kind == "delta4":                                             # resume chung cho mọi config: kiểm một lần
        _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--max-iter", "2"])
        _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--resume"])
    else:
        a = b
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4 and set(ka["model"]) == set(kb["model"])
    assert "clip_sha" in kb["model"] and not any("vision_backbone" in k for k in kb["model"])
    assert bytes(kb["model"]["clip_sha"][1].tolist()) == FAKE_TEXT_SHA
    assert (kind == "delta3") == (not any(k.startswith(("noise_net.", "vision.")) for k in kb["model"]))
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    assert float(kb["model"]["refiner.heads.0.3.weight"].abs().sum()) > 0       # head rời 0 sau train
    logs = [h for h in kb["history"] if "loss" in h]
    post = kind.startswith("delta3_")
    want_log = ({"loss_eps", "loss_eps1", "delta_abs", "eps_ratio", "refiner_gates"} if post else
                {"loss_aux", "delta_abs", "aux_ratio", "refiner_gates"} | ({"loss_eps"} if kind != "delta3" else set())
                | ({"loss_mid", "mid_ratio"} if kind == "delta4_1" else set()) | ({"loss_giou"} if kind == "delta5_2" else set()))
    assert all(want_log <= set(h) for h in logs) and (kind != "delta3" or not any("loss_eps" in h for h in logs))
    assert not post or not any("loss_aux" in h for h in logs)
    assert (kind == "delta4_1") == ("mid_in.weight" in kb["model"]) and (kind != "delta4_1" or float(kb["model"]["mid_in.weight"]
                                                                                                    .abs().sum()) > 0)
    res = _eval(monkeypatch, os.path.join(b, "best.pth"), str(tmp_path / "res.json"),
                ["--add-samplers", "ddim" if kind == "delta5_2" else "mock", "--dump-boxes", str(tmp_path / "boxes.json")])
    assert kind != "delta5_2" or (res["add_samplers"] == ["ddim"] and res["ddim_steps"] == 100 and res["eta"] == 1.0)
    want = {"inpainted", "original"} | (set() if kind == "delta3" else {"inpainted_norefine", "original_norefine"})
    want |= {"inpainted_noobj", "original_noobj"} if cfg["model"].get("obj_attn") else set()
    assert set(res["results"]) == want
    for key, r in res["results"].items():
        assert r["n"] > 0 and bool(r["refine_delta_by_t"]) == ("norefine" not in key)
    with pytest.raises(RuntimeError, match="vân tay"):
        _eval(monkeypatch, os.path.join(b, "best.pth"), str(tmp_path / "y.json"), ["--add-samplers", "ddpm"], clip_seed=1)


def test_full_flow_delta1_1_train_eval(tmp_path, monkeypatch):
    """DELTA1.1 thu nhỏ: không nạp CLIP, vision / text_proj giữ nguyên weight khởi tạo; box ảnh inpaint == ảnh gốc (không nhìn ảnh)."""
    import ce_localization.models.clip_refiner as cr
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = _gamma2_cfg(tmp_path, base, "delta1_1")
    a = str(tmp_path / "a")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a])
    monkeypatch.setattr(cr, "load_clip", lambda *x: (_ for _ in ()).throw(AssertionError("không được nạp CLIP")))
    k = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    assert k["iter"] == 4 and "clip_sha" not in k["model"] and not any(n.startswith("refiner.") for n in k["model"])
    from ce_localization.models.detector import build_model
    torch.manual_seed(cfg["training"]["seed"])
    init = build_model(cfg).state_dict()
    for n in k["model"]:
        if n.startswith(("vision.", "text_proj.")) and "running" not in n and "num_batches" not in n:
            assert torch.equal(k["model"][n], init[n]), n
    import ce_localization.eval as ea
    import ce_localization.train as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out, dump = str(tmp_path / "res.json"), str(tmp_path / "boxes.json")
    monkeypatch.setattr(sys, "argv", ["eval.py", "--ckpt", os.path.join(a, "best.pth"), "--split", "test", "--n-samples", "3",
                                      "--num-workers", "0", "--out", out, "--device", "cpu", "--add-samplers", "mock",
                                      "--dump-boxes", dump])
    ea.main()
    with open(dump) as f:
        d = json.load(f)
    assert set(d["results"]) == {"inpainted", "original"}
    for x, y in zip(d["results"]["inpainted"], d["results"]["original"]):
        assert x["image_id"] == y["image_id"] and x["boxes"] == y["boxes"]


@pytest.mark.parametrize("kind", ["delta4", "delta3"])
def test_eval_cocount_delta(tmp_path, monkeypatch, kind):
    """CE-CoCount + --obj-size + --dump-boxes chạy với checkpoint DELTA (density trống, token chữ của lớp CoCount)."""
    from ce_localization.models.detector import build_model
    from tests.ce_localization.test_cocount import _fake_cocount
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    root = _fake_cocount(str(tmp_path / "cc"))
    _, cfg = _gamma2_cfg(tmp_path, base, kind)
    _patch_clip(monkeypatch)
    torch.manual_seed(0)
    model = build_model(cfg, pretrained_backbone=False)
    model.set_text_fingerprint(FAKE_TEXT_SHA)
    ck = str(tmp_path / "ck.pth")
    torch.save({"model": model.state_dict(), "config": cfg, "iter": 0}, ck)
    res = _eval(monkeypatch, ck, str(tmp_path / "res.json"),
                ["--dataset", "cocount", "--cocount-root", root, "--add-samplers", "mock", "--cocount-obj-filter", "3",
                 "--obj-size", "mean", "--dump-boxes", str(tmp_path / "b.json")])
    keys = {"cocount"} | ({"cocount_norefine", "cocount_noobj"} if kind == "delta4" else set())
    want = keys | {k + "_objsize" for k in keys}
    assert set(res["results"]) == want and res["results"]["cocount"]["n"] == 4


def test_train_max_iter_without_steps_and_batch_override(tmp_path, monkeypatch):
    """G3 / G4 cho config lịch cosine (không có `training.steps`): `--max-iter` không vỡ; `--batch-size` ghi đè batch toàn cục."""
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    _, cfg = _gamma2_cfg(tmp_path, base, "delta1_1")
    cfg["training"].pop("steps")
    cfg["training"]["batch_size"] = 4
    p = str(tmp_path / "c.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    save = str(tmp_path / "s")
    _run_train(monkeypatch, ["--config", p, "--save-dir", save, "--max-iter", "2", "--batch-size", "2"])
    k = torch.load(os.path.join(save, "last.pth"), weights_only=False)
    assert k["iter"] == 2 and k["config"]["training"]["batch_size"] == 2 and "steps" not in k["config"]["training"]


def _ddp_worker_delta(rank, world, port, argv):
    from tests.ce_localization.test_train_eval import _ddp_worker_strict
    _ddp_worker_strict(rank, world, port, argv, None)


@pytest.mark.parametrize("kind", ["delta4", "delta3_2", "delta4_1", "delta5_2"])
def test_ddp_delta_no_find_unused(tmp_path, kind):
    """DELTA trên 2 tiến trình gloo, DDP KHÔNG find_unused + gradient_as_bucket_view: tham số nào thiếu grad hay lệch stride thì lỗi
    ngay; loss hữu hạn, đủ iter."""
    import socket
    import torch.multiprocessing as tmp
    from ce_localization.train import ddp_find_unused
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = _gamma2_cfg(tmp_path, base, kind)
    assert not ddp_find_unused(cfg)
    save = str(tmp_path / "ddp")
    with socket.socket() as so:
        so.bind(("127.0.0.1", 0))
        port = so.getsockname()[1]
    tmp.spawn(_ddp_worker_delta, args=(2, port, ["--config", c_path, "--save-dir", save, "--max-iter", "3", "--eval-every", "3"]),
              nprocs=2, join=True)
    ck = torch.load(os.path.join(save, "last.pth"), weights_only=False)
    assert ck["iter"] == 3 and all(np.isfinite(h["loss"]) and h["skipped"] == 0 for h in ck["history"] if "loss" in h)


@pytest.mark.skipif(os.environ.get("CE_HF_TESTS") != "1", reason="cửa G1-HF: CLIP thật từ HF — chạy trên server / Kaggle với CE_HF_TESTS=1")
def test_hf_clip_real(monkeypatch):
    """G1-HF: CLIP ViT thật của run (`CE_HF_CLIP`, mặc định B/16) — lưới = ảnh / patch, `encode_context` dùng đúng
    `hidden_states[-2][:, 1:]` ở độ phân giải của ViT; token chữ CLIP text B/16 = tokenize theo batch của tác giả; in vân tay."""
    import importlib
    import ce_localization.models.clip_refiner as cr
    importlib.reload(cr)                                             # bỏ CLIP giả của fixture
    from transformers import CLIPTextModel, CLIPTokenizer
    from ce_localization.models.text import encode_class_tokens
    name, text_name = os.environ.get("CE_HF_CLIP", B16), B16
    vis, th = cr.load_clip(name, text_name)
    res, g = vis.config.image_size, vis.config.image_size // vis.config.patch_size
    assert th == 512
    r = cr.ClipBoxRefiner({**AUTHOR_REFINER, "clip_model_name": name, "text_model_name": text_name}, vis, th).eval()
    assert r.grid == g and r.clip_res == res
    rgb = torch.rand(2, 3, 512, 512)
    with torch.no_grad():
        ctx = r.encode_context(rgb, torch.zeros(2, 1, 512, 512), torch.zeros(2, 3, 512), torch.ones(2, 3, dtype=torch.bool))
        px = F.interpolate(rgb, size=(res, res), mode="bicubic", align_corners=False, antialias=True)
        px = (px.clamp(0, 1) - r.clip_mean) / r.clip_std
        hid = vis(pixel_values=px, output_hidden_states=True).hidden_states[-2][:, 1:]
        assert ctx["image_tokens"].shape == (2, g * g, 256)
        assert torch.allclose(ctx["image_tokens"], r.image_proj(hid) + r.density_proj(torch.zeros(2, g * g, 1)), atol=1e-5)
    names = ["apple", "hot air balloon", "candy piece", "sea shell"]
    table, sha = encode_class_tokens(names, text_name)
    tok, model = CLIPTokenizer.from_pretrained(text_name), CLIPTextModel.from_pretrained(text_name).eval()
    inp = tok(names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**inp).last_hidden_state
    for i, n in enumerate(names):
        m = inp["attention_mask"][i].bool()
        assert torch.allclose(table[n], ref[i][m], atol=1e-5), n
    print(f"\n[G1-HF] {name}: lưới {g}×{g}, ảnh {res} | vân tay CLIP ViT {cr.weights_sha256(vis).hex()} | CLIP text B/16 {sha.hex()}")
