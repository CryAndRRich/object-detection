"""DELTA (docs/EXPERIMENT_DELTA.md): BoxRefiner của tác giả gốc trong CE-Loc (`models/clip_refiner.py`, `models/box_policy.py`).

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

from tests.ce_localization.helpers import (CFG_G, FAKE_TEXT_SHA, _fake_load_clip, _fake_text_table, _fake_turn_index,
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
    vis, th = _fake_load_clip("x")
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


def _policy(refiner=AUTHOR_REFINER, use_condition=True, seed=0, **kw):
    from ce_localization.models.box_policy import BoxPolicy
    torch.manual_seed(seed)
    return BoxPolicy(in_channels=4, pretrained_backbone=False, num_timesteps=20, vision="r18_paper", ss_mask=True,
                     refiner=refiner, use_condition=use_condition, **kw)


def _batch(B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, 4, 128, 128, generator=g)
    vhw = torch.tensor([[96, 128], [128, 112]])[:B]
    tok = (torch.randn(B, 5, 512, generator=g), torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)[:B])
    return x, torch.randn(B, 512, generator=g), vhw, torch.rand(B, 4, generator=g) * 2 - 1, tok


# ----------------------------------------------------------------------------- port đúng tác giả

@pytest.mark.parametrize("case", ["per_row", "shared", "flow_no_text", "flow_full"])
def test_refiner_split_equals_author_forward(monkeypatch, case):
    """Đường tách đôi (luồng ngữ cảnh một lần mỗi (ảnh, t)) == `forward` nguyên văn của tác giả: đầu ra VÀ grad mọi tham số (float64,
    AdaLN + head khác 0, 2 stage, chữ dài khác nhau). `shared`: 3 mẫu / ảnh cùng t (vòng mock / DDPM); flow ảnh -> box: đường full."""
    flow = {"flow_no_text": {"sample": ["sample", "image"], "image": ["image", "text"], "text": ["text", "image"]},
            "flow_full": {"sample": ["sample", "image", "text"], "image": ["image", "text", "sample"], "text": ["text"]}}.get(case)
    r = _refiner({"attention_flow": flow} if flow else None)
    assert r.split_ok == (case != "flow_full")
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
    vis, th = _fake_load_clip("x")
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
    monkeypatch.setattr(cr, "load_clip", lambda name: _fake_load_clip(name, seed=1))
    b = _policy()
    with pytest.raises(RuntimeError, match="vân tay CLIP ViT"):
        b.load_state_dict(sd)
    monkeypatch.setattr(cr, "load_clip", lambda name: _fake_load_clip(name, seed=0))
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
    monkeypatch.setattr(cr, "load_clip", lambda name: (_ for _ in ()).throw(AssertionError("không được nạp CLIP")))
    m = _policy(use_condition=False).train()
    assert m.refiner is None and not m.needs_text_tokens and "clip_sha" not in m.state_dict()
    frozen = {n for n, p in m.named_parameters() if not p.requires_grad}
    assert frozen == {n for n, _ in m.named_parameters() if n.startswith(("vision.", "text_proj."))} and frozen
    x, text, vhw, x0, _ = _batch()
    m(x, text, vhw, x0, k=2).backward()
    assert [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None] == []
    m.eval()
    with torch.no_grad():
        a = m.sample(x, text, vhw, 3, generator=torch.Generator().manual_seed(1), sampler="mock", mock_steps=10)
        b = m.sample(torch.rand_like(x), torch.randn_like(text), vhw, 3, generator=torch.Generator().manual_seed(1), sampler="mock",
                     mock_steps=10)
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


def test_sample_shared_context_equals_expanded_and_mock_steps():
    """`sample_from_cond` (ngữ cảnh một lần / ảnh) == ngữ cảnh nhân sẵn cho từng mẫu; `mock_steps` = số lần gọi ε_θ."""
    m = _policy().eval()
    for blk in m.refiner.stages[0]:
        nn.init.normal_(blk.ada[-1].weight, std=0.05)
    nn.init.normal_(m.refiner.heads[0][-1].weight, std=0.05)
    x, text, vhw, _, tok = _batch()
    calls = []
    h = m.noise_net.register_forward_hook(lambda *a: calls.append(1))
    with torch.no_grad():
        cond = m.condition(x, text, vhw)
        ctx = m.refiner_context(x, tok)
        a = m.sample_from_cond(cond, 3, torch.Generator().manual_seed(5), "mock", ctx=ctx, mock_steps=7)
        assert len(calls) == 7
        ci = torch.arange(2).repeat_interleave(3)
        ctx_e, cond_e = m.refiner.index_context(ctx, ci), cond.repeat_interleave(3, 0)
        from ce_localization.models.box_policy import mock_sample
        b = mock_sample(lambda xx, t: m.predict_noise(xx, t, cond_e, ctx_e)[0], 6, m.alphas_cumprod, steps=7,
                        generator=torch.Generator().manual_seed(5))
    h.remove()
    assert torch.allclose(a.view(6, 4), b, atol=1e-5)


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
    from ce_localization.models.detector import build_model
    with pytest.raises(ValueError, match="obj_attn"):
        _policy(obj_attn=True)
    from ce_localization.models.box_policy import BoxPolicy
    with pytest.raises(ValueError, match="4 kênh"):
        BoxPolicy(in_channels=3, pretrained_backbone=False, num_timesteps=20, vision="r18_paper", refiner=AUTHOR_REFINER)
    with pytest.raises(ValueError, match="canvas"):
        BoxPolicy(in_channels=4, pretrained_backbone=False, num_timesteps=20, refiner=AUTHOR_REFINER)
    cfg = yaml.safe_load(open(CFG_G["delta1"]))
    cfg["data"]["input_style"] = "ours"
    cfg["model"]["pretrained_backbone"] = False
    with pytest.raises(ValueError, match="input_style"):
        build_model(cfg)


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
    assert keys(add_variants(["mock"])) == [""]
    v = add_variants(["mock"], [100, 200, 500, 1000], refiner=True)
    assert keys(v) == ["", "_mock200", "_mock500", "_mock1000", "_norefine"]
    assert [x["sample_kw"] for _, _, x in v] == [{"mock_steps": n} for n in (100, 200, 500, 1000)] + \
        [{"mock_steps": 100, "use_refiner": False}]
    assert keys(add_variants(["mock", "ddpm"], [100, 1000])) == ["_mock", "_mock1000", "_ddpm"]
    assert keys(add_variants(["ddpm"])) == [""]
    o = add_variants(["mock"], [100, 500], obj_attn=True)
    assert keys(o) == ["", "_mock500", "_noobj", "_mock500_noobj"] and o[3][2]["sample_kw"] == {"mock_steps": 500, "use_objects": False}


def test_grad_groups_refiner():
    from ce_localization.utils.grad_monitor import group_of
    assert group_of("refiner.stages.0.1.qkv.weight") == "refiner.blocks"
    assert group_of("refiner.heads.0.3.weight") == "refiner.head"
    assert group_of("refiner.image_proj.1.weight") == "refiner.embed"
    assert group_of("noise_net.final_conv.1.weight") == "unet1d"


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


def test_full_flow_delta1_train_resume_eval(tmp_path, monkeypatch):
    """DELTA1 thu nhỏ (refiner kích thước thật, CLIP giả): train 2 + --resume tới 4 == train liền 4 (mọi tensor kể cả `clip_sha`);
    history có thống kê refiner; eval mock 5 / 10 bước + `_norefine` cho cả 2 loại ảnh; CLIP khác bản lúc train ⇒ eval từ chối."""
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = _gamma2_cfg(tmp_path, base, "delta1")
    assert cfg["model"]["refiner"]["d_model"] == 256
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--max-iter", "2"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4 and set(ka["model"]) == set(kb["model"])
    assert "clip_sha" in kb["model"] and not any("vision_backbone" in k for k in kb["model"])
    assert bytes(kb["model"]["clip_sha"][1].tolist()) == FAKE_TEXT_SHA
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    assert float(kb["model"]["refiner.heads.0.3.weight"].abs().sum()) > 0       # head rời 0 sau train
    logs = [h for h in kb["history"] if "loss" in h]
    assert all({"loss_eps", "loss_aux", "delta_abs", "aux_ratio", "refiner_gates"} <= set(h) for h in logs)
    res = _eval(monkeypatch, os.path.join(b, "best.pth"), str(tmp_path / "res.json"),
                ["--add-samplers", "mock", "--mock-steps", "5", "10", "--dump-boxes", str(tmp_path / "boxes.json")])
    want = {i + s for i in ("inpainted", "original") for s in ("_mock5", "_mock10")}    # không mock 100 => không khoá cũ / _norefine
    assert set(res["results"]) == want and res["mock_steps"] == [5, 10]
    assert all(r["n"] > 0 and r["refine_delta_by_t"] for r in res["results"].values())
    with pytest.raises(SystemExit):                                  # mock 100 bước > T = 20 của config thu nhỏ: dừng ngay
        _eval(monkeypatch, os.path.join(b, "best.pth"), str(tmp_path / "x.json"), ["--add-samplers", "mock"])
    with pytest.raises(RuntimeError, match="vân tay"):
        _eval(monkeypatch, os.path.join(b, "best.pth"), str(tmp_path / "y.json"), ["--add-samplers", "ddpm"], clip_seed=1)


def test_eval_delta1_norefine_variant(tmp_path, monkeypatch):
    """`_norefine` chỉ có ở mock 100 bước: checkpoint DELTA1 dựng với T = 100 để chạy được mock 100."""
    from ce_localization.models.detector import build_model
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    _, cfg = _gamma2_cfg(tmp_path, base, "delta1")
    cfg["diffusion"]["num_timesteps"] = 100
    _patch_clip(monkeypatch)
    torch.manual_seed(0)
    model = build_model(cfg, pretrained_backbone=False)
    model.set_text_fingerprint(FAKE_TEXT_SHA)
    ck = str(tmp_path / "ck.pth")
    torch.save({"model": model.state_dict(), "config": cfg, "iter": 0}, ck)
    res = _eval(monkeypatch, ck, str(tmp_path / "res.json"), ["--add-samplers", "mock", "--mock-steps", "100", "20",
                                                              "--image", "inpainted", "--limit", "2"])
    assert set(res["results"]) == {"inpainted", "inpainted_mock20", "inpainted_norefine"}
    assert res["results"]["inpainted_norefine"]["refine_delta_by_t"] is None


def test_full_flow_delta1_1_train_eval(tmp_path, monkeypatch):
    """DELTA1.1 thu nhỏ: không nạp CLIP, vision / text_proj giữ nguyên weight khởi tạo; box ảnh inpaint == ảnh gốc (không nhìn ảnh)."""
    import ce_localization.models.clip_refiner as cr
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, cfg = _gamma2_cfg(tmp_path, base, "delta1_1")
    a = str(tmp_path / "a")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", a])
    monkeypatch.setattr(cr, "load_clip", lambda name: (_ for _ in ()).throw(AssertionError("không được nạp CLIP")))
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
                                      "--mock-steps", "10", "--dump-boxes", dump])
    ea.main()
    with open(dump) as f:
        d = json.load(f)
    assert set(d["results"]) == {"inpainted_mock10", "original_mock10"}
    for x, y in zip(d["results"]["inpainted_mock10"], d["results"]["original_mock10"]):
        assert x["image_id"] == y["image_id"] and x["boxes"] == y["boxes"]


def test_eval_cocount_delta1(tmp_path, monkeypatch):
    """CE-CoCount + --obj-size + --dump-boxes chạy với checkpoint DELTA1 (density trống, token chữ của lớp CoCount)."""
    from ce_localization.models.detector import build_model
    from tests.ce_localization.test_cocount import _fake_cocount
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    root = _fake_cocount(str(tmp_path / "cc"))
    _, cfg = _gamma2_cfg(tmp_path, base, "delta1")
    _patch_clip(monkeypatch)
    torch.manual_seed(0)
    model = build_model(cfg, pretrained_backbone=False)
    model.set_text_fingerprint(FAKE_TEXT_SHA)
    ck = str(tmp_path / "ck.pth")
    torch.save({"model": model.state_dict(), "config": cfg, "iter": 0}, ck)
    res = _eval(monkeypatch, ck, str(tmp_path / "res.json"),
                ["--dataset", "cocount", "--cocount-root", root, "--add-samplers", "mock", "--mock-steps", "10",
                 "--cocount-obj-filter", "3", "--obj-size", "mean", "--dump-boxes", str(tmp_path / "b.json")])
    assert set(res["results"]) == {"cocount_mock10", "cocount_mock10_objsize"} and res["results"]["cocount_mock10"]["n"] == 4


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


@pytest.mark.parametrize("kind", ["delta1", "delta1_1"])
def test_ddp_delta_no_find_unused(tmp_path, kind):
    """DELTA1 / 1.1 trên 2 tiến trình gloo, DDP KHÔNG find_unused + gradient_as_bucket_view: tham số nào thiếu grad hay lệch stride
    thì lỗi ngay; loss hữu hạn, đủ iter."""
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


@pytest.mark.skipif(os.environ.get("CE_HF_TESTS") != "1", reason="cửa G1-HF: CLIP B/16 thật từ HF — chạy trên server với CE_HF_TESTS=1")
def test_hf_clip_b16_real(monkeypatch):
    """G1-HF (server): CLIP ViT-B/16 thật — lưới 14, hidden 768; `encode_context` dùng đúng `hidden_states[-2][:, 1:]`; bảng token
    chữ = tokenize theo batch của tác giả; in vân tay (ghi vào docs)."""
    import importlib
    import ce_localization.models.clip_refiner as cr
    importlib.reload(cr)                                             # bỏ CLIP giả của fixture
    from transformers import CLIPTextModel, CLIPTokenizer
    from ce_localization.models.text import encode_class_tokens
    name = AUTHOR_REFINER["clip_model_name"]
    vis, th = cr.load_clip(name)
    assert th == 512 and vis.config.hidden_size == 768 and vis.config.image_size // vis.config.patch_size == 14
    r = cr.ClipBoxRefiner(AUTHOR_REFINER, vis, th).eval()
    rgb = torch.rand(2, 3, 512, 512)
    with torch.no_grad():
        ctx = r.encode_context(rgb, torch.zeros(2, 1, 512, 512), torch.zeros(2, 3, 512), torch.ones(2, 3, dtype=torch.bool))
        px = F.interpolate(rgb, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        px = (px.clamp(0, 1) - r.clip_mean) / r.clip_std
        hid = vis(pixel_values=px, output_hidden_states=True).hidden_states[-2][:, 1:]
        assert ctx["image_tokens"].shape == (2, 196, 256)
        assert torch.allclose(ctx["image_tokens"], r.image_proj(hid) + r.density_proj(torch.zeros(2, 196, 1)), atol=1e-5)
    names = ["apple", "hot air balloon", "candy piece", "sea shell"]
    table, sha = encode_class_tokens(names, name)
    tok, model = CLIPTokenizer.from_pretrained(name), CLIPTextModel.from_pretrained(name).eval()
    inp = tok(names, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        ref = model(**inp).last_hidden_state
    for i, n in enumerate(names):
        m = inp["attention_mask"][i].bool()
        assert torch.allclose(table[n], ref[i][m], atol=1e-5), n
    print(f"\n[G1-HF] vân tay CLIP ViT {cr.weights_sha256(vis).hex()} | CLIP text {sha.hex()}")
