"""Mô hình (`models/`): backbone R-50 + FPN (3 / 4 kênh), chọn tầng + RoIAlign, `apply_deltas`,
memory (SpatialSoftmax có mask, PE 2D, lưới), detector (grad mọi tham số, bias focal, sample).
"""

import math

import pytest
import torch
import torch.nn as nn
from torchvision.ops import roi_align
from torchvision.ops.misc import FrozenBatchNorm2d

from ce_localization.engine import criterion as C
from ce_localization.engine.diffusion import prepare_train_boxes
from ce_localization.models.backbone import ResNet50FPN
from ce_localization.models.detector import Detector
from ce_localization.models.head import SCALE_CLAMP, apply_deltas
from ce_localization.models.memory import MemoryEncoder, masked_spatial_softmax, sine_pos_2d, valid_cells_mask
from ce_localization.models.roi import MultiLevelRoIAlign, assign_levels


def test_backbone_frozen_bn_all_convs_trainable():
    m = ResNet50FPN(pretrained=False)
    assert not any(isinstance(x, nn.BatchNorm2d) for x in m.modules())
    assert sum(isinstance(x, FrozenBatchNorm2d) for x in m.modules()) == 53
    convs = [x for x in m.modules() if isinstance(x, nn.Conv2d)]
    assert convs and all(c.weight.requires_grad for c in convs)       # KHÔNG đóng băng lớp nào
    out = m(torch.randn(1, 3, 128, 128))
    assert [tuple(v.shape[-2:]) for v in out.values()] == [(32, 32), (16, 16), (8, 8), (4, 4)]
    assert all(v.shape[1] == 256 for v in out.values())


def test_backbone_4ch_density_weight_zero_and_equals_3ch():
    from ce_localization.models.backbone import density_weight_ratio
    torch.manual_seed(0)
    m3 = ResNet50FPN(pretrained=False).eval()
    torch.manual_seed(0)
    m4 = ResNet50FPN(pretrained=False, in_channels=4).eval()
    w = m4.stem[0].weight
    assert w.shape == (64, 4, 7, 7) and w.requires_grad and w[:, 3].abs().max() == 0
    assert density_weight_ratio(m4) == 0.0 and density_weight_ratio(m3) is None
    assert [n for n, _ in m3.named_parameters()] == [n for n, _ in m4.named_parameters()]  # thứ tự cũ
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        o3 = m3(x)
        o4a = m4(torch.cat([x, torch.zeros(1, 1, 64, 64)], 1))
        o4b = m4(torch.cat([x, torch.rand(1, 1, 64, 64)], 1))
    for k in o3:
        assert torch.allclose(o3[k], o4a[k], atol=1e-6) and torch.allclose(o4a[k], o4b[k], atol=1e-6), k


def test_assign_levels_detectron2_formula():
    s = torch.tensor([16.0, 111.9, 112.0, 224.0, 448.0, 2000.0, 0.0])
    boxes = torch.stack([torch.zeros_like(s), torch.zeros_like(s), s, s], 1)
    # floor(4 + log2(s/224)): 16 -> 0.19 -> 2 (kẹp) ; 112 -> 3 ; 224 -> 4 ; 448 -> 5 ; 2000 -> 5 (kẹp)
    assert assign_levels(boxes).tolist() == [0, 0, 1, 2, 3, 3, 0]


def test_multilevel_roi_align_equals_roi_align_on_assigned_level():
    torch.manual_seed(0)
    feats = [torch.randn(2, 8, 128 // s, 128 // s) for s in (4, 8, 16, 32)]
    boxes = torch.tensor([[[5, 5, 20, 20], [0, 0, 120, 120]], [[30, 40, 90, 100], [10, 10, 12, 60]]],
                         dtype=torch.float32)
    out = MultiLevelRoIAlign()(feats, boxes)
    flat = boxes.reshape(-1, 4)
    lv = assign_levels(flat)
    for k in range(4):
        b = k // 2
        ref = roi_align(feats[lv[k]], [flat[k:k + 1]], 7, spatial_scale=1 / (4 * 2 ** int(lv[k])),
                        sampling_ratio=2, aligned=True)
        assert torch.allclose(out[k], ref[0] if b == 0 else roi_align(
            feats[lv[k]][1:2], [flat[k:k + 1]], 7, spatial_scale=1 / (4 * 2 ** int(lv[k])),
            sampling_ratio=2, aligned=True)[0], atol=1e-6)


def test_apply_deltas_formula():
    boxes = torch.tensor([[10.0, 20.0, 30.0, 60.0]])                   # w 20, h 40, tâm (20, 40)
    assert torch.allclose(apply_deltas(torch.zeros(1, 4), boxes), boxes)
    # dx=2 / wx=2 -> dịch 1·w ; dw=log2 -> rộng gấp đôi
    out = apply_deltas(torch.tensor([[2.0, 0.0, math.log(2), 0.0]]), boxes)
    assert torch.allclose(out, torch.tensor([[20.0, 20.0, 60.0, 60.0]]))
    big = apply_deltas(torch.tensor([[0.0, 0.0, 100.0, 100.0]]), boxes)
    assert torch.allclose(big[0, 2] - big[0, 0], torch.tensor(20.0 * math.exp(SCALE_CLAMP)))


def test_masked_spatial_softmax_uniform_onehot_and_padding():
    B, C_, H, W, s = 1, 3, 4, 4, 32
    valid_hw = torch.tensor([[96, 128]])                              # 3 hàng thật, 4 cột
    valid = valid_cells_mask(valid_hw, H, W, s)
    assert valid[0].sum() == 12 and not valid[0, 3].any()
    feat = torch.zeros(B, C_, H, W)
    feat[0, 1, 1, 2] = 50.0                                          # kênh 1: một đỉnh ở ô (hàng 1, cột 2)
    feat[0, 2, 3, :] = 1e4                                           # kênh 2: đỉnh CHỈ ở hàng đệm
    xy = masked_spatial_softmax(feat, valid, valid_hw, s)
    assert torch.allclose(xy[0, 0], torch.tensor([0.5, 0.5]), atol=1e-6)          # đều -> tâm vùng thật
    assert torch.allclose(xy[0, 1], torch.tensor([2.5 * 32 / 128, 1.5 * 32 / 96]), atol=1e-4)  # (x, y)
    assert torch.allclose(xy[0, 2], xy[0, 0], atol=1e-6)             # hàng đệm không ảnh hưởng


def test_sine_pos_2d_ignores_padding_rows():
    v_pad = valid_cells_mask(torch.tensor([[96, 128]]), 4, 4, 32)     # 3 hàng thật + 1 hàng đệm
    v_full = torch.ones(1, 3, 4, dtype=torch.bool)
    p_pad, p_full = sine_pos_2d(v_pad, 16), sine_pos_2d(v_full, 16)
    assert p_pad.shape == (1, 32, 4, 4)
    assert torch.allclose(p_pad[:, :, :3], p_full, atol=1e-6)


def test_memory_encoder_kinds():
    t, text = torch.tensor([5, 999]), torch.randn(2, 512)
    p5 = torch.randn(2, 256, 4, 4)
    vh = torch.tensor([[96, 128], [128, 128]])
    for kind, M in (("none", 2), ("spatial_softmax", 3), ("grid", 18)):
        enc = MemoryEncoder(kind)
        assert hasattr(enc, "ss_proj") == (kind == "spatial_softmax")
        assert hasattr(enc, "grid_proj") == (kind == "grid")
        tok, msk = enc.image_tokens(p5, vh)
        mem, mask = enc(t, text, tok, msk)
        assert mem.shape == (2, M, 256)
        if kind == "grid":
            assert mask.shape == (2, 18) and not mask[:, :2].any()
            assert mask[0].sum() == 4 and mask[1].sum() == 0          # ảnh 0: 1 hàng đệm = 4 ô
        else:
            assert mask is None


def test_grid_pooling_uses_only_real_region_and_fixed_tokens():
    """grid_size=G: cắt vùng thật của P5 rồi pool về G×G -> luôn G² token, không mask, và giá trị ở
    hàng đệm không ảnh hưởng (canvas 1024: P5 32×32 -> 16×16)."""
    torch.manual_seed(0)
    enc = MemoryEncoder("grid", grid_size=4)
    p5 = torch.randn(2, 256, 8, 8)
    vh = torch.tensor([[5 * 32 - 10, 256], [256, 256]])              # ảnh 0: 5 hàng thật (hàng cuối lẻ)
    tok, mask = enc.image_tokens(p5, vh)
    assert tok.shape == (2, 16, 256) and mask is None
    p5b = p5.clone()
    p5b[0, :, 5:] = 1e3                                              # đổi hàng đệm của ảnh 0
    tok_b, _ = enc.image_tokens(p5b, vh)
    assert torch.allclose(tok, tok_b)
    ref = torch.nn.functional.adaptive_avg_pool2d(p5[0:1, :, :5, :], 4)
    pos = sine_pos_2d(torch.ones(1, 4, 4, dtype=torch.bool), 128)
    want = enc.grid_proj(ref.flatten(2).transpose(1, 2)) + pos.flatten(2).transpose(1, 2)
    assert torch.allclose(tok[0:1], want, atol=1e-5)
    cx, cy, valid = enc.grid_geometry(vh, 8, 8)
    assert cx.shape == (2, 16) and valid.all() and cy[0].max() < 5 * 32 and cx[0].max() < 256


@pytest.mark.parametrize("kind", ["none", "spatial_softmax", "grid"])
def test_model_all_params_get_grad_and_focal_bias_only_on_class_logits(kind):
    torch.manual_seed(0)
    m = Detector(memory=kind, pretrained_backbone=False)
    bias = -math.log(0.99 / 0.01)
    hit = [n for n, p in m.named_parameters() if p.numel() and torch.all(p == bias)]
    assert hit == [f"head.stages.{i}.class_logits.bias" for i in range(6)]
    images = torch.randn(2, 3, 128, 128)
    gt = [torch.tensor([[10.0, 10.0, 40.0, 30.0], [60, 20, 90, 70.0]]), torch.tensor([[5.0, 5.0, 50.0, 50.0]])]
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])
    vh = torch.tensor([[96, 128], [128, 128]])
    boxes, t = prepare_train_boxes(gt, whwh, 16, m.alphas_cumprod, 2.0, torch.Generator().manual_seed(0))
    logits, pred = m(images, torch.randn(2, 512), vh, boxes, t)
    assert logits.shape == (6, 2, 16, 1) and pred.shape == (6, 2, 16, 4)
    crit = C.Criterion({"alpha": 0.25, "gamma": 2.0, "class_weight": 2.0, "l1_weight": 5.0,
                             "giou_weight": 2.0}, {"ota_k": 5, "center_radius": 2.5})
    loss, _ = crit(logits, pred, C.build_targets(gt, whwh))
    loss.backward()
    no_grad = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    # chỉ conv đầu ra của tầng FPN không có RoI nào rơi vào mới được phép thiếu grad (ảnh 128 px:
    # box lớn nhất sqrt(area) = 128 -> tối đa P4) — vì vậy DDP chạy find_unused_parameters=True
    assert all(n.startswith("backbone.fpn.layer_blocks.") for n in no_grad), no_grad
    if kind != "none":                                               # P5 vào memory -> có grad
        assert not any(n.startswith("backbone.fpn.layer_blocks.3.") for n in no_grad)
    assert not any(".norm_in." in n for n, _ in m.named_parameters() if n.startswith("head.stages.0."))


def test_model_sample_one_and_multi_step():
    m = Detector(memory="grid", pretrained_backbone=False).eval()
    img = torch.randn(1, 3, 128, 128)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]])
    o = m.sample(img, torch.randn(1, 512), torch.tensor([[96, 128]]), whwh, 10, steps=1)
    assert o["boxes"].shape == (1, 10, 4) and o["stage_boxes"].shape == (6, 1, 10, 4)
    o4 = m.sample(img, torch.randn(1, 512), torch.tensor([[96, 128]]), whwh, 10, steps=4)
    assert o4["boxes"].shape[1] == 40


# ----------------------------------------------------------------------------- GAMMA (bài add)

def test_forward_p5_equals_fpn_p5():
    m = ResNet50FPN(pretrained=False).eval()
    x = torch.randn(1, 3, 128, 96)
    with torch.no_grad():
        assert torch.allclose(m.forward_p5(x), m(x)["p5"], atol=1e-6)


def test_unet1d_horizon_one_shape():
    from ce_localization.models.unet1d import ConditionalUnet1D
    net = ConditionalUnet1D(4, global_cond_dim=256)
    out = net(torch.randn(5, 1, 4), torch.randint(0, 1000, (5,)), torch.randn(5, 256))
    assert out.shape == (5, 1, 4)


def test_box_unit_roundtrip():
    from ce_localization.models.box_policy import boxes_to_unit, unit_to_boxes
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]])
    b = torch.tensor([[10.0, 20.0, 50.0, 60.0]])
    u = boxes_to_unit(b, whwh)
    assert torch.allclose(u, torch.tensor([[30 / 128 * 2 - 1, 40 / 96 * 2 - 1, 40 / 128 * 2 - 1, 40 / 96 * 2 - 1]]))
    assert torch.allclose(unit_to_boxes(u, whwh), b, atol=1e-5)


@pytest.mark.parametrize("in_ch,src", [(3, "c5"), (4, "c5"), (4, "p5")])
def test_box_policy_loss_trains_backbone_cond_and_unet(in_ch, src):
    from ce_localization.models.box_policy import BoxPolicy
    torch.manual_seed(0)
    m = BoxPolicy(in_channels=in_ch, pretrained_backbone=False, num_timesteps=20, ss_source=src)
    assert m.vis_proj.in_features == (4096 if src == "c5" else 512)
    if in_ch == 4:
        assert m.backbone.stem[0].weight[:, 3].abs().max() == 0          # kênh density khởi tạo 0 (như ALPHA3)
    x = torch.randn(2, in_ch, 128, 128)
    loss = m(x, torch.randn(2, 512), torch.tensor([[96, 128], [128, 128]]), torch.rand(2, 4) * 2 - 1, k=3)
    loss.backward()
    g = {n: p.grad for n, p in m.named_parameters()}
    for n in ("backbone.layer4.2.conv3.weight", "backbone.stem.0.weight", "vis_proj.weight",
              "text_proj.0.weight", "noise_net.final_conv.1.weight", "noise_net.diffusion_step_encoder.1.weight"):
        assert g[n] is not None and g[n].abs().sum() > 0, n
    assert g["backbone.fpn.layer_blocks.0.0.weight"] is None             # P2..P4 không dùng
    fpn_used = g["backbone.fpn.layer_blocks.3.0.weight"] is not None
    assert fpn_used == (src == "p5")                                     # C5: không qua FPN
    assert m.alphas_cumprod.shape == (20,)
    with torch.no_grad():
        s = m.sample(x, torch.randn(2, 512), torch.tensor([[96, 128], [128, 128]]), 4)
    assert s.shape == (2, 4, 4) and torch.isfinite(s).all()



def test_paper_spatial_softmax_matches_original_code():
    """`SpatialSoftmax` của refs/repos/Count-Editing/CE-LocModel/models/spatial_softmax.py chép nguyên (meshgrid mặc
    định = 'ij')."""
    import torch.nn.functional as F
    from ce_localization.models.box_policy import _paper_spatial_softmax
    fm = torch.randn(2, 5, 4, 6)
    N, C, H, W = fm.shape
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    att = F.softmax(fm.reshape(N, C, -1), dim=-1)
    ex = torch.sum(pos_x.reshape(H * W) * att, dim=-1, keepdim=True)
    ey = torch.sum(pos_y.reshape(H * W) * att, dim=-1, keepdim=True)
    assert torch.equal(_paper_spatial_softmax(fm), torch.cat([ex, ey], dim=-1).reshape(N, -1))


def test_load_celoc_paper_renames_and_checks(tmp_path):
    from ce_localization.models.box_policy import BoxPolicy
    from tests.ce_localization.helpers import _fake_paper_ckpt
    p = str(tmp_path / "best_model.pth")
    ref = _fake_paper_ckpt(p)
    m, clip, info = BoxPolicy.load_celoc_paper(p)
    assert info["in_channels"] == 4 and info["num_timesteps"] == 20 and info["epoch"] == 113
    assert list(clip) == ["text_model.final_layer_norm.weight"] and info["schedule_max_err"] == 0
    for k, v in ref.state_dict().items():
        assert torch.equal(m.state_dict()[k], v), k
    assert m.vision.backbone[0].weight.shape == (64, 4, 7, 7) and m.vision.projection.weight.shape == (128, 1024)
    ck = torch.load(p, weights_only=False)
    ck["model_state_dict"]["vision_encoder.extra"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        BoxPolicy.load_celoc_paper(ck)
    ck["model_state_dict"].pop("vision_encoder.extra")
    ck["model_state_dict"]["alphas_cumprod"] = ck["model_state_dict"]["alphas_cumprod"] * 0.9
    with pytest.raises(RuntimeError):
        BoxPolicy.load_celoc_paper(ck)


def test_null_text_zeroes_text_part_of_condition():
    from ce_localization.models.box_policy import BoxPolicy
    m = BoxPolicy(pretrained_backbone=False, num_timesteps=20, vision="r18_paper", in_channels=4).eval()
    x, t = torch.randn(1, 4, 128, 128), torch.randn(1, 512)
    with torch.no_grad():
        c, c0 = m.condition(x, t, None), m.condition(x, t, None, null_text=True)
    assert torch.equal(c[:, :128], c0[:, :128]) and c0[:, 128:].abs().max() == 0 and c[:, 128:].abs().max() > 0


def test_forward_c5_is_layer4_output():
    m = ResNet50FPN(pretrained=False).eval()
    x = torch.randn(1, 3, 128, 96)
    with torch.no_grad():
        c5 = m.forward_c5(x)
        ref = m.layer4(m.layer3(m.layer2(m.layer1(m.stem(x)))))
    assert c5.shape == (1, 2048, 4, 3) and torch.equal(c5, ref)


# ----------------------------------------------------------------------------- BoxRefiner (GAMMA1)

def test_backbone_forward_with_c5_matches_forward_and_c5():
    torch.manual_seed(0)
    m = ResNet50FPN(pretrained=False, in_channels=4).eval()
    x = torch.randn(1, 4, 128, 128)
    f, c5 = m.forward_with_c5(x)
    assert torch.equal(c5, m.forward_c5(x)) and c5.shape == (1, 2048, 4, 4)
    for k, v in m(x).items():
        assert torch.equal(f[k], v), k


def test_noisy_to_boxes_clamps_and_keeps_min_size():
    from ce_localization.models.box_refiner import MIN_WH, noisy_to_boxes
    whwh = torch.tensor([[200.0, 100.0, 200.0, 100.0]])
    b = noisy_to_boxes(torch.tensor([[0.0, 0.0, -3.0, 5.0]]), whwh)       # w <= 0 sau kẹp -> MIN_WH; h kẹp 1
    w, h = b[0, 2] - b[0, 0], b[0, 3] - b[0, 1]
    assert torch.isclose(w, torch.tensor(MIN_WH * 200)) and torch.isclose(h, torch.tensor(100.0))
    assert torch.allclose((b[0, 0] + b[0, 2]) / 2, torch.tensor(100.0))


def test_paired_giou_matches_matrix_diagonal():
    from ce_localization.models.box_refiner import paired_giou
    from ce_localization.utils.box_ops import generalized_box_iou
    torch.manual_seed(0)
    a = torch.rand(6, 2) * 50
    a = torch.cat([a, a + torch.rand(6, 2) * 40 + 1], 1)
    b = torch.rand(6, 2) * 50
    b = torch.cat([b, b + torch.rand(6, 2) * 40 + 1], 1)
    assert torch.allclose(paired_giou(a, b), torch.diagonal(generalized_box_iou(a, b)), atol=1e-6)


@pytest.mark.parametrize("in_ch,tok", [(3, "roi"), (4, "roi"), (4, "coords")])
def test_box_refiner_loss_every_stage_trains_all_parts(in_ch, tok):
    from ce_localization.models.box_refiner import BoxRefiner
    torch.manual_seed(0)
    m = BoxRefiner(in_channels=in_ch, pretrained_backbone=False, num_timesteps=20, box_token=tok)
    assert not any("self_attn" in n for n, _ in m.named_parameters())      # không self-attention
    x = torch.randn(2, in_ch, 128, 128)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])
    tgt = torch.tensor([[10.0, 20.0, 50.0, 60.0], [60.0, 70.0, 100.0, 120.0]])
    loss, st = m(x, torch.randn(2, 512), torch.tensor([[96, 128], [128, 128]]), tgt, whwh, k=3)
    assert st["loss_per_stage"].shape == (6,) and torch.isclose(loss, st["loss_per_stage"].sum())
    loss.backward()
    g = {n: p.grad for n, p in m.named_parameters()}
    names = ["backbone.stem.0.weight", "backbone.layer4.2.conv3.weight", "memory.ss_proj.weight",
             "memory.text_proj.weight", "memory.encoder.0.weight"] + (["backbone.fpn.layer_blocks.0.0.weight"] if tok == "roi" else [])
    proj = "roi_proj.weight" if tok == "roi" else "box_proj.weight"
    names += [f"head.stages.{i}.{p}" for i in range(6) for p in (proj, "cross_attn.in_proj_weight",
                                                                 "ffn.0.weight", "bboxes_delta.weight")]
    for n in names:
        assert g[n] is not None and g[n].abs().sum() > 0, n
    if tok == "coords":                                                 # GAMMA1.1: không RoI, FPN không dùng
        assert not any("roi_proj" in n for n in g) and g["backbone.fpn.layer_blocks.0.0.weight"] is None
        assert m.head.stages[0].box_proj.in_features == 4


@pytest.mark.parametrize("tok", ["roi", "coords"])
def test_box_refiner_samples_are_independent(tok):
    """Không self-attn: box của một mẫu không đổi khi có / không có mẫu khác cùng ảnh."""
    from ce_localization.models.box_refiner import BoxRefiner
    torch.manual_seed(0)
    m = BoxRefiner(pretrained_backbone=False, num_timesteps=20, box_token=tok).eval()
    x, vhw = torch.randn(1, 3, 128, 128), torch.tensor([[128, 128]])
    with torch.no_grad():
        feats, vis = m.encode_image(x, vhw)
        boxes = torch.tensor([[[10.0, 10.0, 40.0, 50.0], [60.0, 30.0, 120.0, 90.0], [5.0, 70.0, 30.0, 100.0]]])
        t = torch.tensor([3, 11, 19])
        text = torch.randn(1, 512)
        norm = torch.full((3, 4), 128.0)
        all_, _ = m.refine(feats, vis, text, t, boxes, norm=norm)
        for j in range(3):
            one, _ = m.refine(feats, vis, text, t[j:j + 1], boxes[:, j:j + 1], norm=norm[j:j + 1])
            assert torch.allclose(one[:, 0], all_[:, j], rtol=1e-4, atol=1e-3), j


def test_box_refiner_sample_steps_and_attention():
    from ce_localization.models.box_refiner import BoxRefiner
    from ce_localization.models.box_policy import unit_to_boxes
    torch.manual_seed(0)
    m = BoxRefiner(in_channels=4, pretrained_backbone=False, num_timesteps=20).eval()
    x, vhw = torch.randn(2, 4, 128, 128), torch.tensor([[96, 128], [128, 128]])
    m.track_attn = True
    for steps in (1, 3):
        g = torch.Generator().manual_seed(1)
        u, stages = m.sample(x, torch.randn(2, 512), vhw, 5, generator=g, steps=steps, return_stages=True)
        assert u.shape == (2, 5, 4) and torch.isfinite(u).all()
        assert len(stages) == steps and stages[0]["stages"].shape == (6, 2, 5, 4) and stages[0]["noisy"].shape == (2, 5, 4)
        assert stages[0]["t"] == 19
        b = unit_to_boxes(u, torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])[:, None])
        assert torch.allclose(b, stages[-1]["stages"][-1], rtol=1e-4, atol=1e-2)   # bước cuối: box tầng cuối, KHÔNG kẹp
    g = torch.Generator().manual_seed(1)
    u, st = m.sample(x, torch.randn(2, 512), vhw, 3, generator=g, steps=4, t_start=9, return_stages=True)
    assert [d["t"] for d in st] == [9, 6, 4, 1] and u.shape == (2, 3, 4)                    # DDIM từ t = 9
    g = torch.Generator().manual_seed(1)
    u, st = m.sample(x, torch.randn(2, 512), vhw, 3, generator=g, steps=10, sampler="mock", return_stages=True)
    assert [d["t"] for d in st] == list(range(9, -1, -1)) and torch.isfinite(u).all()
    assert torch.equal(st[0]["x"], torch.randn(6, 4, generator=torch.Generator().manual_seed(1)).view(2, 3, 4))
    with pytest.raises(ValueError):
        m.sample(x, torch.randn(2, 512), vhw, 3, steps=30, sampler="mock")
    att = m.pop_attn()
    assert len(att) == 6 and all(abs(sum(a.values()) - 1) < 1e-4 for a in att) and set(att[0]) == {"t", "text", "vis"}
    assert m.pop_attn() is None


# ----------------------------------------------------------------------------- "CE-Loc gốc + R-50" (GAMMA)

def test_backbone_bn_train_and_rgb_mean_density_init():
    torch.manual_seed(0)
    m = ResNet50FPN(pretrained=False, in_channels=4, norm="bn", density_init="rgb_mean")
    assert not any(isinstance(x, FrozenBatchNorm2d) for x in m.modules())
    assert sum(isinstance(x, nn.BatchNorm2d) for x in m.modules()) == 53
    w = m.stem[0].weight
    assert torch.allclose(w[:, 3:], w[:, :3].mean(dim=1, keepdim=True))
    with pytest.raises(ValueError):
        ResNet50FPN(pretrained=False, norm="gn")


def test_paper_keypoints_and_canvas_norm():
    from ce_localization.models.box_policy import _paper_spatial_softmax, norm_whwh, spatial_keypoints
    f = torch.randn(2, 6, 4, 4)
    vhw = torch.tensor([[96, 128], [128, 128]])
    assert torch.equal(spatial_keypoints(f, vhw, "paper"), _paper_spatial_softmax(f))      # không mask: bỏ qua valid_hw
    assert spatial_keypoints(f, vhw, "masked").shape == (2, 12)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]])
    assert torch.equal(norm_whwh(nn.Module(), whwh, 128), whwh)
    m = nn.Module()
    m.box_norm = "canvas"
    assert torch.equal(norm_whwh(m, whwh, 128), torch.full((1, 4), 128.0))


@pytest.mark.parametrize("arch", ["box_policy", "box_refiner"])
def test_gamma_paper_keys_build_train_sample(arch):
    """Config GAMMA0 / GAMMA1 thật (khoá "CE-Loc gốc + R-50") dựng được, BN train, loss lùi được, sample hữu hạn."""
    import yaml
    from ce_localization.models.detector import build_model
    from tests.ce_localization.helpers import CFG_G
    with open(CFG_G["density" if arch == "box_policy" else "refiner"]) as f:
        cfg = yaml.safe_load(f)
    cfg["diffusion"]["num_timesteps"] = 20
    torch.manual_seed(0)
    m = build_model(cfg, pretrained_backbone=False)
    assert m.box_norm == "canvas" and any(isinstance(x, nn.BatchNorm2d) for x in m.modules())
    x, vhw = torch.rand(2, 4, 128, 128), torch.tensor([[96, 128], [128, 128]])
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])
    tgt = torch.tensor([[10.0, 20.0, 50.0, 60.0], [60.0, 70.0, 100.0, 120.0]])
    if arch == "box_policy":
        from ce_localization.models.box_policy import boxes_to_unit, norm_whwh
        loss = m(x, torch.randn(2, 512), vhw, boxes_to_unit(tgt, norm_whwh(m, whwh, 128)), k=1)
    else:
        loss, _ = m(x, torch.randn(2, 512), vhw, tgt, whwh, k=1)
    loss.backward()
    assert m.backbone.layer4[2].conv3.weight.grad.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        s = m.sample(x, torch.randn(2, 512), vhw, 3) if arch == "box_policy" else m.sample(x, torch.randn(2, 512), vhw, 3, steps=2)
    assert s.shape == (2, 3, 4) and torch.isfinite(s).all()


# ----------------------------------------------------------------------------- GAMMA2: CE-Loc ResNet18 + refine DiffusionDet

def test_paper_encoder_masked_spatial_softmax_ignores_padding():
    from ce_localization.models.box_policy import PaperVisionEncoder, _paper_spatial_softmax
    torch.manual_seed(0)
    enc = PaperVisionEncoder(128, 4, pretrained=False, ss_mask=True).eval()
    x = torch.rand(2, 4, 128, 128)
    c = enc.features(x)
    assert [f.shape[1] for f in c] == [64, 128, 256, 512] and c[-1].shape[-1] == 4
    assert torch.equal(c[-1], enc.backbone(x))
    vhw = torch.tensor([[64, 128], [128, 128]])                           # ảnh 0: nửa dưới là phần đệm
    with torch.no_grad():
        xy, att = enc.keypoints(x, vhw)
        assert float(att[0, :, 2:].abs().sum()) == 0 and torch.allclose(att[1].sum((1, 2)), torch.ones(512))
        assert torch.equal(enc.keypoints_flat(c[-1], vhw)[1], _paper_spatial_softmax(c[-1])[1])   # không đệm: như bài
    enc.ss_mask = False
    assert torch.equal(enc.keypoints_flat(c[-1], vhw), _paper_spatial_softmax(c[-1]))


def test_box_policy_r18_trainable_with_mask():
    from ce_localization.models.box_policy import BoxPolicy
    torch.manual_seed(0)
    m = BoxPolicy(in_channels=4, pretrained_backbone=False, num_timesteps=20, vision="r18_paper", ss_mask=True)
    assert m.box_norm == "canvas" and m.vision.ss_mask
    x, vhw = torch.rand(2, 4, 128, 128), torch.tensor([[64, 128], [128, 96]])
    loss = m(x, torch.randn(2, 512), vhw, torch.rand(2, 4) * 2 - 1, k=1)
    loss.backward()
    assert m.vision.backbone[0].weight.grad.abs().sum() > 0 and m.vision.projection.weight.grad.abs().sum() > 0
    # MỌI tham số có grad => train.py tắt DDP find_unused_parameters, dùng gradient_as_bucket_view (`ddp_find_unused`)
    assert [n for n, p in m.named_parameters() if p.grad is None] == []
    with torch.no_grad():
        s = m.sample(x, torch.randn(2, 512), vhw, 3)
        kp = m.vision.keypoints_flat(m.vision.backbone(x), vhw)
        g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
        a = m.sample(x, torch.randn(2, 512, generator=torch.Generator().manual_seed(1)), vhw, 3, generator=g1)
        b = m.sample_from_cond(m.cond_from_keypoints(kp, torch.randn(2, 512, generator=torch.Generator().manual_seed(1))),
                               3, g2)
    assert s.shape == (2, 3, 4) and torch.allclose(a, b, atol=1e-5)


def test_ddp_find_unused_only_off_for_paper_r18():
    """Tắt find_unused_parameters: CE-Loc gốc ResNet18 và propose_refine (GAMMA2 / 2.1 — mọi tham số train luôn có grad);
    GAMMA0 / 0.1 (R-50: FPN không dùng) và GAMMA1 / 1.1 (box_refiner) giữ bật như lúc đã chạy; GAMMA2.1 bỏ ε-MSE CE-Loc thì U-Net không có grad -> bật."""
    import yaml
    from ce_localization.train import ddp_find_unused
    from tests.ce_localization.helpers import CFG_G
    on = {k: ddp_find_unused(yaml.safe_load(open(p))) for k, p in CFG_G.items()}
    assert on == {k: k in ("density", "rgb", "refiner", "refiner_coords") for k in CFG_G}, on
    joint = yaml.safe_load(open(CFG_G["pr_joint"]))
    joint["loss"]["proposer_weight"] = 0.0
    assert ddp_find_unused(joint)


def test_roi_pooler_empty_levels_still_get_grad():
    """Mọi RoI rơi vào MỘT tầng: các tầng khác vẫn vào đồ thị (grad 0) — điều kiện để DDP chạy không find_unused."""
    from ce_localization.models.roi import MultiLevelRoIAlign
    feats = [torch.randn(2, 8, 128 // s, 128 // s, requires_grad=True) for s in (4, 8, 16, 32)]
    boxes = torch.tensor([[[10.0, 10.0, 20.0, 20.0]] * 3, [[30.0, 30.0, 42.0, 40.0]] * 3])     # nhỏ -> chỉ P2
    out = MultiLevelRoIAlign(7)(feats, boxes)
    out.sum().backward()
    assert all(f.grad is not None for f in feats)
    assert feats[0].grad.abs().sum() > 0 and all(float(f.grad.abs().sum()) == 0 for f in feats[1:])


def test_dynamic_conv_matches_diffusiondet_formula():
    from ce_localization.models.dynamic_head import DynamicConv
    torch.manual_seed(0)
    dc = DynamicConv(16, 4, 2, 3).eval()
    pro, roi = torch.randn(1, 5, 16), torch.randn(9, 5, 16)
    out = dc(pro, roi)
    p = dc.dynamic_layer(pro).permute(1, 0, 2)
    p1, p2 = p[:, :, :64].reshape(5, 16, 4), p[:, :, 64:].reshape(5, 4, 16)
    f = roi.permute(1, 0, 2)
    f = torch.relu(dc.norm1(f @ p1))
    f = torch.relu(dc.norm2(f @ p2))
    ref = torch.relu(dc.norm3(dc.out_layer(f.flatten(1))))
    assert out.shape == (5, 16) and torch.allclose(out, ref, atol=1e-6)


def _propose_refine(freeze, geo=False, relation=False, relation_k=32):
    from ce_localization.models.propose_refine import ProposeRefine
    pk = dict(in_channels=4, pretrained_backbone=False, num_timesteps=20, vision="r18_paper", ss_mask=True)
    torch.manual_seed(0)
    return ProposeRefine(pk, d_model=64, dim_feedforward=128, nhead=4, dim_dynamic=8, num_timesteps=20,
                         freeze_proposer=freeze, geo=geo, geo_hidden=32, relation=relation, relation_k=relation_k)


@pytest.mark.parametrize("freeze", [True, False])
def test_propose_refine_freeze_and_joint_loss(freeze):
    m = _propose_refine(freeze).train()
    assert m.proposer.training is not freeze                              # đóng băng: BN của CE-Loc ở eval
    before = {k: v.clone() for k, v in m.proposer.state_dict().items()}
    x, vhw = torch.rand(2, 4, 128, 128), torch.tensor([[96, 128], [128, 128]])
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])
    tgt = torch.tensor([[10.0, 20.0, 50.0, 60.0], [60.0, 70.0, 100.0, 120.0]])
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    loss, st = m(x, torch.randn(2, 512), vhw, tgt, whwh, k=3)
    assert st["loss_per_stage"].shape == (6,) and ("loss_eps" in st) is not freeze              # đóng băng: không báo ε-MSE
    assert freeze or float(st["loss_eps"]) > 0
    loss.backward()
    g = {n: p.grad for n, p in m.named_parameters()}
    for n in ("fpn.layer_blocks.0.0.weight", "memory.ss_proj.weight", "head.stages.0.inst_interact.dynamic_layer.weight",
              "head.stages.5.cross_attn.in_proj_weight", "head.time_mlp.1.weight"):
        assert g[n] is not None and g[n].abs().sum() > 0, n
    pg = g["proposer.vision.backbone.0.weight"]
    if freeze:
        assert pg is None and not any(p.requires_grad for p in m.proposer.parameters())
    else:
        assert pg.abs().sum() > 0 and g["proposer.noise_net.final_conv.1.weight"].abs().sum() > 0
    opt.step()
    same = all(torch.equal(before[k], v) for k, v in m.proposer.state_dict().items())
    assert same is freeze                                                 # đóng băng: kể cả thống kê BN không đổi


def test_propose_refine_head_stays_fp32_under_autocast():
    """`training.amp`: autocast hạ backbone + FPN xuống nửa chính xác, nhưng memory + head refine (kiểu DiffusionDet) chạy
    fp32 — box mọi stage và loss là fp32, backward qua được. CPU dùng bf16 thay cho fp16 CUDA (cùng cơ chế autocast)."""
    m = _propose_refine(False).train()
    x, vhw = torch.rand(2, 4, 128, 128), torch.tensor([[96, 128], [128, 128]])
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])
    tgt = torch.tensor([[10.0, 20.0, 50.0, 60.0], [60.0, 70.0, 100.0, 120.0]])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        feats, kp = m.encode(x, vhw)
        assert feats[0].dtype == torch.bfloat16                           # autocast có tới backbone + FPN
        boxes = tgt[:, None].repeat(1, 3, 1)
        preds, att = m.refine(feats, m.vis_token(kp), torch.randn(2, 512), torch.randint(0, 1000, (6,)), boxes, True)
        assert preds.dtype == torch.float32 and att.dtype == torch.float32
        loss, st = m(x, torch.randn(2, 512), vhw, tgt, whwh, k=3)
    assert loss.dtype == torch.float32 and torch.isfinite(loss) and st["loss_per_stage"].dtype == torch.float32
    loss.backward()
    assert m.head.stages[0].bboxes_delta.weight.grad.dtype == torch.float32


def test_propose_refine_sample_variants():
    m = _propose_refine(True).eval()
    x, vhw = torch.rand(2, 4, 128, 128), torch.tensor([[96, 128], [128, 128]])
    text = torch.randn(2, 512)
    g = torch.Generator().manual_seed(0)
    ce, r5, rn = m.sample_variants(x, text, vhw, 3, g, [{"refine_t": None}, {"refine_t": 5, "refine_steps": 2},
                                                        {"refine_t": "noise"}])
    assert ce.shape == r5.shape == rn.shape == (2, 3, 4) and torch.isfinite(r5).all() and torch.isfinite(rn).all()
    # biến thể none = đúng box CE-Loc (đổi từ chuẩn hoá canvas sang chuẩn hoá vùng thật)
    from ce_localization.models.box_policy import boxes_to_unit, unit_to_boxes
    g2 = torch.Generator().manual_seed(0)
    with torch.no_grad():
        u = m.proposer.sample(x, text, vhw, 3, generator=g2)
    canvas = torch.full((4,), 128.0)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])[:, None]
    assert torch.allclose(ce, boxes_to_unit(unit_to_boxes(u, canvas), whwh), atol=1e-4)
    m.track_attn = True
    m.sample(x, text, vhw, 2, refine_t=5)
    att = m.pop_attn()
    assert len(att) == 6 and abs(sum(att[0].values()) - 1) < 1e-4


# ----------------------------------------------------------------------------- GAMMA3: FiLM theo box vật đang có

def test_geo_features_geometry():
    """Box trùng vật 1, vật 2 cách 2 bề rộng sang phải cùng cỡ: đúng từng thành phần (models/geo.py); thiếu vật -> 0, cờ 0."""
    import math
    from ce_localization.models.geo import GEO_DIM, GEO_K, geo_features, pad_objects
    box = torch.tensor([[10.0, 10.0, 30.0, 50.0]])
    objs, mask = pad_objects([torch.tensor([[10.0, 10.0, 30.0, 50.0], [50.0, 10.0, 70.0, 50.0]])], "cpu")
    assert objs.shape == (1, GEO_K, 4) and mask.tolist() == [[True, True, False, False]]
    f = geo_features(box, objs, mask)
    assert f.shape == (1, GEO_DIM) and torch.isfinite(f).all()
    assert torch.isclose(f[0, 0], torch.tensor(1.0)) and torch.isclose(f[0, 1], torch.tensor(math.log(2)))  # max IoU, log(1+ΣIoA)
    near = f[0, 2:2 + GEO_K * 7].view(GEO_K, 7)
    assert torch.allclose(near[0], torch.tensor([0, 0, 0, 0, 1, 1, 1.0]))                  # chính nó: IoU = IoA = 1
    assert torch.allclose(near[1], torch.tensor([math.log(3), 0, 0, 0, 0, 0, 1.0]), atol=1e-6)   # Δx/w = 2 -> slog = log 3
    assert (near[2:] == 0).all()
    tail = f[0, 2 + GEO_K * 7:]
    assert torch.allclose(tail, torch.tensor([0.0, 0.0, math.log(2), math.log(3)]), atol=1e-6)   # trung vị cỡ, gần (|Δ| < 2), số vật


def test_geo_features_ignore_padding_and_order():
    from ce_localization.models.geo import geo_features, pad_objects
    g = torch.Generator().manual_seed(0)
    lt = torch.rand(9, 2, generator=g) * 100
    o = torch.cat([lt, lt + torch.rand(9, 2, generator=g) * 30 + 1], 1)
    boxes = torch.tensor([[20.0, 20.0, 45.0, 60.0], [0.0, 0.0, 3.0, 2.0], [50.0, 50.0, 140.0, 130.0]])
    objs, mask = pad_objects([o], "cpu")
    ref = geo_features(boxes, objs.expand(3, -1, -1), mask.expand(3, -1))
    junk = torch.cat([objs, torch.rand(1, 5, 4) * 500], 1)                                   # ô đệm rác, mask False
    jm = torch.cat([mask, torch.zeros(1, 5, dtype=torch.bool)], 1)
    assert torch.allclose(geo_features(boxes, junk.expand(3, -1, -1), jm.expand(3, -1)), ref, atol=1e-6)
    perm = torch.randperm(9, generator=g)
    po, pm = pad_objects([o[perm]], "cpu")
    assert torch.allclose(geo_features(boxes, po.expand(3, -1, -1), pm.expand(3, -1)), ref, atol=1e-6)
    eo, em = pad_objects([torch.zeros(0, 4)], "cpu")                                         # không vật nào
    e = geo_features(boxes, eo.expand(3, -1, -1), em.expand(3, -1))
    assert torch.isfinite(e).all() and (e == 0).all()


def test_propose_refine_geo_zero_init_then_used():
    """Lớp cuối MLP geo khởi tạo 0 => lúc đầu box ra y hệt khi tắt geo (= GAMMA2); có weight thì box đổi; train có grad tới
    nhánh geo; model.geo mà không truyền vật thì báo lỗi."""
    m = _propose_refine(True, geo=True).eval()
    x, vhw = torch.rand(2, 4, 128, 128), torch.tensor([[96, 128], [128, 128]])
    text = torch.randn(2, 512)
    objects = [torch.tensor([[5.0, 5.0, 30.0, 40.0], [60.0, 50.0, 90.0, 90.0]]), torch.zeros(0, 4)]
    with pytest.raises(ValueError, match="objects"):
        m.sample(x, text, vhw, 3, refine_t=5)

    def run(v):
        return m.sample_variants(x, text, vhw, 3, torch.Generator().manual_seed(0), [v], objects=objects)[0]
    off = run({"refine_t": 5, "geo": False})
    assert torch.allclose(run({"refine_t": 5}), off, atol=1e-6)
    for f in m.head.geo_films:
        torch.nn.init.normal_(f[-1].weight, std=0.5)
    assert not torch.allclose(run({"refine_t": 5}), off, atol=1e-4)
    assert torch.allclose(run({"refine_t": 5, "geo": False}), off, atol=1e-6)                 # tắt geo: không phụ thuộc weight geo

    m = _propose_refine(True, geo=True).train()
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])
    tgt = torch.tensor([[10.0, 20.0, 50.0, 60.0], [60.0, 70.0, 100.0, 120.0]])
    loss, _ = m(x, text, vhw, tgt, whwh, k=3, objects=objects)
    loss.backward()
    for i in range(6):
        assert m.head.geo_films[i][-1].weight.grad.abs().sum() > 0, i


# ----------------------------------------------------------------------------- GAMMA3.1: attention tới vật kiểu Relation-DETR

def test_relation_encoding_matches_relation_detr_formula():
    """e(b, i) = [log(|Δx|/w + 1), log(|Δy|/h + 1), log(w/wᵢ), log(h/hᵢ)] (paper công thức (2)); sine như get_sine_pos_embed."""
    import math
    from ce_localization.models.relation import PositionRelationEmbedding, box_rel_encoding, sine_embed
    e = box_rel_encoding(torch.tensor([100.0, 100.0, 40.0, 60.0]), torch.tensor([20.0, 100.0, 60.0, 90.0]))
    assert torch.allclose(e, torch.tensor([math.log(3), 0.0, math.log(40 / 60), math.log(60 / 90)]), atol=1e-5)
    # dim_t = 10000^(2j/4) = [1, 100] ; x·scale/dim_t = [50, 0.5] ; sin / cos xen kẽ
    se = sine_embed(torch.tensor([0.5]), 4, 10000.0, 100.0)
    assert torch.allclose(se, torch.tensor([math.sin(50), math.cos(50), math.sin(0.5), math.cos(0.5)]), atol=1e-6)
    assert sine_embed(torch.rand(3, 5, 4), 16, scale=100.0).shape == (3, 5, 64)
    pe = PositionRelationEmbedding(16, 4)
    src = (torch.rand(6, 4) * 50 + 1).requires_grad_()
    r = pe(src, torch.rand(6, 4) * 50 + 1)
    assert r.shape == (6, 4) and (r >= 0).all()
    r.sum().backward()
    assert src.grad is None and pe.pos_proj.weight.grad is not None        # phần mã hoá dưới no_grad như bài


def _rel_inputs():
    x, vhw = torch.rand(2, 4, 128, 128), torch.tensor([[96, 128], [128, 128]])
    return x, vhw, torch.randn(2, 512)


def test_object_relation_uses_near_objects_only():
    """Mức module: chỉ k vật có tâm gần box nhất vào attention — đổi vật xa -> y nguyên, đổi vật gần -> đổi; thứ tự vật không quan
    trọng; bỏ Rel -> đổi; không vật nào -> chỉ LayerNorm (cộng 0)."""
    from ce_localization.models.relation import ObjectRelation, relation_attend
    torch.manual_seed(0)
    sh = ObjectRelation(d_model=32, n_head=4, k_near=2)
    attn, norm = nn.MultiheadAttention(32, 4, batch_first=True), nn.LayerNorm(32)
    W = torch.randn(4, 32)
    box = torch.tensor([[37.5, 37.5, 67.5, 67.5]])
    pro = torch.randn(1, 32)
    whwh = torch.tensor([[128.0, 128.0, 128.0, 128.0]])

    def run(objs, use_bias=True, mask=None):
        o = torch.tensor(objs)[None]
        rel = {"objs": o, "mask": torch.ones(o.shape[:2], dtype=torch.bool) if mask is None else mask,
               "img": torch.zeros(1, dtype=torch.long), "whwh": whwh, "whwh_img": whwh}
        rel = {**rel, "feat": torch.tanh(o / 64 @ W), "pos": sh.pos(o, whwh[:, None])}   # feature theo chính box vật
        return relation_attend(attn, norm, pro, box, whwh, sh, sh.gather(box, rel), use_bias)
    near = [[30.0, 30.0, 60.0, 70.0], [50.0, 40.0, 80.0, 75.0]]
    far = [[0.0, 0.0, 4.0, 4.0], [120.0, 0.0, 127.0, 6.0], [0.0, 90.0, 5.0, 95.0]]
    ref = run(near + far)
    assert torch.allclose(run(near + [[0.0, 0.0, 2.0, 9.0]] + far[1:]), ref, atol=1e-6)
    assert torch.allclose(run(far[::-1] + near[::-1]), ref, atol=1e-6)
    assert not torch.allclose(run([[30.0, 30.0, 40.0, 40.0]] + near[1:] + far), ref, atol=1e-4)
    assert not torch.allclose(run(near + far, use_bias=False), ref, atol=1e-4)
    assert torch.allclose(run([[0.0, 0.0, 1.0, 1.0]], mask=torch.zeros(1, 1, dtype=torch.bool)), norm(pro), atol=1e-6)


def test_propose_refine_relation_variants():
    """`geo: False` (bỏ bước attention tới vật) và `rel_bias: False` (bỏ Rel) đều đổi box ra; ảnh không có vật -> hữu hạn."""
    m = _propose_refine(True, relation=True).eval()
    x, vhw, text = _rel_inputs()
    objects = [torch.tensor([[5.0, 5.0, 30.0, 40.0], [60.0, 50.0, 90.0, 90.0], [20.0, 60.0, 50.0, 90.0]]), torch.zeros(0, 4)]

    def run(v):
        return m.sample_variants(x, text, vhw, 3, torch.Generator().manual_seed(0), [v], objects=objects)[0]
    ref = run({"refine_t": 5})
    assert torch.isfinite(ref).all()
    assert not torch.allclose(run({"refine_t": 5, "geo": False})[0], ref[0], atol=1e-4)
    assert not torch.allclose(run({"refine_t": 5, "rel_bias": False})[0], ref[0], atol=1e-4)
    assert torch.allclose(run({"refine_t": 5}), ref)


def test_propose_refine_relation_trains():
    m = _propose_refine(True, relation=True).train()
    x, vhw, text = _rel_inputs()
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0], [128.0, 128.0, 128.0, 128.0]])
    tgt = torch.tensor([[10.0, 20.0, 50.0, 60.0], [60.0, 70.0, 100.0, 120.0]])
    with pytest.raises(ValueError, match="objects"):
        m(x, text, vhw, tgt, whwh, k=3)
    objects = [torch.tensor([[5.0, 5.0, 30.0, 40.0], [60.0, 50.0, 90.0, 90.0]]), torch.zeros(0, 4)]
    loss, _ = m(x, text, vhw, tgt, whwh, k=3, objects=objects)
    assert torch.isfinite(loss)
    loss.backward()
    g = {n: p.grad for n, p in m.named_parameters()}
    for n in ("head.stages.0.rel_attn.in_proj_weight", "head.stages.5.rel_attn.out_proj.weight", "head.relation.rel_embed.pos_proj.weight",
              "head.relation.ref_point_head.0.weight", "fpn.layer_blocks.0.0.weight"):
        assert g[n] is not None and g[n].abs().sum() > 0, n
