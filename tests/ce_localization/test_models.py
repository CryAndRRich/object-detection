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


@pytest.mark.parametrize("in_ch", [3, 4])
def test_box_policy_loss_trains_backbone_cond_and_unet(in_ch):
    from ce_localization.models.box_policy import BoxPolicy
    torch.manual_seed(0)
    m = BoxPolicy(in_channels=in_ch, pretrained_backbone=False, num_timesteps=20)
    if in_ch == 4:
        assert m.backbone.stem[0].weight[:, 3].abs().max() == 0          # kênh density khởi tạo 0 (như ALPHA3)
    x = torch.randn(2, in_ch, 128, 128)
    loss = m(x, torch.randn(2, 512), torch.tensor([[96, 128], [128, 128]]), torch.rand(2, 4) * 2 - 1, k=3)
    loss.backward()
    g = {n: p.grad for n, p in m.named_parameters()}
    for n in ("backbone.layer4.2.conv3.weight", "backbone.fpn.layer_blocks.3.0.weight", "vis_proj.weight",
              "text_proj.0.weight", "noise_net.final_conv.1.weight", "noise_net.diffusion_step_encoder.1.weight"):
        assert g[n] is not None and g[n].abs().sum() > 0, n
    assert g["backbone.fpn.layer_blocks.0.0.weight"] is None             # P2..P4 không dùng
    assert m.alphas_cumprod.shape == (20,)
    with torch.no_grad():
        s = m.sample(x, torch.randn(2, 512), torch.tensor([[96, 128], [128, 128]]), 4)
    assert s.shape == (2, 4, 4) and torch.isfinite(s).all()

