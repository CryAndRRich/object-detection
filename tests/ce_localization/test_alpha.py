"""EXPERIMENT ALPHA — cửa G1 (docs/EXPERIMENT_ALPHA.md mục 8): công thức từng phần + TRỌN LUỒNG
train -> dừng -> --resume -> eval trên CE-130 giả, và DDP 2 tiến trình (gloo, CPU).

Không tải gì: backbone `pretrained_backbone: false`, text thay bằng embedding giả.
Mọi lượt train / eval trong test ép `--device cpu`: trên GPU, backward của conv (cuDNN) và
`roi_align` không tất định (resume lệch ~1e-5 so với train liền) và GPU server dùng chung có thể
hết bộ nhớ (đã gặp 2026-09-29).
"""

import json
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision.ops import roi_align
from torchvision.ops.misc import FrozenBatchNorm2d

from ce_localization.alpha import criterion as C
from ce_localization.alpha.backbone import ResNet50FPN
from ce_localization.alpha.data import AlphaCE130, collate, letterbox, scale_boxes
from ce_localization.alpha.diffusion import cosine_alphas_cumprod, ddim_sample, prepare_train_boxes
from ce_localization.alpha.head import SCALE_CLAMP, apply_deltas
from ce_localization.alpha.memory import (MemoryEncoder, masked_spatial_softmax, sine_pos_2d,
                                          valid_cells_mask)
from ce_localization.alpha.model import AlphaDetector
from ce_localization.alpha.roi import MultiLevelRoIAlign, assign_levels
from ce_localization.alpha.text import TextTable
from ce_localization.alpha.train_utils import epoch_batches, warmup_multistep

CFG0 = os.path.join(os.path.dirname(__file__), "..", "..", "ce_localization", "config", "alpha0.yaml")


# ----------------------------------------------------------------------------- dữ liệu giả

def _make_branch(root, split, iid, boxes_xyxy, category, w=200, h=150, seed=0):
    br = os.path.join(root, split, f"{iid}_b1")
    os.makedirs(br, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(
        os.path.join(br, "ground_truth.jpg"))
    with open(os.path.join(br, "annotation.json"), "w") as f:
        json.dump({"all_bboxes": boxes_xyxy, "inpainted_bboxes": boxes_xyxy[:1],
                   "class_based_caption": category}, f)
    return br


def _fake_ce130(root, n_train=4, n_val=2, n_test=2):
    rng = np.random.default_rng(1)
    k = 0
    for split, n, cats in (("train", n_train, ["apple", "egg"]), ("val", n_val, ["bird"]),
                           ("test", n_test, ["cup"])):
        for i in range(n):
            boxes = []
            for _ in range(int(rng.integers(3, 8))):
                x1, y1 = rng.uniform(0, 150), rng.uniform(0, 110)
                boxes.append([x1, y1, x1 + rng.uniform(8, 45), y1 + rng.uniform(8, 35)])
            _make_branch(root, split, str(1000 + k), boxes, cats[i % len(cats)], seed=k)
            k += 1


def _fake_text_table(names, cfg, dev):
    table = {}
    for n in names:
        g = torch.Generator().manual_seed(sum(map(ord, n)))
        table[n] = torch.randn(512, generator=g)
    return TextTable(table)


def _test_cfg(tmp_path, memory="none", data_root=None):
    with open(CFG0) as f:
        cfg = yaml.safe_load(f)
    cfg["data"].update(root=data_root, image_size=128, num_workers=0)
    cfg["model"].update(memory=memory, pretrained_backbone=False)
    cfg["diffusion"]["num_proposals"] = 20
    cfg["training"].update(max_iter=4, steps=[3], warmup_iters=2, log_every=1, ckpt_every=2,
                           eval_every=2)
    cfg["eval"]["batch_size"] = 2
    p = str(tmp_path / f"cfg_{memory}.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p, cfg


# ----------------------------------------------------------------------------- dữ liệu

def test_letterbox_matches_celoc_original():
    """Canvas + scale y như `resize_and_pad` của CE-Loc gốc (bản viết lại có test nạp strict)."""
    from ce_localization.legacy.celoc_vision import resize_and_pad
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (384, 683, 3), dtype=np.uint8))
    canvas, scale, nw, nh = letterbox(img, 512)
    ref, _, ref_scale = resize_and_pad(img, Image.new("L", img.size, 0), 512)
    assert np.array_equal(canvas, np.asarray(ref)) and scale == ref_scale
    assert (nw, nh) == (512, int(384 * 512 / 683))
    assert canvas[nh:].max() == 0                               # đệm ĐEN ở đáy


def test_scale_boxes_clips_and_drops():
    b = scale_boxes([[10, 10, 20, 20], [190, 10, 250, 20], [210, 5, 230, 9]], 0.5, 100, 75)
    assert np.allclose(b, [[5, 5, 10, 10], [95, 5, 100, 10]])  # box ngoài vùng thật bị bỏ


def test_dataset_item_and_collate(tmp_path):
    root = str(tmp_path / "ce")
    _make_branch(root, "train", "1", [[10, 20, 60, 80], [100, 30, 150, 120]], "egg")
    ds = AlphaCE130(root, "train", 128)
    s = ds[0]
    assert s["image"].shape == (3, 128, 128) and s["valid_hw"] == (96, 128)
    assert torch.allclose(s["boxes"], torch.tensor([[10, 20, 60, 80], [100, 30, 150, 120]]) * 0.64)
    pad = s["image"][:, 96:]                                     # đen sau chuẩn hoá ImageNet
    assert torch.allclose(pad[0], torch.full_like(pad[0], -0.485 / 0.229))
    b = collate([s, s])
    assert b["whwh"].tolist() == [[128, 96, 128, 96]] * 2 and b["valid_hw"].tolist() == [[96, 128]] * 2


# ----------------------------------------------------------------------------- backbone / RoI

def test_backbone_frozen_bn_all_convs_trainable():
    m = ResNet50FPN(pretrained=False)
    assert not any(isinstance(x, nn.BatchNorm2d) for x in m.modules())
    assert sum(isinstance(x, FrozenBatchNorm2d) for x in m.modules()) == 53
    convs = [x for x in m.modules() if isinstance(x, nn.Conv2d)]
    assert convs and all(c.weight.requires_grad for c in convs)       # KHÔNG đóng băng lớp nào
    out = m(torch.randn(1, 3, 128, 128))
    assert [tuple(v.shape[-2:]) for v in out.values()] == [(32, 32), (16, 16), (8, 8), (4, 4)]
    assert all(v.shape[1] == 256 for v in out.values())


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


# ----------------------------------------------------------------------------- khuếch tán

def test_prepare_train_boxes_placeholders_subsample_and_t():
    ac = cosine_alphas_cumprod(1000)
    g = torch.Generator().manual_seed(0)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]] * 3)
    gts = [torch.tensor([[10.0, 10.0, 40.0, 30.0]]), torch.rand(50, 2).repeat(1, 2) * 50 + torch.tensor([0, 0, 5, 5.0]),
           torch.zeros(0, 4)]
    boxes, t = prepare_train_boxes(gts, whwh, 20, ac, 2.0, g)
    assert boxes.shape == (3, 20, 4) and t.shape == (3,) and len(set(t.tolist())) == 3   # mỗi ảnh một t
    # DiffusionDet kẹp CXCYWH về [0,1] (không kẹp xyxy): tâm và cỡ nằm trong vùng thật
    cx, cy = (boxes[..., 0] + boxes[..., 2]) / 2, (boxes[..., 1] + boxes[..., 3]) / 2
    w, h = boxes[..., 2] - boxes[..., 0], boxes[..., 3] - boxes[..., 1]
    assert (cx >= 0).all() and (cx <= 128).all() and (w >= 0).all() and (w <= 128 + 1e-4).all()
    assert (cy >= 0).all() and (cy <= 96).all() and (h >= 0).all() and (h <= 96 + 1e-4).all()
    b0, _ = prepare_train_boxes(gts[:1], whwh[:1], 20, ac, 2.0, torch.Generator().manual_seed(0), t=0)
    assert torch.allclose(b0[0, 0], gts[0][0], atol=1.0)               # t=0: nhiễu < 1 px
    # placeholder: randn/6 + 0.5 -> tâm trung bình ~0.5 vùng thật
    bb, _ = prepare_train_boxes([torch.zeros(0, 4)], whwh[:1], 4000, ac, 2.0,
                                torch.Generator().manual_seed(1), t=0)
    cx = (bb[0, :, 0] + bb[0, :, 2]) / 2 / 128
    assert abs(cx[1:].mean().item() - 0.5) < 0.02 and abs(cx[1:].std().item() - 1 / 6) < 0.02


def test_ddim_sample_shapes_and_renewal():
    ac = cosine_alphas_cumprod(1000)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]])

    def head_fn(boxes, t):
        S, B, N = 3, boxes.shape[0], boxes.shape[1]
        return torch.zeros(S, B, N, 1), boxes[None].repeat(S, 1, 1, 1)

    o1 = ddim_sample(head_fn, 1, 16, whwh, ac, steps=1)
    assert o1["boxes"].shape == (1, 16, 4) and o1["stage_boxes"].shape == (3, 1, 16, 4)
    o4 = ddim_sample(head_fn, 1, 16, whwh, ac, steps=4)                # score 0.5 -> renewal bỏ hết
    assert o4["boxes"].shape == (1, 64, 4) and o4["scores"].shape == (1, 64)   # 4 bước đều vào ensemble
    with pytest.raises(ValueError):
        ddim_sample(head_fn, 2, 16, whwh.repeat(2, 1), ac, steps=4)


# ----------------------------------------------------------------------------- matcher / loss

def _targets(gt, wh=(100.0, 100.0)):
    return C.build_targets([gt], torch.tensor([[wh[0], wh[1], wh[0], wh[1]]]))


def test_simota_matches_obvious_queries_and_dedupes():
    gt = torch.tensor([[10.0, 10.0, 30.0, 30.0], [60.0, 60.0, 90.0, 90.0]])
    pred = torch.tensor([[[10.0, 10.0, 30.0, 30.0], [11.0, 11.0, 31.0, 31.0], [60.0, 60.0, 90.0, 90.0],
                          [0.0, 0.0, 5.0, 5.0], [40.0, 40.0, 50.0, 50.0], [59.0, 61.0, 89.0, 91.0]]])
    logits = torch.zeros(1, 6, 1)
    (sel, gi), = C.match(logits, pred, _targets(gt))
    matched = dict(zip(torch.nonzero(sel).flatten().tolist(), gi.tolist()))
    assert matched[0] == 0 and matched[2] == 1                       # box trùng GT được ghép đúng GT
    assert 3 not in matched and 4 not in matched                     # ngoài vùng tâm -> không ghép
    assert len(gi) == int(sel.sum())                                 # mỗi query tối đa 1 GT


def test_simota_terminates_when_more_gt_than_queries():
    """Ảnh CE-130 có tới 501 box, N=200: vòng cứu phải DỪNG (hành vi gốc)."""
    g = torch.Generator().manual_seed(0)
    xy = torch.rand(12, 2, generator=g) * 80
    gt = torch.cat([xy, xy + 10], 1)
    pred = gt[:5].clone()[None]
    (sel, gi), = C.match(torch.zeros(1, 5, 1), pred, _targets(gt))
    assert int(sel.sum()) == len(gi) <= 5


def test_simota_cap_does_not_change_result_when_enough_queries(monkeypatch):
    """Số GT <= số query: vòng cứu tự dừng trước trần -> kết quả Y HỆT khi gần như không có trần
    (= bản gốc). Lưu ý: GT có thể vắng trong `gi` — hai GT cùng chọn một query không thuộc
    `anchor_matching_gt` cũ thì query giữ cả hai, `max(1)` chỉ trả một (hành vi gốc)."""
    orig = C.dynamic_k_matching
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        xy = torch.rand(15, 2, generator=g) * 80
        gt = torch.cat([xy, xy + 5 + torch.rand(15, 2, generator=g) * 15], 1)
        q = torch.rand(40, 2, generator=g) * 90
        pred = torch.cat([q, q + 3 + torch.rand(40, 2, generator=g) * 20], 1)[None]
        logits = torch.randn(1, 40, 1, generator=g)
        (s1, g1), = C.match(logits, pred, _targets(gt))
        monkeypatch.setattr(C, "dynamic_k_matching", lambda c, i, n, k=5: orig(c, i, n, k, 10 ** 5))
        (s2, g2), = C.match(logits, pred, _targets(gt))
        monkeypatch.setattr(C, "dynamic_k_matching", orig)
        assert torch.equal(s1, s2) and torch.equal(g1, g2), seed


def test_simota_empty_gt():
    (sel, gi), = C.match(torch.zeros(1, 4, 1), torch.rand(1, 4, 4) * 10,
                         C.build_targets([torch.zeros(0, 4)], torch.tensor([[100.0] * 4])))
    assert sel.sum() == 0 and len(gi) == 0


def test_criterion_normalises_by_matched_queries_and_sums_stages():
    cfg_l = {"alpha": 0.25, "gamma": 2.0, "class_weight": 2.0, "l1_weight": 5.0, "giou_weight": 2.0}
    crit = C.AlphaCriterion(cfg_l, {"ota_k": 5, "center_radius": 2.5})
    gt = torch.tensor([[10.0, 10.0, 30.0, 30.0]])
    pred = torch.tensor([[[10.0, 10.0, 30.0, 30.0], [0.0, 0.0, 4.0, 4.0], [80, 80, 99, 99.0]]])
    logits = torch.tensor([[[0.3], [-1.0], [0.5]]])
    tg = _targets(gt)
    loss, st = crit.loss_one(logits, pred, tg)
    (sel, gi), = C.match(logits, pred, tg)
    n = int(sel.sum())
    tcls = sel.float()[None, :, None]
    from ce_localization.models.criterion import sigmoid_focal_loss
    ce = sigmoid_focal_loss(logits.flatten(0, 1), tcls.flatten(0, 1)).sum() / n
    assert torch.allclose(st["loss_ce"], ce) and st["n_matched"] == n
    assert abs(float(st["loss_bbox"])) < 1e-6 and abs(float(st["loss_giou"])) < 1e-6
    tot, st6 = crit(torch.stack([logits] * 6), torch.stack([pred] * 6), tg)
    assert torch.allclose(tot, 6 * loss) and len(st6["loss_per_stage"]) == 6


# ----------------------------------------------------------------------------- memory

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


# ----------------------------------------------------------------------------- model

@pytest.mark.parametrize("kind", ["none", "spatial_softmax", "grid"])
def test_model_all_params_get_grad_and_focal_bias_only_on_class_logits(kind):
    torch.manual_seed(0)
    m = AlphaDetector(memory=kind, pretrained_backbone=False)
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
    crit = C.AlphaCriterion({"alpha": 0.25, "gamma": 2.0, "class_weight": 2.0, "l1_weight": 5.0,
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
    m = AlphaDetector(memory="grid", pretrained_backbone=False).eval()
    img = torch.randn(1, 3, 128, 128)
    whwh = torch.tensor([[128.0, 96.0, 128.0, 96.0]])
    o = m.sample(img, torch.randn(1, 512), torch.tensor([[96, 128]]), whwh, 10, steps=1)
    assert o["boxes"].shape == (1, 10, 4) and o["stage_boxes"].shape == (6, 1, 10, 4)
    o4 = m.sample(img, torch.randn(1, 512), torch.tensor([[96, 128]]), whwh, 10, steps=4)
    assert o4["boxes"].shape[1] == 40


# ----------------------------------------------------------------------------- lịch / batch

def test_warmup_multistep_matches_detectron2():
    assert warmup_multistep(0, [9000, 11000]) == pytest.approx(0.01)
    assert warmup_multistep(500, [9000, 11000]) == pytest.approx(0.01 * 0.5 + 0.5)
    assert warmup_multistep(1000, [9000, 11000]) == 1.0
    assert warmup_multistep(9000, [9000, 11000]) == pytest.approx(0.1)
    assert warmup_multistep(11999, [9000, 11000]) == pytest.approx(0.01)


def test_epoch_batches_disjoint_across_ranks_and_reproducible():
    a = epoch_batches(11, 1, 0, 2, 0, 3)
    b = epoch_batches(11, 1, 1, 2, 0, 3)
    assert len(a) == len(b) == 5
    assert not set(sum(a, [])) & set(sum(b, []))
    assert a == epoch_batches(11, 1, 0, 2, 0, 3) and a != epoch_batches(11, 1, 0, 2, 0, 4)


# ----------------------------------------------------------------------------- trọn luồng

def _run_train(monkeypatch, argv):
    import ce_localization.train_alpha as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    monkeypatch.setattr(sys, "argv", ["train_alpha.py"] + argv + ["--device", "cpu"])
    ta.main()


@pytest.mark.parametrize("kind", ["none", "spatial_softmax", "grid"])
def test_full_flow_train_resume_eval(tmp_path, monkeypatch, kind):
    """train 2 iter -> dừng -> --resume tới 4 iter == train liền 4 iter ; rồi eval ghi JSON."""
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    cfg_path, _ = _test_cfg(tmp_path, kind, root)
    a, b = str(tmp_path / "a"), str(tmp_path / "b")

    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--max-iter", "2"])
    ck = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    assert ck["iter"] == 2 and ck["best"]["iter"] == 2
    with pytest.raises(SystemExit):                                  # last.pth có rồi mà thiếu --resume
        _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", a, "--resume"])
    _run_train(monkeypatch, ["--config", cfg_path, "--save-dir", b])
    ka = torch.load(os.path.join(a, "last.pth"), weights_only=False)
    kb = torch.load(os.path.join(b, "last.pth"), weights_only=False)
    assert ka["iter"] == kb["iter"] == 4
    for k in kb["model"]:
        assert torch.allclose(ka["model"][k].float(), kb["model"][k].float(), atol=1e-5), k
    assert sorted(os.listdir(a)) == ["best.pth", "history.json", "last.pth"]
    with open(os.path.join(a, "last.pth"), "rb") as f:               # pickle, KHÔNG zip (Kaggle)
        assert f.read(2) != b"PK"

    import ce_localization.eval_alpha as ea
    import ce_localization.train_alpha as ta
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(sys, "argv", ["eval_alpha.py", "--ckpt", os.path.join(a, "best.pth"),
                                      "--split", "test", "--nms", "--oracle-score", "--steps", "1", "4",
                                      "--attn-diag", "1", "--batch-size", "2", "--num-workers", "0",
                                      "--out", out, "--device", "cpu"])
    ea.main()
    with open(out) as f:
        res = json.load(f)
    for s in ("steps1", "steps4"):
        r = res["results"][s]
        assert 0 <= r["oracle_recall"] <= 1 and len(r["oracle_recall_per_stage"]) == 6
        assert "AP50" in r["oracle_score"] and set(r["size_recall"]) == {"<1 ô P5", "1-4 ô P5", ">4 ô P5"}
    att = res["attention"]["999"]
    assert len(att["time"]) == 6
    assert ("ss" in att) == (kind == "spatial_softmax") and ("grid_lift_in_gt" in att) == (kind == "grid")


def test_resume_refuses_changed_model_config(tmp_path, monkeypatch):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    p0, cfg = _test_cfg(tmp_path, "none", root)
    save = str(tmp_path / "s")
    _run_train(monkeypatch, ["--config", p0, "--save-dir", save, "--max-iter", "2"])
    p1, _ = _test_cfg(tmp_path, "grid", root)
    with pytest.raises(SystemExit):
        _run_train(monkeypatch, ["--config", p1, "--save-dir", save, "--resume"])


def test_lr_override_used_and_recorded(tmp_path, monkeypatch):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    p0, _ = _test_cfg(tmp_path, "none", root)
    save = str(tmp_path / "lr")
    _run_train(monkeypatch, ["--config", p0, "--save-dir", save, "--max-iter", "2", "--lr", "1e-4"])
    ck = torch.load(os.path.join(save, "last.pth"), weights_only=False)
    assert ck["config"]["training"]["lr"] == 1e-4
    assert ck["optimizer"]["param_groups"][0]["initial_lr"] == 1e-4


def test_bench_writes_nothing(tmp_path, monkeypatch, capsys):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root, n_train=2, n_val=1, n_test=1)
    p0, _ = _test_cfg(tmp_path, "none", root)
    save = str(tmp_path / "bench")
    _run_train(monkeypatch, ["--config", p0, "--save-dir", save, "--bench", "2"])
    assert "[bench]" in capsys.readouterr().out
    assert not os.path.exists(os.path.join(save, "last.pth"))


def _ddp_worker(rank, world, port, argv):
    import ce_localization.train_alpha as ta
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    ta.build_text_table = _fake_text_table
    sys.argv = ["train_alpha.py"] + argv + ["--device", "cpu"]
    ta.main()


def test_ddp_two_processes(tmp_path):
    """torchrun 2 tiến trình (gloo, CPU): batch toàn cục 2 = 1/GPU, chỉ rank 0 ghi, eval + dừng đúng."""
    import socket
    import torch.multiprocessing as tmp
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    p0, _ = _test_cfg(tmp_path, "none", root)     # ALPHA0: P5 không có RoI -> tham số không dùng
    save = str(tmp_path / "ddp")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tmp.spawn(_ddp_worker, args=(2, port, ["--config", p0, "--save-dir", save, "--max-iter", "2"]),
              nprocs=2, join=True)
    ck = torch.load(os.path.join(save, "last.pth"), weights_only=False)
    assert ck["iter"] == 2 and ck["best"] is not None
    assert sorted(os.listdir(save)) == ["best.pth", "history.json", "last.pth"]


def test_config_diff_ignores_data_root_and_workers():
    """Kaggle gắn dataset ở đường dẫn khác nhau giữa các phiên: resume vẫn phải chạy."""
    from ce_localization.train_alpha import config_diff
    with open(CFG0) as f:
        cfg = yaml.safe_load(f)
    other = json.loads(json.dumps(cfg))
    other["data"].update(root="/kaggle/input/x/all_phase2_V2", num_workers=2)
    other["training"]["max_iter"] = 99
    assert config_diff(cfg, other) == ([], ["training"])
    other["model"]["memory"] = "grid"
    assert config_diff(cfg, other)[0] == ["model"]
