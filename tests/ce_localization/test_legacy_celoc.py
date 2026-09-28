"""Vision encoder CE-Loc GỐC viết lại (ce_localization/legacy) + phép đo của
tools/inspect_spatial_softmax.py."""

import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from ce_localization.legacy.celoc_vision import (
    SpatialSoftmax,
    SpatialVisualEncoder,
    grid_to_canvas,
    load_vision_encoder,
    to_input,
)
from ce_localization.tools.inspect_spatial_softmax import coverage, lift, mask_coverage

CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "celoc", "best_model.pth")


def original_spatial_softmax(feature_map):
    """Chép nguyên văn phép tính của refs/.../models/spatial_softmax.py (meshgrid mặc định = ij)."""
    N, C, H, W = feature_map.shape
    pos_x, pos_y = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
    pos_x, pos_y = pos_x.reshape(H * W), pos_y.reshape(H * W)
    attention = F.softmax(feature_map.reshape(N, C, -1), dim=-1)
    ex = torch.sum(pos_x * attention, dim=-1, keepdim=True)
    ey = torch.sum(pos_y * attention, dim=-1, keepdim=True)
    return torch.cat([ex, ey], dim=-1).reshape(N, -1)


def test_spatial_softmax_matches_original():
    f = torch.randn(2, 5, 16, 16) * 3
    out, att = SpatialSoftmax()(f, return_attention=True)
    assert torch.allclose(out, original_spatial_softmax(f), atol=1e-6)
    assert att.shape == (2, 5, 16, 16)
    assert torch.allclose(att.sum((2, 3)), torch.ones(2, 5), atol=1e-5)


def test_first_output_is_vertical_and_maps_to_cell_centre():
    """Đỉnh ở (hàng 3, cột 12): output (x, y) = (DỌC, NGANG) và ra đúng tâm ô trên canvas 512."""
    f = torch.zeros(1, 1, 16, 16)
    f[0, 0, 3, 12] = 60.0
    out = SpatialSoftmax()(f).reshape(-1).numpy()
    row, col = grid_to_canvas(out, 16)
    assert row == pytest.approx(3 * 32 + 16, abs=1e-3)
    assert col == pytest.approx(12 * 32 + 16, abs=1e-3)


def test_uniform_features_collapse_to_centre():
    """Không temperature: kênh không kích hoạt -> điểm kỳ vọng ở giữa lưới."""
    out = SpatialSoftmax()(torch.zeros(1, 3, 16, 16)).numpy()
    assert np.allclose(out, 0, atol=1e-6)
    assert np.allclose(grid_to_canvas(out, 16), 256)


def test_encoder_key_names_and_shapes():
    enc = SpatialVisualEncoder(output_dim=128)
    sd = enc.state_dict()
    assert sd["backbone.0.weight"].shape == (64, 4, 7, 7)
    assert "backbone.7.1.bn2.running_var" in sd and "projection.weight" in sd
    emb, kp, att, feat = enc.eval()(torch.rand(1, 3, 512, 512), torch.rand(1, 1, 512, 512))
    assert emb.shape == (1, 128) and kp.shape == (1, 512, 2) and att.shape == (1, 512, 16, 16)


def test_preprocess_like_original_dataset():
    """472x384 -> 512x416 góc trên-trái, độn đen, KHÔNG chuẩn hoá; density RGBA qua L."""
    img = Image.new("RGB", (472, 384), (255, 255, 255))
    den = Image.new("RGBA", (472, 384), (0, 0, 127, 255))
    rgb, d, scale = to_input(img, den)
    assert scale == pytest.approx(512 / 472)
    assert rgb.shape == (1, 3, 512, 512) and d.shape == (1, 1, 512, 512)
    assert rgb[0, :, :416].min() == 1.0 and rgb[0, :, 416:].max() == 0.0
    assert d[0, 0, :416].unique().tolist() == [pytest.approx(14 / 255)]
    assert d[0, 0, 416:].max() == 0.0


def test_lift_is_one_for_uniform_attention_and_high_on_box():
    scale = 1.0
    cov = coverage([np.array([0, 0, 64, 64], float)], scale, 16)          # 2x2 ô
    assert cov.sum() == pytest.approx(4)
    w = np.ones(2)
    uni = np.full((2, 16, 16), 1 / 256)
    on = np.zeros((2, 16, 16))
    on[:, 0, 0] = 1
    assert lift(uni, w, cov)[0] == pytest.approx(1.0)
    assert lift(on, w, cov)[0] == pytest.approx(64.0)                    # 1 / (4/256)


def test_mask_coverage_follows_resize_and_pad():
    m = np.zeros((384, 472), bool)
    m[:, :236] = True                                                     # nửa trái ảnh
    cov = mask_coverage(m, 512 / 472, 16)
    assert cov[:13, :8].min() == 1.0 and cov[:, 8:].max() == 0.0
    assert cov[13:].max() == 0.0                                          # hàng độn


@pytest.mark.skipif(not os.path.exists(CKPT), reason="thiếu weights/celoc/best_model.pth")
def test_checkpoint_loads_strict():
    enc, info = load_vision_encoder(CKPT)
    assert info["conv1_in_channels"] == 4


# ============================================================================
# CE-Loc gốc: model đầy đủ, sampler, IoU, dataset, train/eval trọn luồng
# ============================================================================

import json as _json
import math as _math
import sys as _sys

from ce_localization.legacy.celoc_data import ObjectPlacementDataset
from ce_localization.legacy.celoc_model import (
    ObjectPlacementPolicy,
    iou_original_formula,
    iou_pixels,
    load_policy,
    sample_ddpm,
    sample_mock,
)


def _tiny_policy(use_density=True, num_timesteps=1000):
    """Đúng kiến trúc, không tải gì; text encoder thay bằng hằng (không có tokenizer offline)."""
    m = ObjectPlacementPolicy(use_density=use_density, pretrained_vision=False, pretrained_text=False,
                              num_timesteps=num_timesteps)
    m.text_encoder.forward = lambda texts: torch.zeros(len(texts), 128)
    return m


@pytest.mark.skipif(not os.path.exists(CKPT), reason="thiếu weights/celoc/best_model.pth")
def test_policy_architecture_matches_original_checkpoint():
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
    m = ObjectPlacementPolicy(pretrained_vision=False, pretrained_text=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert [k for k in missing + unexpected if not k.endswith("position_ids")] == []
    model, _ = load_policy(CKPT, pretrained_text=False)
    assert model.use_density and model.num_timesteps == 1000


def test_cosine_schedule_reproduces_checkpoint_lr():
    """lr trong optimizer của best_model.pth (epoch 113) = 3,420311e-5 với lr đầu 5e-5."""
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=5e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=300)
    for _ in range(114):
        opt.step()
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(3.420311381711698e-05, rel=1e-6)


def test_no_density_encoder_is_three_channel_and_ignores_density():
    enc = SpatialVisualEncoder(in_channels=3).eval()
    assert enc.backbone[0].weight.shape == (64, 3, 7, 7)
    rgb = torch.rand(1, 3, 128, 128)
    assert torch.equal(enc(rgb, torch.rand(1, 1, 128, 128))[0], enc(rgb)[0])


def test_four_channel_density_init_is_rgb_mean():
    w = SpatialVisualEncoder(in_channels=4).backbone[0].weight
    assert torch.allclose(w[:, 3:], w[:, :3].mean(dim=1, keepdim=True))


def test_unet_horizon_one_shape():
    m = _tiny_policy()
    out = m.noise_net(torch.randn(5, 1, 4), torch.randint(0, 1000, (5,)), torch.randn(5, 256))
    assert out.shape == (5, 1, 4)


def _calculate_iou_original(box1, box2):
    """Chép nguyên văn test_mul_box.calculate_iou."""
    b1_x1, b1_y1 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2
    b1_x2, b1_y2 = box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    b2_x1, b2_y1 = box2[0] - box2[2] / 2, box2[1] - box2[3] / 2
    b2_x2, b2_y2 = box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    xi1, yi1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    xi2, yi2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (b1_area + b2_area - inter_area + 1e-6)


def test_iou_original_formula_matches_original_code():
    rng = np.random.default_rng(0)
    p, g = rng.uniform(-1, 1, (50, 4)), rng.uniform(-1, 1, (50, 4))
    ref = [_calculate_iou_original(a, b) for a, b in zip(p, g)]
    assert np.allclose(iou_original_formula(p, g), ref)


def test_iou_pixels_denormalizes_width_and_height():
    to_u = lambda px: np.array(px, float) / 512 * 2 - 1                  # noqa: E731
    a = to_u([100, 100, 40, 40])
    assert iou_pixels(a, a) == pytest.approx(1.0)
    assert iou_pixels(a, to_u([300, 300, 40, 40])) == pytest.approx(0.0)
    assert iou_pixels(a, to_u([120, 100, 40, 40])) == pytest.approx(1 / 3)
    # công thức gốc trên giá trị chuẩn hoá: w âm -> KHÔNG ra 1/3
    assert abs(iou_original_formula(a, to_u([120, 100, 40, 40])) - 1 / 3) > 0.05


class _OracleNet:
    """Dự đoán ε đúng cho một x0 biết trước -> DDPM phải trả đúng x0."""

    def __init__(self, x0, ab):
        self.x0, self.ab = x0, ab

    def __call__(self, x, t, c):
        ab = self.ab[t].unsqueeze(-1)
        return ((x.squeeze(1) - torch.sqrt(ab) * self.x0) / torch.sqrt(1 - ab)).unsqueeze(1)


def test_sample_ddpm_recovers_x0_with_oracle_eps():
    T = 1000
    ab = torch.cumprod(1 - torch.linspace(1e-4, 0.02, T), 0)
    x0 = torch.tensor([0.3, -0.2, -0.8, -0.6])
    m = type("M", (), {})()
    m.alphas_cumprod, m.num_timesteps, m.noise_net = ab, T, _OracleNet(x0, ab)
    out = sample_ddpm(m, torch.zeros(2, 256), 3, generator=torch.Generator().manual_seed(0))
    assert out.shape == (2, 3, 4)
    assert torch.allclose(out, x0.expand_as(out), atol=1e-3)


def test_sample_mock_is_original_loop():
    net = lambda x, t, c: x * 0.5 + t[:, None, None].float() * 1e-3    # noqa: E731
    m = type("M", (), {"noise_net": staticmethod(net)})()
    out = sample_mock(m, torch.zeros(1, 256), 4, generator=torch.Generator().manual_seed(1))
    box = torch.randn((4, 4), generator=torch.Generator().manual_seed(1))
    for t in reversed(range(100)):
        box = box - (1.0 / 100) * net(box.unsqueeze(1), torch.full((4,), t), None).squeeze(1)
    assert torch.allclose(out[0], box, atol=1e-6)


def _fake_samples(root, n=4):
    """Thư mục kiểu samples/: ảnh 472x384, density RGBA jet, annotation cxcywh pixel."""
    for sub in ("images", "density", "annotation"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (384, 472, 3), dtype=np.uint8)).save(f"{root}/images/{i}_1.png")
        Image.new("RGBA", (472, 384), (0, 0, 127, 255)).save(f"{root}/density/{i}_1.png")
        with open(f"{root}/annotation/{i}_1.json", "w") as f:
            _json.dump({"class": "cup", "target_bbox": [236.0, 192.0, 47.2, 38.4]}, f)


def test_dataset_normalizes_like_original(tmp_path):
    _fake_samples(tmp_path, 1)
    item = ObjectPlacementDataset(str(tmp_path))[0]
    s = 512 / 472
    exp = [(236 * s / 512) * 2 - 1, (192 * s / 512) * 2 - 1, (47.2 * s / 512) * 2 - 1, (38.4 * s / 512) * 2 - 1]
    assert np.allclose(item["bbox"].numpy(), exp, atol=1e-6)
    assert item["pixel_values"].shape == (3, 512, 512) and item["density_map"].shape == (1, 512, 512)
    assert item["density_map"][0, 0, 0] == pytest.approx(14 / 255)
    assert "density_map" not in ObjectPlacementDataset(str(tmp_path), use_density=False)[0]


def test_loss_backward_trains_vision_and_unet_not_clip():
    m = _tiny_policy()
    loss = m.compute_loss(torch.rand(2, 3, 128, 128), torch.rand(2, 1, 128, 128), ["a", "b"],
                          torch.rand(2, 4) * 2 - 1, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert _math.isfinite(loss.item())
    assert m.vision_encoder.backbone[0].weight.grad.abs().sum() > 0
    assert m.noise_net.final_conv[1].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.text_encoder.backbone.parameters())


@pytest.mark.parametrize("no_density", [False, True])
def test_train_then_resume_then_eval_full_flow(tmp_path, monkeypatch, no_density):
    """Chạy TRỌN legacy/train.py (1 epoch + eval) -> --resume thêm 1 epoch -> legacy/eval.py.
    T = 50 cho DDPM nhanh trên CPU (kiến trúc không đổi)."""
    import ce_localization.legacy.eval as ev
    import ce_localization.legacy.train as tr

    data = str(tmp_path / "samples")
    _fake_samples(data, 3)
    save = str(tmp_path / "ck")
    monkeypatch.setattr(tr, "ObjectPlacementPolicy", lambda use_density: _tiny_policy(use_density, 50))
    base = ["train.py", "--save-dir", save, "--data", data, "--eval-data", data, "--batch-size", "2",
            "--num-workers", "0", "--eval-every", "1", "--eval-n", "2", "--epochs", "2"]
    base += ["--no-density"] if no_density else []

    monkeypatch.setattr(_sys, "argv", base)
    monkeypatch.setattr(tr.CheckpointManager, "save", _stop_after_first(tr.CheckpointManager.save))
    with pytest.raises(_StopTrain):
        tr.main()
    h1 = torch.load(os.path.join(save, "last.pt"), weights_only=False)
    assert h1["epoch"] == 0 and "eval" in h1["history"][0]

    monkeypatch.undo()
    monkeypatch.setattr(tr, "ObjectPlacementPolicy", lambda use_density: _tiny_policy(use_density, 50))
    monkeypatch.setattr(_sys, "argv", base + ["--resume"])
    tr.main()
    h2 = torch.load(os.path.join(save, "last.pt"), weights_only=False)
    assert h2["epoch"] == 1 and len(h2["history"]) == 2
    assert os.path.exists(os.path.join(save, "best.pt"))

    monkeypatch.setattr(ev, "load_policy", lambda p, d: _load_tiny(p, not no_density))
    out = str(tmp_path / "res.json")
    monkeypatch.setattr(_sys, "argv", ["eval.py", "--ckpt", os.path.join(save, "best.pt"), "--data", data,
                                       "--prior-from", data, "--n-samples", "3", "--num-workers", "0",
                                       "--out", out])
    ev.main()
    res = _json.load(open(out))
    assert set(res["summary"]) == {"mock", "ddpm", "prior"}
    assert len(res["per_image"]["ddpm"]["best"]) == 3
    assert res["summary"]["prior"]["best_iou"] == pytest.approx(1.0)   # mọi box trùng nhau


class _StopTrain(Exception):
    pass


def _stop_after_first(save):
    def wrapped(self, state, is_best):
        save(self, state, is_best)
        raise _StopTrain
    return wrapped


def _load_tiny(path, use_density):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = _tiny_policy(use_density, 50)
    m.load_state_dict(ck["model_state_dict"])
    return m.eval(), ck
