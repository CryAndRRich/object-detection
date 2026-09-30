"""Vision encoder Diffusion Policy (Push-T) viết lại (ce_localization/celoc_paper/dp_vision.py) +
trọn luồng tools/inspect_dp_spatial_softmax.py trên checkpoint / zarr giả."""

import json
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ce_localization.celoc_paper.dp_vision import (
    IMAGE_KEY,
    DPSpatialSoftmax,
    DPVisualCore,
    grid_to_crop,
    load_dp_encoder,
    preprocess,
)
from ce_localization.tools.inspect_dp_spatial_softmax import (
    block_scenes,
    block_sprite,
    empty_scene,
    episode_splits,
    goal_scenes,
    object_masks,
)

CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "diffusion_policy",
                    "epoch=1850-test_mean_score=0.898.ckpt")


def robomimic_spatial_softmax(feature, conv_w, conv_b):
    """Chép phép tính của robomimic 0.2.0 `SpatialSoftmax.forward` (meshgrid numpy = xy)."""
    _, _, H, W = feature.shape
    pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., W), np.linspace(-1., 1., H))
    pos_x = torch.from_numpy(pos_x.reshape(1, H * W)).float()
    pos_y = torch.from_numpy(pos_y.reshape(1, H * W)).float()
    f = F.conv2d(feature, conv_w, conv_b).reshape(-1, H * W)
    att = F.softmax(f / 1.0, dim=-1)
    xy = torch.cat([torch.sum(pos_x * att, 1, keepdim=True), torch.sum(pos_y * att, 1, keepdim=True)], 1)
    return xy.view(feature.shape[0], -1, 2)


def test_spatial_softmax_matches_robomimic():
    ss = DPSpatialSoftmax(in_c=8, num_kp=5)
    f = torch.randn(2, 8, 3, 3) * 3
    kp, att = ss(f, return_attention=True)
    assert torch.allclose(kp, robomimic_spatial_softmax(f, ss.nets.weight, ss.nets.bias), atol=1e-6)
    assert att.shape == (2, 5, 3, 3)
    assert torch.allclose(att.sum((2, 3)), torch.ones(2, 5), atol=1e-5)


def test_first_coordinate_is_horizontal_and_maps_to_cell_centre():
    ss = DPSpatialSoftmax(in_c=1, num_kp=1)
    with torch.no_grad():
        ss.nets.weight.fill_(1.0)
        ss.nets.bias.zero_()
    f = torch.zeros(1, 1, 3, 3)
    f[0, 0, 0, 2] = 50.0                                   # hàng 0 (trên), cột 2 (phải)
    kp = ss(f)[0, 0]
    assert torch.allclose(kp, torch.tensor([1.0, -1.0]), atol=1e-4)     # (ngang, dọc)
    assert np.allclose(grid_to_crop(kp.detach().numpy()), [70.0, 14.0], atol=1e-3)


def test_encoder_keys_shapes_and_groupnorm():
    enc = DPVisualCore()
    sd = enc.state_dict()
    for k in ["nets.0.nets.0.weight", "nets.0.nets.1.weight", "nets.1.nets.weight", "nets.1.pos_x",
              "nets.1.pos_y", "nets.1.temperature", "nets.3.weight", "backbone.nets.0.weight", "pool.nets.weight"]:
        assert k in sd, k
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in enc.modules())
    gn = enc.nets[0].nets[1]
    assert isinstance(gn, torch.nn.GroupNorm) and gn.num_groups == 4 and gn.num_channels == 64
    assert sd["nets.3.weight"].shape == (64, 64)
    emb, kp, att, feat = enc(torch.zeros(2, 3, 84, 84))
    assert feat.shape == (2, 512, 3, 3) and kp.shape == (2, 32, 2) and att.shape == (2, 32, 3, 3)
    assert emb.shape == (2, 64) and (emb >= 0).all()      # ReLU của ObservationEncoder


def test_preprocess_normalizes_and_center_crops():
    img = np.zeros((1, 96, 96, 3), np.float32)
    img[0, 6, 6] = 255.0                                   # góc trên-trái của crop giữa
    x = preprocess(img)
    assert x.shape == (1, 3, 84, 84)
    assert torch.allclose(x[0, :, 0, 0], torch.ones(3))
    assert x[0, :, 1, 1].eq(-1).all() and x.min() == -1


def test_episode_splits_match_config():
    s = episode_splits(206)
    assert (s == "train").sum() == 90 and (s == "val").sum() == 4 and (s == "unused").sum() == 112


def test_object_masks_on_render_colours():
    img = np.full((4, 4, 3), 255, np.uint8)
    img[0, 0] = (119, 136, 153)                            # LightSlateGray (khối T)
    img[0, 1] = (143, 163, 184)                            # viền khối (khử răng cưa)
    img[1, 0] = (65, 105, 225)                             # RoyalBlue (agent)
    img[2, 0] = (144, 238, 144)                            # LightGreen (đích)
    m = object_masks(img)
    assert m["block"][0, 0] and m["block"][0, 1] and m["block"].sum() == 2
    assert m["agent"][1, 0] and m["agent"].sum() == 1
    assert m["goal"][2, 0] and m["goal"].sum() == 1


def _fake_ckpt(path):
    dill = pytest.importorskip("dill")
    sd = {IMAGE_KEY + k: v for k, v in DPVisualCore().state_dict().items()}
    sd["normalizer.params_dict.image.scale"] = torch.tensor([2.0])
    sd["normalizer.params_dict.image.offset"] = torch.tensor([-1.0])
    sd["model.other.weight"] = torch.zeros(1)
    torch.save({"cfg": None, "state_dicts": {"model": sd, "ema_model": sd}, "pickles": {}}, path, pickle_module=dill)


def test_empty_scene_filters_moving_objects():
    frames = np.full((5, 96, 96, 3), 255, np.uint8)
    frames[:, 40:50, 40:50] = (144, 238, 144)             # đích cố định
    for i in range(5):                                     # khối ở chỗ khác nhau mỗi khung
        frames[i, 5 + 15 * i:10 + 15 * i, 5:10] = (119, 136, 153)
    m = object_masks(empty_scene(frames))
    assert m["goal"].sum() == 100 and m["block"].sum() == 0


def _scene():
    scene = np.full((96, 96, 3), 255, np.uint8)
    scene[0, :] = 233                                      # viền khung (không phải đích)
    scene[40:50, 40:50] = (144, 238, 144)
    return scene


def test_goal_scenes_move_only_goal():
    sc = goal_scenes(_scene(), shift=24)
    assert list(sc) == ["center", "up_left", "down_right"]
    for name, d in (("center", 0), ("up_left", -24), ("down_right", 24)):
        ys, xs = np.nonzero(object_masks(sc[name])["goal"])
        assert ys.min() == 40 + d and xs.min() == 40 + d and len(ys) == 100
        assert (sc[name][0, :60] == 233).all() or d < 0    # viền giữ nguyên (trừ chỗ đích đè lên)


def test_block_sprite_rejects_touching_and_block_scenes_place_centroid():
    frame = np.full((96, 96, 3), 255, np.uint8)
    frame[10:20, 10:16] = (119, 136, 153)
    sprite = block_sprite(frame)
    assert sprite is not None
    touching = frame.copy()
    touching[21:25, 10:16] = (65, 105, 225)                # agent sát khối -> loại
    assert block_sprite(touching) is None
    sc = block_scenes(_scene(), sprite, centers={"a": (30, 60)})
    m = object_masks(sc["a"])
    ys, xs = np.nonzero(m["block"])
    assert m["goal"].sum() == 0 and m["block"].sum() == 60
    assert abs(xs.mean() + 0.0 - 30) <= 1 and abs(ys.mean() - 60) <= 1


def test_full_tool_flow_on_fake_checkpoint_and_zarr(tmp_path, monkeypatch):
    zarr = pytest.importorskip("zarr")
    ck = str(tmp_path / "fake.ckpt")
    _fake_ckpt(ck)
    root = zarr.open(str(tmp_path / "pusht.zarr"), "w")
    img = np.full((206 * 3, 96, 96, 3), 255, np.float32)
    img[:, 40:60, 30:50] = (144, 238, 144)
    img[:, 10:16, 70:76] = (65, 105, 225)
    img[::3][::2, 15:25, 10:20] = (119, 136, 153)          # khối ở khung đầu của nửa số episode
    root["data/img"] = img
    root["meta/episode_ends"] = np.arange(1, 207) * 3
    import ce_localization.tools.inspect_dp_spatial_softmax as tool
    out = str(tmp_path / "out")
    monkeypatch.setattr(sys, "argv", ["x", "--ckpt", ck, "--zarr", str(tmp_path / "pusht.zarr"),
                                      "--episode", "5", "--out", out])
    tool.main()
    assert sorted(os.listdir(out)) == ["block_only.png", "goal_only.png", "metrics.json", "pusht_episode_005.png"]
    m = json.load(open(os.path.join(out, "metrics.json")))
    assert m["episode_info"]["frames"] == [15, 16, 17] and m["block_source_episode"] == 0
    for name in ("episode", "goal_only", "block_only"):
        for r in m[name].values():
            assert 0 <= r["frac_center_cell"] <= 1 and 1 <= r["eff_cells"] <= 9


@pytest.mark.skipif(not os.path.exists(CKPT), reason="thiếu weights/diffusion_policy/epoch=1850-*.ckpt")
def test_checkpoint_loads_strict():
    pytest.importorskip("dill")
    pytest.importorskip("omegaconf")
    enc, info = load_dp_encoder(CKPT)
    assert info["crop_shape"] == [84, 84] and info["eval_fixed_crop"] and info["obs_encoder_group_norm"]
    assert info["n_keys_image"] == len(DPVisualCore().state_dict())
    assert enc.nets[1].temperature.item() == 1.0
