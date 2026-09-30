"""Vision encoder của Diffusion Policy (Chi et al., RSS 2023) bản Push-T ảnh — CNN hybrid,
`DiffusionUnetHybridImagePolicy` — viết lại ĐÚNG từng phép tính của robomimic 0.2.0
(`VisualCore` = `ResNet18Conv` -> `SpatialSoftmax(num_kp=32)` -> Flatten -> Linear 64) cộng hai
sửa đổi của diffusion_policy (BatchNorm -> GroupNorm(C/16, C); crop giữa 84×84 khi eval), để
nạp strict các key `obs_encoder.obs_nets.image.*` của checkpoint công bố
(`weights/diffusion_policy/epoch=1850-test_mean_score=0.898.ckpt`).

Khác SpatialSoftmax của CE-Loc (`celoc_vision.py`):
1. Có conv 1×1 512 -> 32 TRƯỚC softmax: 32 keypoint, không phải 512.
2. `np.meshgrid(x, y)` (indexing "xy") -> keypoint k là (NGANG, DỌC), ngược CE-Loc.
3. Ảnh 96×96 [0,1] -> normalizer x*2-1 -> crop giữa 84 -> layer4 chỉ 3×3 ô (28 px/ô).
4. Sau Linear còn ReLU (`feature_activation` mặc định của ObservationEncoder).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = ["DPSpatialSoftmax", "DPVisualCore", "preprocess", "grid_to_crop", "load_dp_encoder",
           "IMAGE_KEY", "CROP", "RAW"]

IMAGE_KEY = "obs_encoder.obs_nets.image."
RAW, CROP = 96, 84


class ResNet18Conv(nn.Module):
    """robomimic `ResNet18Conv`: resnet18 bỏ avgpool + fc; BN -> GroupNorm như diffusion_policy."""

    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        self.nets = nn.Sequential(*list(net.children())[:-2])
        _bn_to_gn(self.nets)

    def forward(self, x):
        return self.nets(x)


def _bn_to_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            _bn_to_gn(child)


class DPSpatialSoftmax(nn.Module):
    """robomimic `SpatialSoftmax`, thêm `return_attention`. Buffer trùng tên bản gốc."""

    def __init__(self, in_c=512, in_h=3, in_w=3, num_kp=32, temperature=1.0):
        super().__init__()
        self.nets = nn.Conv2d(in_c, num_kp, kernel_size=1)
        self._in_h, self._in_w, self._num_kp = in_h, in_w, num_kp
        self.register_buffer("temperature", torch.ones(1) * temperature)
        pos_x, pos_y = np.meshgrid(np.linspace(-1., 1., in_w), np.linspace(-1., 1., in_h))
        self.register_buffer("pos_x", torch.from_numpy(pos_x.reshape(1, in_h * in_w)).float())
        self.register_buffer("pos_y", torch.from_numpy(pos_y.reshape(1, in_h * in_w)).float())

    def forward(self, feature, return_attention=False):
        B = feature.shape[0]
        feature = self.nets(feature).reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(self.pos_x * attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=1, keepdim=True)
        kp = torch.cat([expected_x, expected_y], 1).view(-1, self._num_kp, 2)
        if return_attention:
            return kp, attention.reshape(B, self._num_kp, self._in_h, self._in_w)
        return kp


class DPVisualCore(nn.Module):
    """nets = [ResNet18Conv, SpatialSoftmax, Flatten, Linear] như `VisualCore` gốc."""

    def __init__(self, num_kp=32, feature_dim=64):
        super().__init__()
        self.nets = nn.Sequential(ResNet18Conv(), DPSpatialSoftmax(num_kp=num_kp),
                                  nn.Flatten(1), nn.Linear(num_kp * 2, feature_dim))
        # bản gốc gán cùng module cho cả hai tên -> state_dict có cả `nets.0.*` lẫn `backbone.*`
        self.backbone, self.pool = self.nets[0], self.nets[1]

    def forward(self, x):
        """x: ảnh ĐÃ preprocess [B,3,84,84]. Trả (emb [B,D] sau ReLU, keypoints [B,K,2] (ngang, dọc)
        trong [-1,1], attention [B,K,3,3], feature layer4 [B,512,3,3])."""
        feat = self.nets[0](x)
        kp, att = self.nets[1](feat, return_attention=True)
        emb = F.relu(self.nets[3](self.nets[2](kp)))
        return emb, kp, att, feat


def preprocess(img_hwc):
    """Ảnh zarr [N,96,96,3] (0..255) -> [N,3,84,84]: /255, normalizer ảnh (x*2-1), crop giữa."""
    x = torch.as_tensor(np.asarray(img_hwc, dtype=np.float32) / 255.0).permute(0, 3, 1, 2)
    x = x * 2.0 - 1.0
    o = (RAW - CROP) // 2
    return x[:, :, o:o + CROP, o:o + CROP].contiguous()


def grid_to_crop(u, n_cell=3, size=CROP):
    """Toạ độ lưới [-1,1] -> pixel trên crop 84. -1 / +1 là TÂM ô đầu / ô cuối."""
    cell = size / n_cell
    return cell / 2 + (np.asarray(u) + 1) / 2 * (n_cell - 1) * cell


def load_dp_encoder(ckpt_path, device="cpu", use_ema=True):
    """Nạp STRICT `obs_encoder.obs_nets.image.*` (mặc định từ EMA — bản dùng khi eval).
    Kiểm luôn normalizer ảnh = x*2-1. Trả (encoder.eval(), thông tin checkpoint)."""
    import dill
    ck = torch.load(ckpt_path, map_location="cpu", pickle_module=dill, weights_only=False)
    sd = ck["state_dicts"]["ema_model" if use_ema else "model"]
    sub = {k[len(IMAGE_KEY):]: v for k, v in sd.items() if k.startswith(IMAGE_KEY)}
    if not sub:
        raise KeyError(f"không có key {IMAGE_KEY}* trong {ckpt_path}")
    enc = DPVisualCore(num_kp=sub["nets.1.nets.weight"].shape[0], feature_dim=sub["nets.3.weight"].shape[0])
    enc.load_state_dict(sub, strict=True)
    scale = sd.get("normalizer.params_dict.image.scale")
    offset = sd.get("normalizer.params_dict.image.offset")
    if scale is not None and not (torch.allclose(scale, torch.tensor(2.0)) and torch.allclose(offset, torch.tensor(-1.0))):
        raise ValueError(f"normalizer ảnh không phải x*2-1: scale {scale} offset {offset}")
    cfg = ck.get("cfg")
    info = {"use_ema": use_ema, "n_keys_image": len(sub), "normalizer_scale_offset": (
        None if scale is None else (scale.flatten().tolist(), offset.flatten().tolist()))}
    if cfg is not None:
        p = cfg["policy"]
        info.update(crop_shape=list(p["crop_shape"]), eval_fixed_crop=p["eval_fixed_crop"],
                    obs_encoder_group_norm=p["obs_encoder_group_norm"], n_obs_steps=p["n_obs_steps"])
    return enc.to(device).eval(), info
