"""Vision encoder của CE-Loc gốc (ResNet18 4 kênh + SpatialSoftmax), viết lại ĐÚNG từng phép
tính của `models/vision_encoder.py`, `models/spatial_softmax.py`, `data/dataset.py` trong
refs/repos/Count-Editing/CE-LocModel — để nạp strict các key `vision_encoder.*` của
`best_paper.pth`.

Ba chi tiết của bản gốc phải giữ nguyên (sai là soi nhầm model):

1. `torch.meshgrid` indexing "ij" -> `pos_x` chạy theo HÀNG. Output kênh c là
   (expected_x, expected_y) = (toạ độ DỌC, toạ độ NGANG), xen kẽ [x0, y0, x1, y1, ...].
2. Ảnh chỉ `to_tensor` về [0,1], KHÔNG chuẩn hoá mean/std; độn ĐEN ở dưới/phải.
3. Density là PNG RGBA tô màu jet, đọc bằng `.convert("L")` (độ sáng, không đơn điệu theo
   mật độ), resize NEAREST, độn 0.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image

__all__ = ["SpatialSoftmax", "SpatialVisualEncoder", "resize_and_pad", "to_input",
           "grid_to_canvas", "load_vision_encoder"]

TARGET = 512


class SpatialSoftmax(nn.Module):
    """Như bản gốc, thêm `return_attention` để lấy bản đồ softmax [B, C, H, W]."""

    def forward(self, feature_map, return_attention=False):
        N, C, H, W = feature_map.shape
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feature_map.device),
            torch.linspace(-1, 1, W, device=feature_map.device),
            indexing="ij",
        )
        attention = F.softmax(feature_map.reshape(N, C, -1), dim=-1)
        expected_x = torch.sum(pos_x.reshape(H * W) * attention, dim=-1, keepdim=True)
        expected_y = torch.sum(pos_y.reshape(H * W) * attention, dim=-1, keepdim=True)
        out = torch.cat([expected_x, expected_y], dim=-1).reshape(N, -1)
        if return_attention:
            return out, attention.reshape(N, C, H, W)
        return out


class SpatialVisualEncoder(nn.Module):
    """ResNet18 -> SpatialSoftmax -> Linear. Tên module trùng bản gốc.

    in_channels=4 (RGB + density, như bản gốc) hoặc 3 (bỏ density: conv1 ImageNet giữ nguyên).
    pretrained=True: ImageNet như `resnet18(pretrained=True)` gốc, kênh density khởi tạo bằng
    trung bình 3 kênh RGB. pretrained=False: weight sẽ lấy từ checkpoint.
    """

    def __init__(self, output_dim=128, in_channels=4, pretrained=False):
        super().__init__()
        if in_channels not in (3, 4):
            raise ValueError(f"in_channels phải là 3 hoặc 4, nhận {in_channels}")
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet18(weights=weights)
        if in_channels == 4:
            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv1.weight[:, :3] = resnet.conv1.weight
                conv1.weight[:, 3:] = resnet.conv1.weight.mean(dim=1, keepdim=True)
            resnet.conv1 = conv1
        self.in_channels = in_channels
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.spatial_softmax = SpatialSoftmax()
        self.projection = nn.Linear(512 * 2, output_dim)

    def forward(self, rgb, density=None):
        """Trả (emb [B,D], keypoints [B,C,2] (dọc, ngang) trong [-1,1], attention [B,C,h,w],
        feature [B,C,h,w]). Bản 3 kênh bỏ qua `density`."""
        x = torch.cat([rgb, density], dim=1) if self.in_channels == 4 else rgb
        feat = self.backbone(x)
        xy, att = self.spatial_softmax(feat, return_attention=True)
        return self.projection(xy), xy.reshape(xy.shape[0], -1, 2), att, feat


def resize_and_pad(img, density=None, target=TARGET):
    """Như `ObjectPlacementDataset.resize_and_pad`: density phải là ảnh "L" (None = bản không density)."""
    w, h = img.size
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), resample=Image.BILINEAR)
    padded_img = Image.new("RGB", (target, target), (0, 0, 0))
    padded_img.paste(img, (0, 0))
    if density is None:
        return padded_img, None, scale
    density = density.resize((nw, nh), resample=Image.NEAREST)
    padded_density = Image.new("L", (target, target), 0)
    padded_density.paste(density, (0, 0))
    return padded_img, padded_density, scale


def to_input(img_rgb, density_any=None):
    """PIL RGB + PIL density (mode bất kỳ, như file PNG gốc) -> tensor [1,3,T,T], [1,1,T,T],
    scale. `.convert("L")` như dataset gốc. density_any=None -> density trả về None."""
    img, den, scale = resize_and_pad(img_rgb.convert("RGB"),
                                     None if density_any is None else density_any.convert("L"))
    rgb = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
    if den is None:
        return rgb[None], None, scale
    d = torch.from_numpy(np.asarray(den, dtype=np.float32) / 255.0)[None]
    return rgb[None], d[None], scale


def grid_to_canvas(u, n_cell, target=TARGET):
    """Toạ độ lưới [-1,1] của SpatialSoftmax -> pixel trên canvas `target`.
    -1 / +1 là TÂM ô đầu / ô cuối (linspace), không phải mép canvas."""
    cell = target / n_cell
    return cell / 2 + (np.asarray(u) + 1) / 2 * (n_cell - 1) * cell


def load_vision_encoder(ckpt_path, device="cpu"):
    """Nạp STRICT các key `vision_encoder.*`. Trả (encoder.eval(), thông tin checkpoint)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    sub = {k[len("vision_encoder."):]: v for k, v in sd.items() if k.startswith("vision_encoder.")}
    if not sub:
        raise KeyError(f"không có key vision_encoder.* trong {ckpt_path}")
    enc = SpatialVisualEncoder(output_dim=sub["projection.weight"].shape[0],
                               in_channels=int(sub["backbone.0.weight"].shape[1]))
    enc.load_state_dict(sub, strict=True)
    info = {k: v for k, v in ck.items() if k not in ("model_state_dict", "optimizer_state_dict")}
    info["n_keys_total"] = len(sd)
    info["n_keys_vision"] = len(sub)
    info["conv1_in_channels"] = int(sub["backbone.0.weight"].shape[1])
    return enc.to(device).eval(), info
