"""R-50 + FPN (P2..P5, 256 kênh) — docs/EXPERIMENT_ALPHA.md mục 2.1.

- Weight ImageNet (torchvision IMAGENET1K_V1), TRAIN MỌI LỚP CONV (như CE-Loc gốc: không đóng
  băng lớp nào).
- BatchNorm là `FrozenBatchNorm2d`: thống kê + scale/shift giữ nguyên ImageNet. Batch chỉ 2
  (Kaggle 1 ảnh/GPU) nên BN train được là vô nghĩa. FrozenBatchNorm2d chỉ có BUFFER, không có
  tham số -> không có gì để SyncBN.
- FPN: `torchvision.ops.FeaturePyramidNetwork` (lateral 1x1 + output 3x3, top-down nearest,
  cộng), không thêm P6 vì head chỉ lấy P2..P5 như DiffusionDet (`Base-DiffusionDet.yaml`).
- `in_channels=4` (ALPHA3): conv1 thêm kênh density, weight kênh mới khởi tạo **0** (người dùng
  chốt 2026-09-29) nên ở bước 0 mô hình trùng R-50 3 kênh. CE-Loc gốc khởi tạo kênh này bằng
  trung bình weight RGB (`refs/repos/Count-Editing/CE-LocModel/models/vision_encoder.py`). conv1 được thay SAU khi dựng FPN để thứ tự
  rút RNG khởi tạo trùng bản 3 kênh (cùng seed -> cùng weight, có test).
- GAMMA (bài add, "CE-Loc gốc + R-50", người dùng chốt 2026-10-01): `norm="bn"` = `nn.BatchNorm2d` TRAIN được (thống
  kê theo batch, khởi tạo từ ImageNet) như ResNet18 của bài; `density_init="rgb_mean"` như bài. Mặc định giữ ALPHA.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.misc import FrozenBatchNorm2d

__all__ = ["ResNet50FPN", "STRIDES", "LEVELS", "NORMS", "DENSITY_INITS", "density_weight_ratio"]

LEVELS = ("p2", "p3", "p4", "p5")
NORMS = ("frozen", "bn")
DENSITY_INITS = ("zero", "rgb_mean")
STRIDES = (4, 8, 16, 32)


class ResNet50FPN(nn.Module):
    def __init__(self, out_channels=256, pretrained=True, in_channels=3, norm="frozen", density_init="zero"):
        super().__init__()
        if in_channels not in (3, 4):
            raise ValueError(f"in_channels phải là 3 hoặc 4, nhận {in_channels}")
        if norm not in NORMS or density_init not in DENSITY_INITS:
            raise ValueError(f"norm {norm!r} / density_init {density_init!r} không thuộc {NORMS} / {DENSITY_INITS}")
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet50(weights=weights, norm_layer=FrozenBatchNorm2d if norm == "frozen" else nn.BatchNorm2d)
        fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], out_channels)   # rút RNG như bản 3 kênh
        conv1 = r.conv1
        if in_channels == 4:
            conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                conv1.weight.zero_()
                conv1.weight[:, :3] = r.conv1.weight
                if density_init == "rgb_mean":
                    conv1.weight[:, 3:] = r.conv1.weight.mean(dim=1, keepdim=True)
        self.stem = nn.Sequential(conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.fpn = fpn                          # đăng ký SAU layer4 như cũ: thứ tự tham số không đổi
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        """[B,C,H,W] (C = in_channels) -> OrderedDict p2..p5, stride 4/8/16/32."""
        return self.forward_with_c5(x)[0]

    def forward_with_c5(self, x):
        """-> (OrderedDict p2..p5, C5) trong MỘT lượt (GAMMA1: RoIAlign trên P2..P5 + SpatialSoftmax trên C5)."""
        c2 = self.layer1(self.stem(x))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn(OrderedDict(zip(LEVELS, (c2, c3, c4, c5)))), c5

    def forward_c5(self, x):
        """C5 = đầu ra layer4 (2048 kênh, stride 32) — weight ImageNet, không qua FPN (GAMMA0: SpatialSoftmax như bài)."""
        return self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))

    def forward_p5(self, x):
        """Chỉ P5 (GAMMA0: SpatialSoftmax trên P5) — bỏ nhánh top-down xuống P2..P4. P5 của FPN chỉ phụ
        thuộc C5: `layer_block[-1](inner_block[-1](C5))` (`FeaturePyramidNetwork.forward`), có test so với
        `forward()["p5"]`."""
        c5 = self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))
        return self.fpn.get_result_from_layer_blocks(self.fpn.get_result_from_inner_blocks(c5, -1), -1)


def density_weight_ratio(backbone):
    """‖W_conv1[:, density]‖ / ‖W_conv1[:, RGB]‖ — kênh density có học không (khởi tạo 0).
    None nếu backbone 3 kênh."""
    if backbone.in_channels != 4:
        return None
    w = backbone.stem[0].weight.detach()
    return float(w[:, 3:].norm() / w[:, :3].norm().clamp_min(1e-12))
