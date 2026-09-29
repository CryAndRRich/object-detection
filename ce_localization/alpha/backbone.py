"""R-50 + FPN (P2..P5, 256 kênh) — docs/EXPERIMENT_ALPHA.md mục 2.1.

- Weight ImageNet (torchvision IMAGENET1K_V1), TRAIN MỌI LỚP CONV (như CE-Loc gốc: không đóng
  băng lớp nào).
- BatchNorm là `FrozenBatchNorm2d`: thống kê + scale/shift giữ nguyên ImageNet. Batch chỉ 2
  (Kaggle 1 ảnh/GPU) nên BN train được là vô nghĩa. FrozenBatchNorm2d chỉ có BUFFER, không có
  tham số -> không có gì để SyncBN.
- FPN: `torchvision.ops.FeaturePyramidNetwork` (lateral 1x1 + output 3x3, top-down nearest,
  cộng), không thêm P6 vì head chỉ lấy P2..P5 như DiffusionDet (`Base-DiffusionDet.yaml`).
"""

from collections import OrderedDict

import torch.nn as nn
import torchvision
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.misc import FrozenBatchNorm2d

__all__ = ["ResNet50FPN", "STRIDES", "LEVELS"]

LEVELS = ("p2", "p3", "p4", "p5")
STRIDES = (4, 8, 16, 32)


class ResNet50FPN(nn.Module):
    def __init__(self, out_channels=256, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], out_channels)
        self.out_channels = out_channels

    def forward(self, x):
        """[B,3,H,W] -> OrderedDict p2..p5, stride 4/8/16/32."""
        c2 = self.layer1(self.stem(x))
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        out = self.fpn(OrderedDict(zip(LEVELS, (c2, c3, c4, c5))))
        return out
