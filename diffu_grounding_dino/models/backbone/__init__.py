"""Backbone assembly: feature extractor + position encoding.

``build_backbone`` returns a ``Joiner``, an ``nn.Sequential`` of
``(feature_extractor, position_encoding)``. That layout is what puts the Swin
weights under ``backbone.0.*`` in the checkpoint, so it is preserved verbatim.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

from util.misc import NestedTensor

from .position_encoding import build_position_encoding
from .swin import SwinTransformer, build_swin_transformer

SWIN_NAMES = ("swin_T_224_1k", "swin_B_224_22k", "swin_B_384_22k", "swin_L_224_22k", "swin_L_384_22k")
RESNET_NAMES = ("resnet50", "resnet101")


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm with fixed statistics and affine parameters.

    Detection batches are small and highly variable in size, so the running
    statistics of a pretrained backbone are more reliable than anything a batch of
    two images can estimate. ``eps`` is folded in before the rsqrt, which is what
    keeps non-torchvision backbones from producing NaNs here.
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Checkpoints from a real BatchNorm carry this; we have no use for it.
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        return x * scale + (b - rm * scale)


class ResNetBackbone(nn.Module):
    """torchvision ResNet returning several stages, with frozen BatchNorm.

    Kept for the ablation of a cheaper backbone; the pretrained GroundingDINO
    checkpoints are all Swin.
    """

    def __init__(self, name: str, return_interm_indices: List[int], dilation: bool = False, pretrained: bool = True):
        super().__init__()
        assert name in RESNET_NAMES, f"unsupported resnet {name!r}"
        assert return_interm_indices in ([0, 1, 2, 3], [1, 2, 3], [3]), return_interm_indices

        weights = "DEFAULT" if pretrained else None
        net = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=weights,
            norm_layer=FrozenBatchNorm2d,
        )
        # Only stages 2-4 are trained; stage 1 stays frozen as in DETR.
        for param_name, param in net.named_parameters():
            if not any(f"layer{i}" in param_name for i in (2, 3, 4)):
                param.requires_grad_(False)

        offset = 5 - len(return_interm_indices)
        return_layers = {f"layer{offset + idx}": str(out_idx) for idx, out_idx in enumerate(return_interm_indices)}
        self.body = IntermediateLayerGetter(net, return_layers=return_layers)
        self.num_channels = [256, 512, 1024, 2048][4 - len(return_interm_indices) :]

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        features = self.body(tensor_list.tensors)
        mask = tensor_list.mask
        assert mask is not None
        return {
            name: NestedTensor(feat, F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0])
            for name, feat in features.items()
        }


class Joiner(nn.Sequential):
    """``(feature_extractor, position_encoding)`` -> ``(features, positions)``."""

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor) -> Tuple[List[NestedTensor], List[Tensor]]:
        features = self[0](tensor_list)
        out, pos = [], []
        for _, feat in features.items():
            out.append(feat)
            pos.append(self[1](feat).to(feat.tensors.dtype))
        return out, pos


def build_backbone(args) -> Joiner:
    """Build the backbone described by ``args``.

    Relevant fields: ``backbone``, ``return_interm_indices``, ``dilation``,
    ``use_checkpoint``, ``backbone_freeze_keywords``, plus whatever
    ``build_position_encoding`` reads.
    """
    position_embedding = build_position_encoding(args)
    return_interm_indices = list(args.return_interm_indices)
    assert return_interm_indices in ([0, 1, 2, 3], [1, 2, 3], [3]), return_interm_indices
    use_checkpoint = getattr(args, "use_checkpoint", False)

    if args.backbone in RESNET_NAMES:
        backbone = ResNetBackbone(args.backbone, return_interm_indices, dilation=args.dilation)
        num_channels = backbone.num_channels
    elif args.backbone in SWIN_NAMES:
        pretrain_img_size = int(args.backbone.split("_")[-2])
        backbone = build_swin_transformer(
            args.backbone,
            pretrain_img_size=pretrain_img_size,
            out_indices=tuple(return_interm_indices),
            dilation=False,
            use_checkpoint=use_checkpoint,
        )
        num_channels = backbone.num_features[4 - len(return_interm_indices) :]
    else:
        raise NotImplementedError(f"unknown backbone {args.backbone!r}")

    freeze_keywords = getattr(args, "backbone_freeze_keywords", None)
    if freeze_keywords:
        for name, param in backbone.named_parameters():
            if any(kw in name for kw in freeze_keywords):
                param.requires_grad_(False)

    assert len(num_channels) == len(return_interm_indices), (
        f"backbone returns {len(num_channels)} levels but return_interm_indices asks for "
        f"{len(return_interm_indices)}"
    )

    model = Joiner(backbone, position_embedding)
    model.num_channels = num_channels
    return model


__all__ = [
    "build_backbone",
    "build_position_encoding",
    "build_swin_transformer",
    "FrozenBatchNorm2d",
    "Joiner",
    "ResNetBackbone",
    "SwinTransformer",
]
