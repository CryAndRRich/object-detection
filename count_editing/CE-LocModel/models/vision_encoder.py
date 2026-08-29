import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import CLIPVisionModel
from models.spatial_softmax import SpatialSoftmax

RESNET_NAMES = ("resnet18", "resnet34", "resnet50")
VIT_NAME = "vit_b32_clip"
VIT_PRETRAINED = "openai/clip-vit-base-patch32"
VIT_INPUT_SIZE = 224  # native pretrain resolution of clip-vit-base-patch32


def _expand_conv2d_to_4ch(conv: nn.Conv2d) -> nn.Conv2d:
    """Return a copy of `conv` accepting 4 input channels (RGB + density).
    RGB weights are copied as-is; the 4th (density) channel is initialized
    with the mean of the RGB weights, matching the original CE-Loc scheme.
    """
    new_conv = nn.Conv2d(
        4, conv.out_channels, kernel_size=conv.kernel_size,
        stride=conv.stride, padding=conv.padding, bias=conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = conv.weight
        new_conv.weight[:, 3:, :, :] = conv.weight.mean(dim=1, keepdim=True)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class SpatialVisualEncoder(nn.Module):
    def __init__(self, output_dim=64, model_name="resnet18", pretrained=True):
        super().__init__()
        self.model_name = model_name

        if model_name in RESNET_NAMES:
            self._build_resnet(model_name, pretrained)
        elif model_name == VIT_NAME:
            self._build_vit(pretrained)
        else:
            raise ValueError(
                f"Unknown vision_encoder.model_name={model_name!r}; "
                f"expected one of {RESNET_NAMES + (VIT_NAME,)}"
            )

        # SPATIAL SOFTMAX: Key component from Diffusion Policy paper.
        # It converts feature maps (H, W) into explicit (x, y) coordinates.
        self.spatial_softmax = SpatialSoftmax(num_features=self.feature_channels)

        # Final projection to match conditioning size (*2 because SpatialSoftmax
        # gives (x,y) per channel).
        self.projection = nn.Linear(self.feature_channels * 2, output_dim)

    def _build_resnet(self, model_name, pretrained=True):
        # `weights=` is the non-deprecated spelling; `pretrained=True` warns on
        # torchvision >= 0.13 (which is what the server env has).
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = getattr(models, model_name)(weights=weights)

        # MODIFY FIRST LAYER: Change input channels from 3 to 4 (RGB + Density).
        # We keep the original weights for RGB and initialize the Density weights.
        resnet.conv1 = _expand_conv2d_to_4ch(resnet.conv1)

        # Remove the classification head (fc) and pooling to keep spatial features.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.feature_channels = resnet.fc.in_features  # 512 for 18/34, 2048 for 50

    def _build_vit(self, pretrained=True):
        if pretrained:
            vit = CLIPVisionModel.from_pretrained(VIT_PRETRAINED)
        else:
            from transformers import CLIPVisionConfig
            vit = CLIPVisionModel(CLIPVisionConfig.from_pretrained(VIT_PRETRAINED))
        vit.vision_model.embeddings.patch_embedding = _expand_conv2d_to_4ch(
            vit.vision_model.embeddings.patch_embedding
        )
        # post_layernorm only ever applies to the pooled CLS output, which
        # _vit_forward drops — leaving it trainable puts two tensors in the
        # optimizer that can never receive a gradient.
        for p in vit.vision_model.post_layernorm.parameters():
            p.requires_grad = False

        self.backbone = vit
        self.feature_channels = vit.config.hidden_size  # 768 for ViT-B/32

    def _resnet_forward(self, x):
        return self.backbone(x)  # [B, C, H/32, W/32]

    def _vit_forward(self, x):
        # CLIP ViT-B/32 is pretrained at 224x224; resize (patch embeddings and
        # positional embeddings are only valid at the pretrain resolution).
        if x.shape[-2:] != (VIT_INPUT_SIZE, VIT_INPUT_SIZE):
            x = F.interpolate(
                x, size=(VIT_INPUT_SIZE, VIT_INPUT_SIZE),
                mode="bilinear", align_corners=False
            )
        out = self.backbone(pixel_values=x)
        tokens = out.last_hidden_state[:, 1:, :]  # drop CLS token -> [B, N_patches, D]
        n_patches = tokens.shape[1]
        side = int(math.isqrt(n_patches))
        assert side * side == n_patches, f"expected a square patch grid, got {n_patches}"
        # [B, N, D] -> [B, D, H', W'] so SpatialSoftmax can pool over space.
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, side, side)

    def forward(self, rgb_image, density_map):
        # rgb: [B, 3, H, W], density: [B, 1, H, W]
        x = torch.cat([rgb_image, density_map], dim=1)
        if self.model_name == VIT_NAME:
            features = self._vit_forward(x)
        else:
            features = self._resnet_forward(x)
        spatial_features = self.spatial_softmax(features)  # [B, C*2]
        return self.projection(spatial_features)