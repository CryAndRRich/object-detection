"""Positional encodings for the image features.

``PositionEmbeddingSineHW`` is the DAB-DETR variant with independent temperatures
for the two axes; GroundingDINO's config sets both to 20 (rather than the usual
10000), which sharpens the encoding considerably at detection resolutions.

None of these modules hold parameters (the learned variant aside), so they add no
keys to the checkpoint.
"""

import math

import torch
from torch import Tensor, nn

from util.misc import NestedTensor


class PositionEmbeddingSineHW(nn.Module):
    """Sine/cosine encoding of normalized pixel coordinates.

    Args:
        num_pos_feats: channels per axis; the output has ``2 * num_pos_feats``.
        temperatureH / temperatureW: wavelength bases for the y and x axes.
        normalize: divide coordinates by the number of valid pixels, so padding
            does not shift the encoding of a smaller image in the batch.
    """

    def __init__(self, num_pos_feats=64, temperatureH=10000, temperatureW=10000, normalize=False, scale=None):
        super().__init__()
        if scale is not None and not normalize:
            raise ValueError("scale requires normalize=True")
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x, mask = tensor_list.tensors, tensor_list.mask
        assert mask is not None, "position encoding needs the padding mask"

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        pos_x = x_embed[:, :, :, None] / self._dim_t(self.temperatureW, x.device)
        pos_y = y_embed[:, :, :, None] / self._dim_t(self.temperatureH, x.device)

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

    def _dim_t(self, temperature: float, device) -> Tensor:
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        return temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)


class PositionEmbeddingLearned(nn.Module):
    """Learned absolute position embedding, capped at a 50x50 feature grid."""

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor) -> Tensor:
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        assert h <= 50 and w <= 50, f"learned position embedding supports up to 50x50, got {h}x{w}"

        x_emb = self.col_embed(torch.arange(w, device=x.device))
        y_emb = self.row_embed(torch.arange(h, device=x.device))
        pos = torch.cat([x_emb[None].repeat(h, 1, 1), y_emb[:, None].repeat(1, w, 1)], dim=-1)
        return pos.permute(2, 0, 1)[None].repeat(x.shape[0], 1, 1, 1)


def build_position_encoding(args) -> nn.Module:
    num_pos_feats = args.hidden_dim // 2
    kind = args.position_embedding

    if kind in ("v2", "sine"):
        return PositionEmbeddingSineHW(
            num_pos_feats,
            temperatureH=args.pe_temperatureH,
            temperatureW=args.pe_temperatureW,
            normalize=True,
        )
    if kind in ("v3", "learned"):
        return PositionEmbeddingLearned(num_pos_feats)
    raise ValueError(f"unsupported position_embedding {kind!r}")
