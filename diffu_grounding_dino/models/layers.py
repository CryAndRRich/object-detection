"""Small shared building blocks: MLP, sine embeddings, encoder proposals, the
contrastive text-alignment head, and focal loss.
"""

import copy
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _get_clones(module: nn.Module, n: int, layer_share: bool = False) -> nn.ModuleList:
    """``n`` copies of ``module``; ``layer_share`` reuses the same object instead."""
    if layer_share:
        return nn.ModuleList([module] * n)
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation: str, d_model: int = 256, batch_dim: int = 0):
    fns = {"relu": F.relu, "gelu": F.gelu, "glu": F.glu, "selu": F.selu}
    if activation in fns:
        return fns[activation]
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(f"unsupported activation {activation!r}")


class MLP(nn.Module):
    """Feed-forward net with ReLU between layers and none after the last."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip([input_dim] + hidden, hidden + [output_dim])
        )

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _sine_dim_t(num_pos_feats: int, temperature: float, device) -> Tensor:
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    return temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)


def gen_sineembed_for_position(pos_tensor: Tensor, num_pos_feats: int = 128) -> Tensor:
    """Sine embedding of a reference point or box.

    Args:
        pos_tensor: (nq, bs, 2) as ``(x, y)`` or (nq, bs, 4) as ``(x, y, w, h)``,
            normalized to [0, 1].

    Returns:
        (nq, bs, 2 * num_pos_feats) or (nq, bs, 4 * num_pos_feats). Note the
        ``(y, x, w, h)`` ordering of the concatenation, which is what the
        pretrained ``ref_point_head`` expects.
    """
    scale = 2 * math.pi
    dim_t = _sine_dim_t(num_pos_feats, 10000, pos_tensor.device)

    def embed(values: Tensor) -> Tensor:
        scaled = values[:, :, None] * scale / dim_t
        return torch.stack((scaled[:, :, 0::2].sin(), scaled[:, :, 1::2].cos()), dim=3).flatten(2)

    pos_x = embed(pos_tensor[:, :, 0])
    pos_y = embed(pos_tensor[:, :, 1])
    if pos_tensor.shape[-1] == 2:
        return torch.cat((pos_y, pos_x), dim=2)
    if pos_tensor.shape[-1] == 4:
        pos_w = embed(pos_tensor[:, :, 2])
        pos_h = embed(pos_tensor[:, :, 3])
        return torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    raise ValueError(f"pos_tensor last dim must be 2 or 4, got {pos_tensor.shape[-1]}")


def get_sine_pos_embed(
    pos_tensor: Tensor, num_pos_feats: int = 128, temperature: float = 10000, exchange_xy: bool = True
) -> Tensor:
    """Sine embedding of an arbitrary-width position tensor, ``[..., n] -> [..., n*num_pos_feats]``.

    ``exchange_xy`` swaps the first two channels of the output so that a ``(x, y)``
    input is encoded as ``(pos(y), pos(x))``, matching ``gen_sineembed_for_position``.
    """
    scale = 2 * math.pi
    dim_t = _sine_dim_t(num_pos_feats, temperature, pos_tensor.device)

    def embed(values: Tensor) -> Tensor:
        scaled = values * scale / dim_t
        return torch.stack((scaled[..., 0::2].sin(), scaled[..., 1::2].cos()), dim=-1).flatten(-2)

    parts = [embed(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)]
    if exchange_xy and len(parts) >= 2:
        parts[0], parts[1] = parts[1], parts[0]
    return torch.cat(parts, dim=-1)


def gen_encoder_output_proposals(memory: Tensor, memory_padding_mask: Tensor, spatial_shapes: Tensor, learnedwh=None):
    """Turn every encoder position into a candidate box proposal.

    Each spatial location becomes an anchor centred on itself with a
    level-dependent size (0.05 * 2^level of the image, so deeper levels propose
    larger boxes). Positions that are padding, or whose anchor falls outside
    (0.01, 0.99), are marked invalid: their proposal becomes ``inf`` (so that
    ``sigmoid`` saturates and top-k never selects them) and their memory is zeroed.

    Returns:
        ``(output_memory, output_proposals)`` with proposals in unsigmoid space.
    """
    bs, _, _ = memory.shape
    proposals = []
    offset = 0

    for level, (h, w) in enumerate(spatial_shapes):
        h, w = int(h), int(w)
        mask_level = memory_padding_mask[:, offset : offset + h * w].view(bs, h, w, 1)
        valid_h = torch.sum(~mask_level[:, :, 0, 0], 1)
        valid_w = torch.sum(~mask_level[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, h - 1, h, dtype=torch.float32, device=memory.device),
            torch.linspace(0, w - 1, w, dtype=torch.float32, device=memory.device),
            indexing="ij",
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # h, w, 2

        # Normalize by the *valid* extent, not the padded one.
        scale = torch.cat([valid_w.unsqueeze(-1), valid_h.unsqueeze(-1)], 1).view(bs, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**level)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**level)

        proposals.append(torch.cat((grid, wh), -1).view(bs, -1, 4))
        offset += h * w

    output_proposals = torch.cat(proposals, 1)
    valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float("inf"))
    output_proposals = output_proposals.masked_fill(~valid, float("inf"))

    output_memory = memory.masked_fill(memory_padding_mask.unsqueeze(-1), 0.0)
    output_memory = output_memory.masked_fill(~valid, 0.0)
    return output_memory, output_proposals


class ContrastiveEmbed(nn.Module):
    """Classification by similarity to text tokens instead of a fixed class head.

    A query's logit for a category is its dot product with that category's text
    tokens. This is what makes the vocabulary open: swapping the prompt swaps the
    classes, with no new parameters. Padding tokens are set to ``-inf``, and the
    result is right-padded to ``max_text_len`` so the shape is prompt-independent.

    Note that ``-inf`` entries are load-bearing downstream: the criterion masks
    them out with ``text_mask`` before computing the focal loss.
    """

    def __init__(self, max_text_len: int = 256):
        super().__init__()
        self.max_text_len = max_text_len

    def forward(self, x: Tensor, text_dict: dict) -> Tensor:
        assert isinstance(text_dict, dict), f"expected a text_dict, got {type(text_dict)}"
        y = text_dict["encoded_text"]  # bs, num_token, d_model
        text_token_mask = text_dict["text_token_mask"]  # bs, num_token

        res = x @ y.transpose(-1, -2)
        res = res.masked_fill(~text_token_mask[:, None, :], float("-inf"))

        padded = torch.full((*res.shape[:-1], self.max_text_len), float("-inf"), device=res.device, dtype=res.dtype)
        padded[..., : res.shape[-1]] = res
        return padded

    def extra_repr(self) -> str:
        return f"max_text_len={self.max_text_len}"


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, no_reduction: bool = False):
    """Focal loss (Lin et al., ICCV 2017) on sigmoid logits."""
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss if no_reduction else loss.mean(1).sum() / num_boxes


__all__ = [
    "MLP",
    "ContrastiveEmbed",
    "_get_activation_fn",
    "_get_clones",
    "gen_encoder_output_proposals",
    "gen_sineembed_for_position",
    "get_sine_pos_embed",
    "sigmoid_focal_loss",
]
