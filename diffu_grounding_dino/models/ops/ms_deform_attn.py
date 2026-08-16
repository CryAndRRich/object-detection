"""Multi-scale deformable attention (Deformable DETR, Zhu et al., ICLR 2021).

Each query predicts, per head and per feature level, a handful of sampling offsets
around its reference point plus a weight for each sample; the output is the
weighted sum of bilinearly interpolated features. Cost is independent of the
feature-map size, which is what makes a 4-level encoder affordable.

**Why a pure-PyTorch core.** The reference implementation requires a compiled CUDA
extension and raises at import time if it is missing, which makes the whole model
unusable on any machine without a build toolchain (Kaggle notebooks included). The
``grid_sample`` formulation below is mathematically the same operation and needs
nothing but torch. If the compiled extension happens to be importable it is used
instead, purely for speed -- so a local CPU debug session and a Kaggle GPU run
exercise the same module with the same weights.

Parameter names (``sampling_offsets``, ``attention_weights``, ``value_proj``,
``output_proj``) match ``groundingdino_swint_ogc.pth``.
"""

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.init import constant_, xavier_uniform_

try:  # optional speed-up; absence is normal and not an error
    import MultiScaleDeformableAttention as _compiled_ops
except Exception:  # noqa: BLE001
    _compiled_ops = None


def _is_power_of_2(n: int) -> bool:
    return (n & (n - 1) == 0) and n != 0


def multi_scale_deformable_attn_pytorch(
    value: Tensor,
    value_spatial_shapes: Tensor,
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    """Reference-free implementation via ``F.grid_sample``.

    Args:
        value: (bs, sum(H*W), num_heads, head_dim)
        value_spatial_shapes: (num_levels, 2) as ``(H, W)``
        sampling_locations: (bs, num_queries, num_heads, num_levels, num_points, 2),
            normalized to [0, 1] with ``(x, y)`` ordering
        attention_weights: (bs, num_queries, num_heads, num_levels, num_points)

    Returns:
        (bs, num_queries, num_heads * head_dim)
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    shapes = [(int(h), int(w)) for h, w in value_spatial_shapes]
    value_list = value.split([h * w for h, w in shapes], dim=1)

    # grid_sample expects [-1, 1]; our locations are in [0, 1].
    sampling_grids = 2 * sampling_locations - 1
    sampled = []
    for level, (h, w) in enumerate(shapes):
        # (bs, H*W, heads, head_dim) -> (bs*heads, head_dim, H, W)
        value_l = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, head_dim, h, w)
        # (bs, queries, heads, points, 2) -> (bs*heads, queries, points, 2)
        grid_l = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        sampled.append(
            F.grid_sample(value_l, grid_l, mode="bilinear", padding_mode="zeros", align_corners=False)
        )

    # (bs, queries, heads, levels, points) -> (bs*heads, 1, queries, levels*points)
    weights = attention_weights.transpose(1, 2).reshape(bs * num_heads, 1, num_queries, num_levels * num_points)
    output = (torch.stack(sampled, dim=-2).flatten(-2) * weights).sum(-1)
    return output.view(bs, num_heads * head_dim, num_queries).transpose(1, 2).contiguous()


class _CompiledMSDeformAttnFunction(torch.autograd.Function):
    """Thin autograd wrapper around the compiled CUDA kernels, when available."""

    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = _compiled_ops.ms_deform_attn_forward(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step
        )
        ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        value, spatial_shapes, level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = _compiled_ops.ms_deform_attn_backward(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


class MultiScaleDeformableAttention(nn.Module):
    """Deformable attention over multi-scale features.

    Args:
        embed_dim: query/value width.
        num_heads: attention heads; ``embed_dim`` must divide evenly.
        num_levels: number of feature maps.
        num_points: sampling points per head per level.
        batch_first: ``False`` (the default, and what the decoder uses) means
            ``(num_query, bs, embed_dim)``.
        force_pytorch_impl: ignore the compiled extension even if importable.
            Useful to prove that a GPU run and a CPU run agree.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        img2col_step: int = 64,
        batch_first: bool = False,
        force_pytorch_impl: bool = False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        if not _is_power_of_2(embed_dim // num_heads):
            warnings.warn(
                f"head dim {embed_dim // num_heads} is not a power of 2; deformable attention is "
                "noticeably slower in that case",
                stacklevel=2,
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.im2col_step = img2col_step
        self.batch_first = batch_first
        self.force_pytorch_impl = force_pytorch_impl

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

        self.init_weights()

    def init_weights(self):
        """Initialise sampling offsets on rings around the reference point.

        Head ``h`` starts out looking along angle ``2*pi*h/num_heads``, and point
        ``p`` sits ``p+1`` units out along that direction. Zero weights on the
        offset projection mean the pattern is purely the bias at step 0, i.e. a
        fixed, well-spread stencil that the model then learns to deform.
        """
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(grid_init.view(-1))

        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def _reset_parameters(self):
        self.init_weights()

    def freeze_sampling_offsets(self):
        self.sampling_offsets.weight.requires_grad_(False)
        self.sampling_offsets.bias.requires_grad_(False)

    def freeze_attention_weights(self):
        self.attention_weights.weight.requires_grad_(False)
        self.attention_weights.bias.requires_grad_(False)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        reference_points: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        level_start_index: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Args:
            query: ``(num_query, bs, embed_dim)`` unless ``batch_first``.
            value: same layout as ``query``; defaults to ``query``.
            key_padding_mask: ``(bs, num_value)``, ``True`` on padding.
            reference_points: ``(bs, num_query, num_levels, 2)`` for points or
                ``(..., 4)`` for boxes, normalized to [0, 1]. With 4 channels the
                offsets are scaled by the box size, so a large box searches a
                proportionally larger neighbourhood.
            spatial_shapes: ``(num_levels, 2)`` as ``(H, W)``.
            level_start_index: ``(num_levels,)``, only used by the compiled kernel.

        Returns:
            Same layout as ``query``.
        """
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()) == num_value, (
            f"spatial_shapes describe {int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())} "
            f"positions but value has {num_value}"
        )

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        # Softmax over levels*points jointly, so a query can move its attention
        # budget between scales rather than only within a scale.
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}")

        output = self._attend(value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output if self.batch_first else output.permute(1, 0, 2)

    def _attend(self, value, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        use_compiled = (
            _compiled_ops is not None
            and not self.force_pytorch_impl
            and value.is_cuda
            and level_start_index is not None
        )
        if not use_compiled:
            return multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        # The kernel is fp32-only; cast around it rather than losing the op.
        half = value.dtype == torch.float16
        if half:
            value = value.float()
            sampling_locations = sampling_locations.float()
            attention_weights = attention_weights.float()
        output = _CompiledMSDeformAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
        )
        return output.half() if half else output

    def extra_repr(self) -> str:
        impl = "pytorch" if (_compiled_ops is None or self.force_pytorch_impl) else "compiled-cuda"
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, num_levels={self.num_levels}, "
            f"num_points={self.num_points}, impl={impl}"
        )


# The transformer refers to this class by the shorter upstream alias.
MSDeformAttn = MultiScaleDeformableAttention

__all__ = ["MSDeformAttn", "MultiScaleDeformableAttention", "multi_scale_deformable_attn_pytorch"]
