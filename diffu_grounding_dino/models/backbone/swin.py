"""Swin Transformer backbone (Liu et al., ICCV 2021), detection variant.

Re-implemented from the paper plus the architectural choices that
``groundingdino_swint_ogc.pth`` was trained with. Every parameter and buffer name
matches that checkpoint exactly -- ``patch_embed.proj``,
``layers.{i}.blocks.{j}.attn.qkv``, ``layers.{i}.downsample.reduction``, and the
per-output ``norm{i}`` layers added as top-level modules. Renaming any of these
silently turns a finetune into training from scratch, so the key list is asserted
in ``tests/test_backbone_text.py``.

Detection-specific deviations from the classification Swin: no classifier head,
features returned at several stages, each passed through its own LayerNorm, and
inputs padded to a multiple of the window size rather than assumed square.
"""

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn

from util.misc import NestedTensor

SWIN_VARIANTS = {
    "swin_T_224_1k": dict(embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7),
    "swin_B_224_22k": dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=7),
    "swin_B_384_22k": dict(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], window_size=12),
    "swin_L_224_22k": dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=7),
    "swin_L_384_22k": dict(embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48], window_size=12),
}


def to_2tuple(value):
    return value if isinstance(value, (tuple, list)) else (value, value)


def drop_path(x: Tensor, drop_prob: float, training: bool) -> Tensor:
    """Stochastic depth: drop whole residual branches, per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self):
        return f"drop_prob={self.drop_prob:.3f}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x: Tensor, window_size: int) -> Tensor:
    """(B, H, W, C) -> (num_windows*B, ws, ws, C)."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows: Tensor, window_size: int, h: int, w: int) -> Tensor:
    """(num_windows*B, ws, ws, C) -> (B, H, W, C)."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    """Multi-head self-attention inside a window, with relative position bias."""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wh, Ww)
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        wh, ww = window_size
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * wh - 1) * (2 * ww - 1), num_heads))
        self.register_buffer("relative_position_index", self._build_relative_position_index(wh, ww))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _build_relative_position_index(wh: int, ww: int) -> Tensor:
        """Flat index into the relative-position bias table for every token pair."""
        coords = torch.stack(torch.meshgrid(torch.arange(wh), torch.arange(ww), indexing="ij"))  # 2, Wh, Ww
        coords_flat = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative = (coords_flat[:, :, None] - coords_flat[:, None, :]).permute(1, 2, 0).contiguous()
        relative[:, :, 0] += wh - 1  # shift the range to start at 0
        relative[:, :, 1] += ww - 1
        relative[:, :, 0] *= 2 * ww - 1
        return relative.sum(-1)  # Wh*Ww, Wh*Ww

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """``x``: (num_windows*B, N, C). ``mask``: (num_windows, N, N) additive."""
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)

        wh, ww = self.window_size
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            wh * ww, wh * ww, -1
        )
        attn = attn + bias.permute(2, 0, 1).contiguous().unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(b_ // num_windows, num_windows, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(self.softmax(attn))

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    """LayerNorm -> (shifted) window attention -> LayerNorm -> MLP, both residual."""

    def __init__(
        self,
        dim,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert 0 <= shift_size < window_size, "shift_size must lie in [0, window_size)"
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

        # Set by the enclosing BasicLayer before each call: a block cannot know the
        # feature-map shape from a flattened (B, L, C) input alone.
        self.H = None
        self.W = None

    def forward(self, x: Tensor, mask_matrix: Tensor) -> Tensor:
        b, length, c = x.shape
        h, w = self.H, self.W
        assert length == h * w, f"expected {h}*{w} tokens, got {length}"

        shortcut = x
        x = self.norm1(x).view(b, h, w, c)

        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        _, hp, wp, _ = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            attn_mask = None

        windows = window_partition(x, self.window_size).view(-1, self.window_size**2, c)
        windows = self.attn(windows, mask=attn_mask)
        x = window_reverse(windows.view(-1, self.window_size, self.window_size, c), self.window_size, hp, wp)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = shortcut + self.drop_path(x.view(b, h * w, c))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchMerging(nn.Module):
    """Halve the resolution and double the channels by concatenating 2x2 neighbours."""

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        b, length, c = x.shape
        assert length == h * w, f"expected {h}*{w} tokens, got {length}"

        x = x.view(b, h, w, c)
        if h % 2 == 1 or w % 2 == 1:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))

        x = torch.cat([x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]], dim=-1)
        x = x.view(b, -1, 4 * c)
        return self.reduction(self.norm(x))


class BasicLayer(nn.Module):
    """One Swin stage: ``depth`` blocks with alternating shift, then a downsample."""

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path_prob=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if i % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path_prob=drop_path_prob[i] if isinstance(drop_path_prob, (list, tuple)) else drop_path_prob,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = downsample(dim=dim, norm_layer=norm_layer) if downsample is not None else None
        self._mask_cache: Dict[Tuple[int, int], Tensor] = {}

    def _shift_attention_mask(self, h: int, w: int, device) -> Tensor:
        """Additive mask that stops tokens from attending across a cyclic-shift seam.

        Depends only on the padded feature size, so it is cached: recomputing it
        per forward (as the reference does) costs a handful of small kernels on
        every block of every stage.
        """
        key = (h, w)
        cached = self._mask_cache.get(key)
        if cached is not None and cached.device == device:
            return cached

        ws, ss = self.window_size, self.shift_size
        hp = -(-h // ws) * ws
        wp = -(-w // ws) * ws
        img_mask = torch.zeros((1, hp, wp, 1), device=device)
        slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        count = 0
        for hs in slices:
            for wsl in slices:
                img_mask[:, hs, wsl, :] = count
                count += 1

        windows = window_partition(img_mask, ws).view(-1, ws * ws)
        attn_mask = windows.unsqueeze(1) - windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        self._mask_cache = {key: attn_mask}  # keep at most one; sizes vary per batch
        return attn_mask

    def forward(self, x: Tensor, h: int, w: int):
        """Returns ``(x_out, H, W, x_downsampled, Wh, Ww)``."""
        attn_mask = self._shift_attention_mask(h, w, x.device)

        for blk in self.blocks:
            blk.H, blk.W = h, w
            if self.use_checkpoint and self.training:
                x = cp.checkpoint(blk, x, attn_mask, use_reentrant=False)
            else:
                x = blk(x, attn_mask)

        if self.downsample is None:
            return x, h, w, x, h, w
        return x, h, w, self.downsample(x, h, w), (h + 1) // 2, (w + 1) // 2


class PatchEmbed(nn.Module):
    """Non-overlapping patch projection, ``patch_size`` stride conv."""

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        ph, pw = self.patch_size
        if w % pw != 0:
            x = F.pad(x, (0, pw - w % pw))
        if h % ph != 0:
            x = F.pad(x, (0, 0, 0, ph - h % ph))

        x = self.proj(x)
        if self.norm is not None:
            wh, ww = x.shape[2], x.shape[3]
            x = self.norm(x.flatten(2).transpose(1, 2))
            x = x.transpose(1, 2).view(-1, self.embed_dim, wh, ww)
        return x


class SwinTransformer(nn.Module):
    """Multi-scale Swin feature extractor.

    ``forward`` takes a ``NestedTensor`` and returns ``{stage_index: NestedTensor}``
    with the padding mask resampled to each feature resolution.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=(2, 2, 6, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        dilation=False,
        use_checkpoint=False,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = tuple(out_indices)
        self.frozen_stages = frozen_stages
        self.dilation = dilation

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        if ape:
            pi = to_2tuple(pretrain_img_size)
            ps = to_2tuple(patch_size)
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, pi[0] // ps[0], pi[1] // ps[1]))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        downsamples = [PatchMerging] * self.num_layers
        downsamples[-1] = None
        if dilation:
            downsamples[-2] = None
            num_features[-1] = num_features[-1] // 2

        self.layers = nn.ModuleList(
            [
                BasicLayer(
                    dim=num_features[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_prob=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    norm_layer=norm_layer,
                    downsample=downsamples[i],
                    use_checkpoint=use_checkpoint,
                )
                for i in range(self.num_layers)
            ]
        )
        self.num_features = num_features

        # One norm per returned stage, registered as a top-level ``norm{i}`` so the
        # checkpoint keys line up.
        for i in self.out_indices:
            self.add_module(f"norm{i}", norm_layer(num_features[i]))

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad_(False)
        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad_(False)
        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(self.frozen_stages - 1):
                self.layers[i].eval()
                for param in self.layers[i].parameters():
                    param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_stages()
        return self

    def forward_features(self, x: Tensor) -> List[Tensor]:
        """Raw multi-scale feature maps, ``(B, C_i, H_i, W_i)`` per returned stage."""
        x = self.patch_embed(x)
        wh, ww = x.shape[2], x.shape[3]

        if self.ape:
            pos = F.interpolate(self.absolute_pos_embed, size=(wh, ww), mode="bicubic", align_corners=False)
            x = (x + pos).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x_out, h, w, x, wh, ww = layer(x, wh, ww)
            if i in self.out_indices:
                x_out = getattr(self, f"norm{i}")(x_out)
                outs.append(x_out.view(-1, h, w, self.num_features[i]).permute(0, 3, 1, 2).contiguous())
        return outs

    def forward(self, tensor_list: NestedTensor) -> Dict[int, NestedTensor]:
        outs = self.forward_features(tensor_list.tensors)
        mask = tensor_list.mask
        assert mask is not None, "the backbone needs the padding mask"

        result = {}
        for idx, feat in enumerate(outs):
            down = F.interpolate(mask[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            result[idx] = NestedTensor(feat, down)
        return result


def build_swin_transformer(modelname: str, pretrain_img_size: int, **kwargs) -> SwinTransformer:
    assert modelname in SWIN_VARIANTS, f"unknown swin variant {modelname!r}"
    config = dict(SWIN_VARIANTS[modelname])
    config.update(kwargs)
    return SwinTransformer(pretrain_img_size=pretrain_img_size, **config)
