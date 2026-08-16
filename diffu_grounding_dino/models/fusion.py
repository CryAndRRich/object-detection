"""Cross-modality feature fusion for the encoder (GroundingDINO's "feature enhancer").

``BiAttentionBlock`` runs attention in both directions at once from a single score
matrix: image queries attend to text values, and text queries attend to image
values. Both updates are gated by a learned per-channel scale initialised to 1e-4,
so fusion starts as a near no-op and grows only as far as training wants -- that
is what keeps a 6-layer stack of these stable.
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone.swin import DropPath

# fp16 has ~6e4 of headroom; the score matrix is clamped well inside that so a
# single large logit cannot turn the whole softmax into NaN.
CLAMP_LIMIT = 50000


class FeatureResizer(nn.Module):
    """Linear projection + LayerNorm + dropout, for matching modality widths."""

    def __init__(self, input_feat_size: int, output_feat_size: int, dropout: float, do_ln: bool = True):
        super().__init__()
        self.do_ln = do_ln
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features: Tensor) -> Tensor:
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        return self.dropout(x)


class BiMultiHeadAttention(nn.Module):
    """Symmetric image<->text attention sharing one score matrix.

    ``v`` is the image sequence and ``l`` the text sequence. The scores are
    computed once as ``proj(v) @ proj(l)^T``; the image update softmaxes over text
    and the text update softmaxes over image (the transpose).
    """

    def __init__(self, v_dim: int, l_dim: int, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.v_dim = v_dim
        self.l_dim = l_dim
        self.scale = self.head_dim**-0.5
        self.dropout = dropout

        self.v_proj = nn.Linear(v_dim, embed_dim)
        self.l_proj = nn.Linear(l_dim, embed_dim)
        self.values_v_proj = nn.Linear(v_dim, embed_dim)
        self.values_l_proj = nn.Linear(l_dim, embed_dim)
        self.out_v_proj = nn.Linear(embed_dim, v_dim)
        self.out_l_proj = nn.Linear(embed_dim, l_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for proj in (self.v_proj, self.l_proj, self.values_v_proj, self.values_l_proj, self.out_v_proj, self.out_l_proj):
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0.0)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int) -> Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, v: Tensor, l: Tensor, attention_mask_v: Tensor = None, attention_mask_l: Tensor = None):
        """
        Args:
            v: (bs, n_img, v_dim); l: (bs, n_text, l_dim)
            attention_mask_v: (bs, n_img) ``True`` on padding
            attention_mask_l: (bs, n_text) ``True`` on padding

        Returns:
            ``(delta_v, delta_l)`` with the input widths.
        """
        bsz, tgt_len, _ = v.shape
        proj_shape = (bsz * self.num_heads, -1, self.head_dim)

        query_states = self._shape(self.v_proj(v) * self.scale, tgt_len, bsz).view(*proj_shape)
        key_states = self._shape(self.l_proj(l), -1, bsz).view(*proj_shape)
        value_v_states = self._shape(self.values_v_proj(v), -1, bsz).view(*proj_shape)
        value_l_states = self._shape(self.values_l_proj(l), -1, bsz).view(*proj_shape)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))  # bs*nh, n_img, n_text
        # Subtracting the global max keeps both softmaxes in range without changing
        # either result; the clamps bound the fp16 dynamic range.
        attn_weights = (attn_weights - attn_weights.max()).clamp(min=-CLAMP_LIMIT, max=CLAMP_LIMIT)

        attn_weights_l = attn_weights.transpose(1, 2)
        attn_weights_l = attn_weights_l - attn_weights_l.max(dim=-1, keepdim=True)[0]
        attn_weights_l = attn_weights_l.clamp(min=-CLAMP_LIMIT, max=CLAMP_LIMIT)

        if attention_mask_v is not None:
            mask = attention_mask_v[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights_l = attn_weights_l.masked_fill(mask, float("-inf"))
        attn_probs_l = F.dropout(attn_weights_l.softmax(dim=-1), p=self.dropout, training=self.training)

        if attention_mask_l is not None:
            mask = attention_mask_l[:, None, None, :].repeat(1, self.num_heads, 1, 1).flatten(0, 1)
            attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_probs_v = F.dropout(attn_weights.softmax(dim=-1), p=self.dropout, training=self.training)

        attn_output_v = torch.bmm(attn_probs_v, value_l_states)
        attn_output_l = torch.bmm(attn_probs_l, value_v_states)

        attn_output_v = attn_output_v.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output_v = attn_output_v.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)

        src_len = key_states.shape[1]
        attn_output_l = attn_output_l.view(bsz, self.num_heads, src_len, self.head_dim)
        attn_output_l = attn_output_l.transpose(1, 2).reshape(bsz, src_len, self.embed_dim)

        return self.out_v_proj(attn_output_v), self.out_l_proj(attn_output_l)


class BiAttentionBlock(nn.Module):
    """Pre-norm bidirectional fusion with a learned residual gate.

    ``gamma_v`` / ``gamma_l`` start at ``init_values`` (1e-4), which means the
    block barely perturbs either stream at initialisation.
    """

    def __init__(
        self,
        v_dim: int,
        l_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.layer_norm_v = nn.LayerNorm(v_dim)
        self.layer_norm_l = nn.LayerNorm(l_dim)
        self.attn = BiMultiHeadAttention(v_dim=v_dim, l_dim=l_dim, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_v = nn.Parameter(init_values * torch.ones(v_dim))
        self.gamma_l = nn.Parameter(init_values * torch.ones(l_dim))

    def forward(self, v: Tensor, l: Tensor, attention_mask_v: Tensor = None, attention_mask_l: Tensor = None):
        v = self.layer_norm_v(v)
        l = self.layer_norm_l(l)
        delta_v, delta_l = self.attn(v, l, attention_mask_v=attention_mask_v, attention_mask_l=attention_mask_l)
        return v + self.drop_path(self.gamma_v * delta_v), l + self.drop_path(self.gamma_l * delta_l)


__all__ = ["BiAttentionBlock", "BiMultiHeadAttention", "FeatureResizer"]
