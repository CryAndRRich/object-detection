"""The cross-modality transformer, split into ``encode`` and ``decode``.

Architecture is GroundingDINO's: a deformable image encoder interleaved with text
self-attention and bidirectional fusion, then a decoder whose queries carry
explicit box reference points refined layer by layer.

**The split is the point of this file.** Upstream runs backbone -> BERT -> encoder
-> decoder in one ``forward``. DDIM sampling needs several decoder evaluations per
image, and re-running the backbone and BERT for each of them would multiply
inference cost by the number of sampling steps for no benefit -- the image and
text conditioning do not change between steps. So:

    enc = transformer.encode(...)                      # once per image
    for t in schedule:                                 # 3 times by default
        hs, refs, ... = transformer.decode(enc, refpoint_embed=..., timesteps=t)

``forward()`` still exists and simply chains the two, so the non-diffusion
baseline is bit-for-bit the same computation as before the refactor.

Timestep conditioning is injected in the decoder; see
``models/diffusion/timestep.py`` for the two modes.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.utils.checkpoint as cp
from torch import Tensor, nn

from util.misc import force_fp32, inverse_sigmoid

from .diffusion.timestep import INJECT_MODES, build_timestep_modules
from .fusion import BiAttentionBlock
from .layers import (
    MLP,
    _get_activation_fn,
    _get_clones,
    gen_encoder_output_proposals,
    gen_sineembed_for_position,
    get_sine_pos_embed,
)
from .ops import MSDeformAttn


@dataclass
class EncoderOutput:
    """Everything the decoder needs, computed once per image.

    Holding this explicitly (rather than passing a dozen positional arguments) is
    what makes the multi-step sampler readable and keeps the "encode once" contract
    hard to violate by accident.
    """

    memory: Tensor  # bs, sum(hw), d_model
    mask_flatten: Tensor  # bs, sum(hw)
    lvl_pos_embed_flatten: Tensor  # bs, sum(hw), d_model
    spatial_shapes: Tensor  # nlevel, 2
    level_start_index: Tensor  # nlevel
    valid_ratios: Tensor  # bs, nlevel, 2
    text_dict: dict
    refpoint_embed: Tensor  # bs, nq, 4 (unsigmoid) -- two-stage proposals
    tgt: Tensor  # bs, nq, d_model -- content queries
    init_box_proposal: Tensor  # bs, nq, 4 (sigmoid)
    tgt_undetach: Optional[Tensor]  # bs, nq, d_model, for the encoder-branch loss
    refpoint_embed_undetach: Optional[Tensor]  # bs, nq, 4 (unsigmoid), ditto

    @property
    def batch_size(self) -> int:
        return self.memory.shape[0]

    @property
    def device(self):
        return self.memory.device


class TextEnhancerLayer(nn.Module):
    """Post-norm self-attention layer over the text tokens.

    Named ``text_layers.{i}`` in the checkpoint. The attention mask arrives as the
    3D sub-sentence mask and is expanded per head here.
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.nhead = nhead

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """``src``: (num_token, bs, d_model). ``src_mask``: (bs, L, L) or (bs*nhead, L, L)."""
        if src_mask is not None and src_mask.dim() == 3 and src_mask.shape[0] == src.shape[1]:
            src_mask = src_mask.repeat(self.nhead, 1, 1)

        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask)[0]
        src = self.norm1(src + self.dropout1(src2))

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout2(src2))


class DeformableTransformerEncoderLayer(nn.Module):
    """Deformable self-attention over the flattened multi-scale image features."""

    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu", n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        self.self_attn = MSDeformAttn(
            embed_dim=d_model, num_levels=n_levels, num_heads=n_heads, num_points=n_points, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        return self.norm2(src + self.dropout3(src2))

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, key_padding_mask=None):
        src2 = self.self_attn(
            query=self.with_pos_embed(src, pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            key_padding_mask=key_padding_mask,
        )
        src = self.norm1(src + self.dropout1(src2))
        return self.forward_ffn(src)


class DeformableTransformerDecoderLayer(nn.Module):
    """Self-attention -> text cross-attention -> image deformable cross-attention -> FFN.

    ``timestep_injector``/``timestep_injector_ffn`` are the diffusion hooks, set by
    the decoder according to ``diff_time_inject_point``:

      ``"post_sa"``   one injector, applied between self-attention and the
                      cross-attentions -- DiffuDETR eq. 3 (``FFN(MSDA(SA(q) + t),
                      r_t, O_enc)``).
      ``"pre_layer"`` the decoder conditions the query before calling the layer
                      instead (no injector passed in here).
      ``"triple"``    the released DiffuDINO's actual scheme: three independently
                      parameterised blocks per layer, one before self-attention
                      (applied by the decoder to its input, same as
                      ``"pre_layer"``), one here between self-attention and the
                      cross-attentions (``timestep_injector``), and one before the
                      FFN (``timestep_injector_ffn``).
    """

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_text_cross_attention=False,
    ):
        super().__init__()
        self.cross_attn = MSDeformAttn(
            embed_dim=d_model, num_levels=n_levels, num_heads=n_heads, num_points=n_points, batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)

        self.use_text_cross_attention = use_text_cross_attention

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt, timestep_injector: Optional[nn.Module] = None, timestep_embed: Optional[Tensor] = None):
        if timestep_injector is not None and timestep_embed is not None:
            tgt = timestep_injector(tgt, timestep_embed)
        # The FFN is the widest matmul in the decoder and the one place upstream
        # pins to fp32; keeping that guarantees identical behaviour under AMP.
        with force_fp32():
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout4(tgt2))

    def forward(
        self,
        tgt: Tensor,  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,
        tgt_query_sine_embed: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, nlevel, 4
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,
        memory_spatial_shapes: Optional[Tensor] = None,
        memory_pos: Optional[Tensor] = None,
        self_attn_mask: Optional[Tensor] = None,
        cross_attn_mask: Optional[Tensor] = None,
        timestep_injector: Optional[nn.Module] = None,
        timestep_embed: Optional[Tensor] = None,
        timestep_injector_ffn: Optional[nn.Module] = None,
    ) -> Tensor:
        assert cross_attn_mask is None, "per-query cross-attention masks are not supported"

        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = self.norm2(tgt + self.dropout2(tgt2))

        if timestep_injector is not None and timestep_embed is not None:
            tgt = timestep_injector(tgt, timestep_embed)

        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                self.with_pos_embed(tgt, tgt_query_pos),
                memory_text.transpose(0, 1),
                memory_text.transpose(0, 1),
                key_padding_mask=text_attention_mask,
            )[0]
            tgt = self.catext_norm(tgt + self.catext_dropout(tgt2))

        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            reference_points=tgt_reference_points.transpose(0, 1).contiguous(),
            value=memory.transpose(0, 1),
            spatial_shapes=memory_spatial_shapes,
            level_start_index=memory_level_start_index,
            key_padding_mask=memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        return self.forward_ffn(tgt, timestep_injector_ffn, timestep_embed)


class TransformerEncoder(nn.Module):
    """Image encoder with per-layer text enhancement and cross-modality fusion.

    Order within a layer, following GroundingDINO: fuse image<->text, then update
    text by self-attention, then update image by deformable self-attention.
    """

    def __init__(
        self,
        encoder_layer,
        num_layers,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        text_enhance_layer=None,
        feature_fusion_layer=None,
        use_checkpoint=False,
        use_transformer_ckpt=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        self.text_layers = nn.ModuleList()
        self.fusion_layers = nn.ModuleList()

        if num_layers > 0:
            self.layers = _get_clones(encoder_layer, num_layers, layer_share=enc_layer_share)
            if text_enhance_layer is not None:
                self.text_layers = _get_clones(text_enhance_layer, num_layers, layer_share=enc_layer_share)
            if feature_fusion_layer is not None:
                self.fusion_layers = _get_clones(feature_fusion_layer, num_layers, layer_share=enc_layer_share)

        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.d_model = d_model
        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """One reference point per encoder position, per level.

        Coordinates are normalized by the *valid* extent so that padding does not
        shift where a position looks.
        """
        reference_points_list = []
        for level, (h, w) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, h - 0.5, int(h), dtype=torch.float32, device=device),
                torch.linspace(0.5, w - 0.5, int(w), dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * h)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * w)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        output = src
        reference_points = None
        if self.num_layers > 0:
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)

        if len(self.text_layers) > 0:
            bs, n_text, _ = memory_text.shape
            # One position channel, so the embedding width is num_pos_feats and must
            # equal d_model. Upstream hardcodes 256 here, which silently only works
            # for the default width.
            if pos_text is None and position_ids is None:
                pos_text = torch.arange(n_text, device=memory_text.device).float()[None, :, None].repeat(bs, 1, 1)
                pos_text = get_sine_pos_embed(pos_text, num_pos_feats=self.d_model, exchange_xy=False)
            if position_ids is not None:
                pos_text = get_sine_pos_embed(position_ids[..., None], num_pos_feats=self.d_model, exchange_xy=False)

        for layer_id, layer in enumerate(self.layers):
            if len(self.fusion_layers) > 0:
                fusion = self.fusion_layers[layer_id]
                if self.use_checkpoint and self.training:
                    output, memory_text = cp.checkpoint(
                        fusion, output, memory_text, key_padding_mask, text_attention_mask, use_reentrant=False
                    )
                else:
                    output, memory_text = fusion(
                        v=output, l=memory_text, attention_mask_v=key_padding_mask, attention_mask_l=text_attention_mask
                    )

            if len(self.text_layers) > 0:
                memory_text = self.text_layers[layer_id](
                    src=memory_text.transpose(0, 1),
                    src_mask=~text_self_attention_masks,  # nn.MultiheadAttention wants True = blocked
                    src_key_padding_mask=text_attention_mask,
                    pos=(pos_text.transpose(0, 1) if pos_text is not None else None),
                ).transpose(0, 1)

            if self.use_transformer_ckpt and self.training:
                output = cp.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                    use_reentrant=False,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

        return output, memory_text


class TransformerDecoder(nn.Module):
    """Decoder with explicit box reference points, refined at every layer.

    Each layer predicts a delta in unsigmoid space that is added to the current
    reference point, then re-sigmoided. This is the mechanism the diffusion branch
    plugs into: replacing the *initial* reference points with noised ones changes
    what is being refined, and nothing else about the loop.

    ``time_inject`` (a ``ModuleList``, one per layer or shared) is present only
    when diffusion is enabled.
    """

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        num_feature_levels=1,
        time_inject=None,
        time_inject_point="post_sa",
        debug_nan=False,
    ):
        super().__init__()
        assert return_intermediate, "the box-refinement head needs every layer's output"
        assert query_dim in (2, 4), f"query_dim must be 2 or 4, got {query_dim}"
        assert time_inject_point in ("post_sa", "pre_layer", "triple"), time_inject_point

        self.layers = _get_clones(decoder_layer, num_layers) if num_layers > 0 else nn.ModuleList()
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        self.num_feature_levels = num_feature_levels
        self.d_model = d_model

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.query_pos_sine_scale = None
        self.query_scale = None
        self.bbox_embed = None
        self.class_embed = None
        self.ref_anchor_head = None

        self.time_inject = time_inject
        self.time_inject_point = time_inject_point
        # Checking for NaN costs a device sync per layer per step, so it is opt-in.
        self.debug_nan = debug_nan

    def _injector(self, layer_id: int, group: int = 0) -> Optional[nn.Module]:
        if self.time_inject is None:
            return None
        modules = self.time_inject[group] if self.time_inject_point == "triple" else self.time_inject
        return modules[layer_id if len(modules) > 1 else 0]

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # nq, bs, 4
        level_start_index: Optional[Tensor] = None,
        spatial_shapes: Optional[Tensor] = None,
        valid_ratios: Optional[Tensor] = None,
        memory_text: Optional[Tensor] = None,
        text_attention_mask: Optional[Tensor] = None,
        timestep_embed: Optional[Tensor] = None,  # bs, emb_dim
    ):
        """
        Args:
            tgt: (nq, bs, d_model) content queries.
            refpoints_unsigmoid: (nq, bs, 4) initial reference boxes, pre-sigmoid.
            timestep_embed: diffusion conditioning; ``None`` on the baseline path.

        Returns:
            ``[intermediate_outputs, reference_points]``, both lists of
            ``(bs, nq, ...)`` tensors. ``reference_points`` has ``num_layers + 1``
            entries: the initial boxes followed by one per layer.
        """
        output = tgt
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]
        intermediate = []

        for layer_id, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )
            else:
                reference_points_input = reference_points[:, :, None] * valid_ratios[None, :]

            # Width must be query_dim * (d_model // 2) to feed ref_point_head, whose
            # input is query_dim // 2 * d_model. Upstream hardcodes 128 = 256 // 2.
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], num_pos_feats=self.d_model // 2
            )
            raw_query_pos = self.ref_point_head(query_sine_embed)
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            post_sa_injector = None
            pre_ffn_injector = None
            if self.time_inject is not None and timestep_embed is not None:
                if self.time_inject_point == "pre_layer":
                    output = self._injector(layer_id)(output, timestep_embed)
                elif self.time_inject_point == "post_sa":
                    post_sa_injector = self._injector(layer_id)
                elif self.time_inject_point == "triple":
                    # Three independently parameterised blocks, matching the
                    # released DiffuDINO: before self-attention (applied here, same
                    # spot as "pre_layer"), between self-attention and the
                    # cross-attentions, and before the FFN.
                    output = self._injector(layer_id, group=0)(output, timestep_embed)
                    post_sa_injector = self._injector(layer_id, group=1)
                    pre_ffn_injector = self._injector(layer_id, group=2)

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
                timestep_injector=post_sa_injector,
                timestep_embed=timestep_embed,
                timestep_injector_ffn=pre_ffn_injector,
            )

            if self.debug_nan and not torch.isfinite(output).all():
                raise FloatingPointError(f"decoder layer {layer_id} produced non-finite activations")

            if self.bbox_embed is not None:
                delta_unsig = self.bbox_embed[layer_id](output)
                new_reference_points = (delta_unsig + inverse_sigmoid(reference_points)).sigmoid()
                # Detached so a layer's box gradient does not flow back through the
                # previous layer's box -- DINO's "look forward once".
                reference_points = new_reference_points.detach()
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [item.transpose(0, 1) for item in intermediate],
            [item.transpose(0, 1) for item in ref_points],
        ]


class Transformer(nn.Module):
    """Encoder + decoder, with the encode/decode split described at module level."""

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        num_encoder_layers=6,
        num_unicoder_layers=0,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        query_dim=4,
        num_patterns=0,
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        learnable_tgt_init=False,
        two_stage_type="no",
        embed_init_tgt=False,
        use_text_enhancer=False,
        use_fusion_layer=False,
        use_checkpoint=False,
        use_transformer_ckpt=False,
        use_text_cross_attention=False,
        text_dropout=0.1,
        fusion_dropout=0.1,
        fusion_droppath=0.0,
        # diffusion
        use_diffusion=False,
        diff_time_inject="film",
        diff_time_inject_point="post_sa",
        diff_time_hidden_mult=4,
        diff_film_residual=False,
        diff_time_share_layers=False,
        debug_nan=False,
    ):
        super().__init__()
        assert query_dim == 4, "GroundingDINO's decoder tracks full boxes"
        assert learnable_tgt_init, "content queries must be learnable"
        assert two_stage_type in ("no", "standard"), f"unknown two_stage_type {two_stage_type!r}"
        assert not normalize_before, "pre-norm encoder is not supported"
        assert diff_time_inject in INJECT_MODES, diff_time_inject

        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_patterns = num_patterns if isinstance(num_patterns, int) else 0
        self.two_stage_type = two_stage_type
        self.learnable_tgt_init = learnable_tgt_init
        self.embed_init_tgt = embed_init_tgt
        self.use_diffusion = use_diffusion

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
        )
        text_enhance_layer = (
            TextEnhancerLayer(
                d_model=d_model, nhead=nhead // 2, dim_feedforward=dim_feedforward // 2, dropout=text_dropout
            )
            if use_text_enhancer
            else None
        )
        feature_fusion_layer = (
            BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
            if use_fusion_layer
            else None
        )
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )

        # Diffusion conditioning. Built before the decoder so the injectors can be
        # handed to it; named ``time_embed`` / ``time_inject`` so that a single
        # ``--finetune_ignore time_`` covers both when loading a pretrained model.
        self.time_embed = None
        time_inject = None
        if use_diffusion:
            self.time_embed, time_inject = build_timestep_modules(
                diff_time_inject,
                d_model=d_model,
                num_layers=num_decoder_layers,
                hidden_mult=diff_time_hidden_mult,
                film_residual=diff_film_residual,
                share_across_layers=diff_time_share_layers,
                num_inject_points=3 if diff_time_inject_point == "triple" else 1,
            )

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            nn.LayerNorm(d_model),
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
            time_inject=time_inject,
            time_inject_point=diff_time_inject_point,
            debug_nan=debug_nan,
        )

        if num_feature_levels > 1 and num_encoder_layers > 0:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        else:
            self.level_embed = None

        if (two_stage_type != "no" and embed_init_tgt) or two_stage_type == "no":
            self.tgt_embed = nn.Embedding(num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        if two_stage_type == "standard":
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None
            self.refpoint_embed = None
        else:
            self.init_ref_points(num_queries)

        # Set from the detector, which owns the shared box/class heads.
        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.level_embed is not None:
            nn.init.normal_(self.level_embed)
        # Xavier above would clobber the deliberate zero-init of the timestep
        # modules, which is what keeps them identity at step 0.
        if self.time_embed is not None:
            self._reset_timestep_parameters()

    def _reset_timestep_parameters(self):
        for module in list(self.time_embed.modules()) + list(self.decoder.time_inject.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        # "triple" mode nests one ModuleList of injectors per inject-point inside
        # ``time_inject``; every other mode has a single flat ModuleList.
        groups = self.decoder.time_inject if self.decoder.time_inject_point == "triple" else [self.decoder.time_inject]
        for group in groups:
            for injector in group:
                last = injector.proj[-1] if isinstance(injector.proj, nn.Sequential) else injector.proj
                nn.init.constant_(last.weight, 0.0)
                nn.init.constant_(last.bias, 0.0)

    def init_ref_points(self, use_num_queries: int):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def get_valid_ratio(self, mask: Tensor) -> Tensor:
        """Fraction of each feature map that is real (not padding), as ``(w, h)``."""
        _, height, width = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_w.float() / width, valid_h.float() / height], -1)

    # ------------------------------------------------------------------ #
    # encode
    # ------------------------------------------------------------------ #
    def encode(
        self,
        srcs: List[Tensor],
        masks: List[Tensor],
        pos_embeds: List[Tensor],
        text_dict: dict,
    ) -> EncoderOutput:
        """Run the image/text encoder and prepare the decoder's initial queries.

        Everything here is independent of the diffusion timestep, so a multi-step
        sampler calls this exactly once.
        """
        src_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        bs = srcs[0].shape[0]

        for level, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            _, _, h, w = src.shape
            spatial_shapes.append((h, w))

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            if self.level_embed is not None:
                pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)

            src_flatten.append(src)
            mask_flatten.append(mask.flatten(1))
            lvl_pos_embed_flatten.append(pos_embed)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory, memory_text = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_dict["encoded_text"],
            text_attention_mask=~text_dict["text_token_mask"],
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["text_self_attention_masks"],
        )
        text_dict = {**text_dict, "encoded_text": memory_text}

        if self.two_stage_type == "standard":
            output_memory, output_proposals = gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
            enc_outputs_coord_unselected = self.enc_out_bbox_embed(output_memory) + output_proposals

            topk_logits = enc_outputs_class_unselected.max(-1)[0]
            topk_proposals = torch.topk(topk_logits, self.num_queries, dim=1)[1]

            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            )
            refpoint_embed = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()

            tgt_undetach = torch.gather(
                output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            if self.embed_init_tgt:
                tgt = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            else:
                tgt = tgt_undetach.detach()
        else:
            tgt = self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            refpoint_embed = self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            init_box_proposal = refpoint_embed.sigmoid()
            tgt_undetach = refpoint_embed_undetach = None

            if self.num_patterns > 0:
                tgt = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt = tgt + self.patterns.weight[None, :, :].repeat_interleave(self.num_queries, 1)

        return EncoderOutput(
            memory=memory,
            mask_flatten=mask_flatten,
            lvl_pos_embed_flatten=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            text_dict=text_dict,
            refpoint_embed=refpoint_embed,
            tgt=tgt,
            init_box_proposal=init_box_proposal,
            tgt_undetach=tgt_undetach,
            refpoint_embed_undetach=refpoint_embed_undetach,
        )

    # ------------------------------------------------------------------ #
    # decode
    # ------------------------------------------------------------------ #
    def decode(
        self,
        enc: EncoderOutput,
        refpoint_embed: Optional[Tensor] = None,
        tgt: Optional[Tensor] = None,
        timesteps: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        """Run the decoder against a prepared ``EncoderOutput``.

        Args:
            refpoint_embed: (bs, nq, 4) unsigmoid boxes replacing the two-stage
                proposals -- this is where the diffusion branch injects noised
                reference points. ``None`` keeps the baseline proposals.
            tgt: (bs, nq, d_model) content queries; ``None`` keeps the learned
                embedding. The diffusion branch deliberately leaves this alone.
            timesteps: (bs,) long. Required when diffusion is enabled.

        Returns:
            ``(hs, references, hs_enc, ref_enc, init_box_proposal)``.
        """
        refpoint_embed = enc.refpoint_embed if refpoint_embed is None else refpoint_embed
        tgt = enc.tgt if tgt is None else tgt

        timestep_embed = None
        if timesteps is not None:
            if self.time_embed is None:
                raise RuntimeError("timesteps were provided but the model was built with use_diffusion=False")
            timestep_embed = self.time_embed(timesteps)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=enc.memory.transpose(0, 1),
            memory_key_padding_mask=enc.mask_flatten,
            pos=enc.lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=enc.level_start_index,
            spatial_shapes=enc.spatial_shapes,
            valid_ratios=enc.valid_ratios,
            tgt_mask=attn_mask,
            memory_text=enc.text_dict["encoded_text"],
            text_attention_mask=~enc.text_dict["text_token_mask"],
            timestep_embed=timestep_embed,
        )

        if self.two_stage_type == "standard":
            hs_enc = enc.tgt_undetach.unsqueeze(0)
            ref_enc = enc.refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None

        return hs, references, hs_enc, ref_enc, enc.init_box_proposal

    # ------------------------------------------------------------------ #
    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None, text_dict=None):
        """Baseline path: encode then decode, exactly as before the split.

        ``refpoint_embed`` / ``tgt`` are the denoising-query inputs of the original
        signature; when given they are prepended to the two-stage queries.
        """
        enc = self.encode(srcs, masks, pos_embeds, text_dict)

        if refpoint_embed is not None:
            refpoint_embed = torch.cat([refpoint_embed, enc.refpoint_embed], dim=1)
            tgt = torch.cat([tgt, enc.tgt], dim=1)
        else:
            refpoint_embed, tgt = enc.refpoint_embed, enc.tgt

        return self.decode(enc, refpoint_embed=refpoint_embed, tgt=tgt, attn_mask=attn_mask)


def build_transformer(args) -> Transformer:
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        query_dim=args.query_dim,
        activation=args.transformer_activation,
        num_patterns=args.num_patterns,
        num_feature_levels=args.num_feature_levels,
        enc_n_points=args.enc_n_points,
        dec_n_points=args.dec_n_points,
        learnable_tgt_init=True,
        two_stage_type=args.two_stage_type,
        embed_init_tgt=args.embed_init_tgt,
        use_text_enhancer=args.use_text_enhancer,
        use_fusion_layer=args.use_fusion_layer,
        use_checkpoint=args.use_checkpoint,
        use_transformer_ckpt=args.use_transformer_ckpt,
        use_text_cross_attention=args.use_text_cross_attention,
        text_dropout=args.text_dropout,
        fusion_dropout=args.fusion_dropout,
        fusion_droppath=args.fusion_droppath,
        use_diffusion=getattr(args, "use_diffusion", False),
        diff_time_inject=getattr(args, "diff_time_inject", "film"),
        diff_time_inject_point=getattr(args, "diff_time_inject_point", "triple"),
        diff_time_hidden_mult=getattr(args, "diff_time_hidden_mult", 4),
        diff_film_residual=getattr(args, "diff_film_residual", False),
        diff_time_share_layers=getattr(args, "diff_time_share_layers", False),
        debug_nan=getattr(args, "debug_nan", False),
    )


__all__ = [
    "EncoderOutput",
    "Transformer",
    "TransformerDecoder",
    "TransformerEncoder",
    "build_transformer",
]
