"""DiffuGroundingDINO: text-conditioned detection with diffused reference points.

The idea, from DiffuDINO (DiffuDETR, ICLR 2026), applied to GroundingDINO:

  * Baseline GroundingDINO initialises the decoder's reference points from the
    encoder's top-k proposals. The decoder then refines them layer by layer.
  * Here, the reference points are instead treated as the latent variable of a
    DDPM. At training time ground-truth boxes are noised to a random timestep ``t``
    and handed to the decoder, which learns to denoise them conditioned on the
    image, the text, and ``t``. At inference the points start from pure noise and a
    short DDIM chain (3 decoder evaluations by default) walks them to boxes.

What is *not* diffused, deliberately:

  * **Content queries.** They stay the learned ``tgt_embed``. The paper describes
    them as "static learnable content queries", and the released DiffuDINO computes
    a noised content query but never uses it. Diffusing what a query *is* about,
    rather than where it looks, is a different (untested) model.
  * **The encoder proposal branch.** ``interm_outputs`` is still produced and still
    supervised. Those proposals no longer initialise the decoder, but the branch is
    what teaches the encoder to produce object-shaped features, and the whole
    pipeline reads from those features. Dropping its loss would weaken the encoder
    to save nothing.

Everything else -- the per-layer box refinement, the contrastive text head, the
set-prediction loss -- is untouched, which is what lets ``use_diffusion=False``
reproduce the baseline exactly.
"""

import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from util.misc import NestedTensor, inverse_sigmoid, nested_tensor_from_tensor_list

from .backbone import build_backbone
from .criterion import SetCriterion
from .diffusion import RefPointDiffusion
from .layers import MLP, ContrastiveEmbed
from .matcher import build_matcher
from .postprocess import PostProcess
from .text import SPECIAL_TOKENS, encode_text, get_pretrained_language_model, get_tokenizer
from .transformer import build_transformer


class DiffuGroundingDINO(nn.Module):
    """GroundingDINO with an optional diffusion process over reference points."""

    def __init__(
        self,
        backbone,
        transformer,
        num_queries,
        aux_loss=False,
        iter_update=False,
        query_dim=4,
        num_feature_levels=1,
        nheads=8,
        two_stage_type="no",
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        num_patterns=0,
        text_encoder_type="bert-base-uncased",
        sub_sentence_present=True,
        max_text_len=256,
        # diffusion
        use_diffusion=False,
        diff_num_timesteps=1000,
        diff_sampling_timesteps=3,
        diff_snr_scale=2.0,
        diff_ddim_eta=0.0,
        diff_schedule="cosine",
        diff_loss_weight_mode="diffudino",
        diff_normalize_loss_weight=True,
        diff_pad_mode="center",
    ):
        super().__init__()
        assert query_dim == 4, "the decoder tracks full boxes"
        assert iter_update, "box refinement across decoder layers is mandatory"
        assert two_stage_type in ("no", "standard"), f"unknown two_stage_type {two_stage_type!r}"

        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.max_text_len = max_text_len
        self.sub_sentence_present = sub_sentence_present
        self.query_dim = query_dim
        self.num_patterns = num_patterns
        self.aux_loss = aux_loss
        self.iter_update = iter_update
        self.two_stage_type = two_stage_type
        self.use_diffusion = use_diffusion

        # ---------------- text tower ----------------
        self.tokenizer = get_tokenizer(text_encoder_type)
        self.bert = get_pretrained_language_model(text_encoder_type)
        # The pooler is never used (we read last_hidden_state) but is kept so the
        # pretrained checkpoint's bert.pooler.* keys still load.
        self.bert.pooler.dense.weight.requires_grad_(False)
        self.bert.pooler.dense.bias.requires_grad_(False)

        self.feat_map = nn.Linear(self.bert.config.hidden_size, hidden_dim, bias=True)
        nn.init.constant_(self.feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.feat_map.weight.data)

        self.specical_tokens = self.tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

        # ---------------- image tower ----------------
        self.backbone = backbone
        self.input_proj = self._build_input_proj(backbone, hidden_dim, num_feature_levels, two_stage_type)

        # ---------------- prediction heads ----------------
        _class_embed = ContrastiveEmbed(max_text_len=max_text_len)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # Zero last layer: at init every layer predicts a zero delta, so the boxes
        # start out exactly at the reference points instead of being scattered.
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for _ in range(transformer.num_decoder_layers)]
        else:
            box_embed_layerlist = [copy.deepcopy(_bbox_embed) for _ in range(transformer.num_decoder_layers)]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList([_class_embed for _ in range(transformer.num_decoder_layers)])
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                assert dec_pred_bbox_embed_share, "sharing the encoder box head requires sharing the decoder's"
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.refpoint_embed = None

        # ---------------- diffusion ----------------
        self.diffusion = None
        if use_diffusion:
            self.diffusion = RefPointDiffusion(
                num_timesteps=diff_num_timesteps,
                sampling_timesteps=diff_sampling_timesteps,
                snr_scale=diff_snr_scale,
                ddim_eta=diff_ddim_eta,
                schedule=diff_schedule,
                loss_weight_mode=diff_loss_weight_mode,
                normalize_loss_weight=diff_normalize_loss_weight,
                pad_mode=diff_pad_mode,
            )

        self._reset_parameters()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_input_proj(backbone, hidden_dim, num_feature_levels, two_stage_type) -> nn.ModuleList:
        """1x1 convs mapping each backbone stage to ``hidden_dim``.

        When more levels are requested than the backbone returns, extra levels are
        produced by strided convs on the deepest map -- that is how a 3-stage Swin
        feeds a 4-level deformable encoder.
        """
        if num_feature_levels == 1:
            assert two_stage_type == "no", "two-stage needs multi-scale features"
            return nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )

        projections = []
        in_channels = None
        for channels in backbone.num_channels:
            in_channels = channels
            projections.append(
                nn.Sequential(nn.Conv2d(channels, hidden_dim, kernel_size=1), nn.GroupNorm(32, hidden_dim))
            )
        for _ in range(num_feature_levels - len(backbone.num_channels)):
            projections.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                )
            )
            in_channels = hidden_dim
        return nn.ModuleList(projections)

    def _reset_parameters(self):
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries: int):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

    # ------------------------------------------------------------------ #
    # feature preparation
    # ------------------------------------------------------------------ #
    def _prepare_image_features(self, samples: NestedTensor):
        """Backbone + input projections + extra pyramid levels."""
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        features, poss = self.backbone(samples)
        srcs, masks = [], []
        for level, feat in enumerate(features):
            src, mask = feat.decompose()
            assert mask is not None, "the backbone must return a padding mask"
            srcs.append(self.input_proj[level](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            for level in range(len(srcs), self.num_feature_levels):
                src = self.input_proj[level](features[-1].tensors if level == len(features) else srcs[-1])
                mask = F.interpolate(samples.mask[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos)

        return srcs, masks, poss

    def _encode_text(self, captions: List[str], device):
        return encode_text(
            tokenizer=self.tokenizer,
            bert=self.bert,
            feat_map=self.feat_map,
            captions=captions,
            special_token_ids=self.specical_tokens,
            max_text_len=self.max_text_len,
            sub_sentence_present=self.sub_sentence_present,
            device=device,
        )

    def _text_mask(self, text_dict: dict, device) -> Tensor:
        """``text_token_mask`` right-padded to ``max_text_len``.

        ``pred_logits`` is always ``max_text_len`` wide, so the loss needs a mask of
        that width even when the prompt is shorter.
        """
        bs, length = text_dict["text_token_mask"].shape
        mask = torch.zeros(bs, self.max_text_len, dtype=torch.bool, device=device)
        mask[:, : min(length, self.max_text_len)] = text_dict["text_token_mask"][:, : self.max_text_len]
        return mask

    # ------------------------------------------------------------------ #
    # heads
    # ------------------------------------------------------------------ #
    def _decode_predictions(self, hs, references, text_dict) -> Tuple[Tensor, Tensor]:
        """Per-layer boxes and class logits.

        ``references[i]`` is the box the layer *started* from; the layer's box head
        predicts a delta in unsigmoid space. This is unchanged by diffusion -- the
        only difference is what ``references[0]`` contains.
        """
        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(references[:-1], self.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            outputs_coord_list.append(layer_outputs_unsig.sigmoid())
        outputs_coord = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [layer_cls_embed(layer_hs, text_dict) for layer_cls_embed, layer_hs in zip(self.class_embed, hs)]
        )
        return outputs_class, outputs_coord

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [
            {"pred_logits": a, "pred_boxes": b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def _build_output(self, outputs_class, outputs_coord, enc, tokenized, device) -> Dict:
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        out["text_mask"] = self._text_mask(enc.text_dict, device)
        out["token"] = tokenized

        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        if enc.tgt_undetach is not None:
            hs_enc = enc.tgt_undetach.unsqueeze(0)
            ref_enc = enc.refpoint_embed_undetach.sigmoid().unsqueeze(0)
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1], enc.text_dict)
            out["interm_outputs"] = {"pred_logits": interm_class, "pred_boxes": ref_enc[-1]}
            out["interm_outputs_for_matching_pre"] = {
                "pred_logits": interm_class,
                "pred_boxes": enc.init_box_proposal,
            }
        return out

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, samples: NestedTensor, targets: Optional[List[dict]] = None, **kw):
        """
        Args:
            samples: batched images with a padding mask.
            targets: required for the diffusion training path; each needs ``boxes``
                in normalized cxcywh and ``caption``.
            kw: ``captions`` when ``targets`` is not given (inference).

        Returns:
            The usual detection dict, plus ``diffusion_t`` (bs,) and
            ``diffusion_loss_weight`` (bs,) when the diffusion training path ran.
        """
        # The training loop moves targets to the device by filtering to tensors, which
        # drops the string caption -- so an explicit ``captions`` kwarg wins, and
        # reading it off the targets is only the fallback.
        captions = kw.get("captions")
        if captions is None:
            if targets is None:
                raise ValueError("forward() needs either captions= or targets carrying 'caption'")
            captions = [t["caption"] for t in targets]
        device = samples.device if hasattr(samples, "device") else samples[0].device

        text_dict, tokenized = self._encode_text(captions, device)
        srcs, masks, poss = self._prepare_image_features(samples)

        # Encode once. Every diffusion sampling step reuses this.
        enc = self.transformer.encode(srcs, masks, poss, text_dict)

        run_diffusion_training = self.use_diffusion and self.training
        if run_diffusion_training and targets is None:
            raise ValueError("diffusion training needs targets to noise the ground-truth boxes")

        if run_diffusion_training:
            gt_boxes = [t["boxes"] for t in targets]
            refpoints, _noise, timesteps = self.diffusion.prepare_diffusion_refpoints_batch(
                gt_boxes, self.num_queries
            )
            refpoints = refpoints.to(device)
            timesteps = timesteps.to(device)
            hs, references, _, _, _ = self.transformer.decode(
                enc, refpoint_embed=refpoints, timesteps=timesteps
            )
            outputs_class, outputs_coord = self._decode_predictions(hs, references, enc.text_dict)
            out = self._build_output(outputs_class, outputs_coord, enc, tokenized, device)
            out["diffusion_t"] = timesteps
            out["diffusion_loss_weight"] = self.diffusion.loss_weight(timesteps)
            return out

        if self.use_diffusion:
            return self.ddim_sample(enc, tokenized, device)

        # Baseline path, untouched.
        hs, references, _, _, _ = self.transformer.decode(enc)
        outputs_class, outputs_coord = self._decode_predictions(hs, references, enc.text_dict)
        return self._build_output(outputs_class, outputs_coord, enc, tokenized, device)

    @torch.no_grad()
    def ddim_sample(self, enc, tokenized, device, return_trajectory: bool = False):
        """Generate boxes by denoising reference points from pure noise.

        Only the decoder runs inside the loop; ``enc`` is fixed. With the default
        ``sampling_timesteps=3`` this is 3 decoder evaluations against one
        backbone/BERT/encoder pass -- the DiffuDETR ablation's optimum, and about
        +17% FLOPs over a single-step baseline rather than +200%.

        The predicted ``x_0`` is taken from the *last* decoder layer's refined box,
        which is the model's best estimate of the clean box at this step.
        """
        batch_size = enc.batch_size
        diffusion = self.diffusion

        x = diffusion.init_latent(batch_size, self.num_queries, device)
        trajectory = []
        outputs_class = outputs_coord = None

        for time, time_next in diffusion.ddim_time_pairs():
            timesteps = torch.full((batch_size,), time, device=device, dtype=torch.long)
            refpoints = diffusion.latent_to_refpoints(x)

            hs, references, _, _, _ = self.transformer.decode(
                enc, refpoint_embed=refpoints, timesteps=timesteps
            )
            outputs_class, outputs_coord = self._decode_predictions(hs, references, enc.text_dict)

            # Map the predicted boxes back into latent space, then take the DDIM
            # step there. Mixing the two spaces (as the released DiffuDINO does,
            # stepping on [0,1] boxes with latent-space coefficients) makes the
            # chain inconsistent with the schedule it was trained under.
            x_start = diffusion.boxes_to_latent(outputs_coord[-1])
            x_start = torch.clamp(x_start, -diffusion.snr_scale, diffusion.snr_scale)
            pred_noise = diffusion.predict_noise_from_start(x, timesteps, x_start)
            x = diffusion.ddim_step(x, x_start, pred_noise, time, time_next)

            if return_trajectory:
                trajectory.append({"t": time, "pred_boxes": outputs_coord[-1].clone()})

        out = self._build_output(outputs_class, outputs_coord, enc, tokenized, device)
        if return_trajectory:
            out["trajectory"] = trajectory
        return out

    # Convenience for the eval loop, which only has images and captions.
    @torch.no_grad()
    def sample(self, samples, captions, **kw):
        return self.forward(samples, targets=None, captions=captions, **kw)


def build_diffu_groundingdino(args):
    """Build model, criterion and post-processors from a config namespace."""
    device = torch.device(args.device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    use_diffusion = bool(getattr(args, "use_diffusion", False))

    model = DiffuGroundingDINO(
        backbone,
        transformer,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        iter_update=True,
        query_dim=4,
        num_feature_levels=args.num_feature_levels,
        nheads=args.nheads,
        dec_pred_bbox_embed_share=args.dec_pred_bbox_embed_share,
        two_stage_type=args.two_stage_type,
        two_stage_bbox_embed_share=args.two_stage_bbox_embed_share,
        two_stage_class_embed_share=args.two_stage_class_embed_share,
        num_patterns=args.num_patterns,
        text_encoder_type=args.text_encoder_type,
        sub_sentence_present=args.sub_sentence_present,
        max_text_len=args.max_text_len,
        use_diffusion=use_diffusion,
        diff_num_timesteps=getattr(args, "diff_num_timesteps", 1000),
        diff_sampling_timesteps=getattr(args, "diff_sampling_timesteps", 3),
        diff_snr_scale=getattr(args, "diff_snr_scale", 2.0),
        diff_ddim_eta=getattr(args, "diff_ddim_eta", 0.0),
        diff_schedule=getattr(args, "diff_schedule", "cosine"),
        diff_loss_weight_mode=getattr(args, "diff_loss_weight_mode", "diffudino"),
        diff_normalize_loss_weight=getattr(args, "diff_normalize_loss_weight", True),
        diff_pad_mode=getattr(args, "diff_pad_mode", "center"),
    )

    matcher = build_matcher(args)

    weight_dict = {
        "loss_ce": args.cls_loss_coef,
        "loss_bbox": args.bbox_loss_coef,
        "loss_giou": args.giou_loss_coef,
    }
    clean_weight_dict = copy.deepcopy(weight_dict)

    if args.aux_loss:
        for i in range(args.dec_layers - 1):
            weight_dict.update({f"{k}_{i}": v for k, v in clean_weight_dict.items()})

    if args.two_stage_type != "no":
        no_interm_box_loss = getattr(args, "no_interm_box_loss", False)
        interm_loss_coef = getattr(args, "interm_loss_coef", 1.0)
        coeff = {
            "loss_ce": 1.0,
            "loss_bbox": 0.0 if no_interm_box_loss else 1.0,
            "loss_giou": 0.0 if no_interm_box_loss else 1.0,
        }
        weight_dict.update(
            {f"{k}_interm": v * interm_loss_coef * coeff[k] for k, v in clean_weight_dict.items()}
        )

    criterion = SetCriterion(
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        losses=["labels", "boxes"],
        max_text_len=args.max_text_len,
        use_timestep_weighting=use_diffusion and bool(getattr(args, "diff_loss_t_weighting", True)),
    )
    criterion.to(device)

    postprocessors = {
        "bbox": PostProcess(
            num_select=args.num_select,
            text_encoder_type=args.text_encoder_type,
            nms_iou_threshold=args.nms_iou_threshold,
            args=args,
        )
    }
    return model, criterion, postprocessors


__all__ = ["DiffuGroundingDINO", "build_diffu_groundingdino"]
