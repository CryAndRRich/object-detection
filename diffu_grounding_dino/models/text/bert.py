"""Text side: tokenizer, BERT encoder, and the sub-sentence attention masks.

GroundingDINO feeds the text encoder a caption built by joining category names
with " . " (``"person . car . dog ."``). Left alone, BERT would let "person"
attend to "car", so the embedding of one category would depend on which other
categories happen to share the prompt -- and detection scores would shift with
the prompt's composition. The fix (``sub_sentence_present``) is a block-diagonal
self-attention mask: each category attends only within its own span, and position
ids restart at every span. That is what
``generate_masks_with_special_tokens_and_transfer_map`` builds.

The BERT weights live under ``bert.*`` in the checkpoint, which is exactly where a
plain ``transformers.BertModel`` assigned to ``self.bert`` puts them, so no
wrapper class is needed here.
"""

import os
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from transformers import AutoTokenizer, BertModel, RobertaModel

SPECIAL_TOKENS = ["[CLS]", "[SEP]", ".", "?"]


def get_tokenizer(text_encoder_type: str):
    """Load a tokenizer from a hub id or a local directory."""
    text_encoder_type = _resolve_encoder_type(text_encoder_type)
    return AutoTokenizer.from_pretrained(text_encoder_type)


def force_eager_attention(model: nn.Module) -> nn.Module:
    """Pin the text encoder to eager attention, in place.

    This is load-bearing, not a preference. transformers defaults BERT to SDPA,
    and ``_prepare_4d_attention_mask_for_sdpa`` unpacks the mask as
    ``_, key_value_length = mask.shape`` -- it assumes a 2D padding mask and dies
    on our 3D sub-sentence mask with "too many values to unpack". Some versions
    guard that path with ``attention_mask.dim() == 2`` and some do not, so we do
    not rely on the version: eager is forced explicitly.

    Both the config field and the module attribute are set, because ``BertModel``
    caches ``config._attn_implementation`` into ``self.attn_implementation`` at
    construction time -- patching only the config would be silently ignored.
    """
    config = getattr(model, "config", None)
    if config is not None:
        for attr in ("_attn_implementation", "attn_implementation"):
            if hasattr(config, attr):
                setattr(config, attr, "eager")
    if hasattr(model, "attn_implementation"):
        model.attn_implementation = "eager"
    for module in model.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = "eager"
    return model


def get_pretrained_language_model(text_encoder_type: str) -> nn.Module:
    """Load the text encoder, pinned to eager attention.

    The pooler is kept (unused at runtime, frozen by the model) because
    ``groundingdino_swint_ogc.pth`` contains ``bert.pooler.dense.*``; dropping it
    would turn those into unexpected keys.
    """
    text_encoder_type = _resolve_encoder_type(text_encoder_type)
    lowered = os.path.basename(text_encoder_type.rstrip("/")).lower()
    loader = RobertaModel if "roberta" in lowered else BertModel

    try:
        model = loader.from_pretrained(text_encoder_type, attn_implementation="eager")
    except TypeError:
        # Older/newer transformers may not accept the kwarg; patch after loading.
        model = loader.from_pretrained(text_encoder_type)
    return force_eager_attention(model)


def _resolve_encoder_type(text_encoder_type) -> str:
    if isinstance(text_encoder_type, str):
        return text_encoder_type
    if hasattr(text_encoder_type, "text_encoder_type"):
        return text_encoder_type.text_encoder_type
    if isinstance(text_encoder_type, dict) and "text_encoder_type" in text_encoder_type:
        return text_encoder_type["text_encoder_type"]
    raise ValueError(f"cannot resolve a text encoder from {type(text_encoder_type)}")


def generate_masks_with_special_tokens_and_transfer_map(
    tokenized, special_token_ids: List[int]
) -> Tuple[Tensor, Tensor, List[Tensor]]:
    """Build the block-diagonal text self-attention mask.

    Special tokens (``[CLS]``, ``[SEP]``, ``.``, ``?``) act as separators. Every
    run of tokens between two separators is one category phrase, and becomes one
    block: tokens inside it attend to each other and to nothing else.

    Args:
        tokenized: the tokenizer output; only ``input_ids`` is read.
        special_token_ids: ids of the separator tokens.

    Returns:
        ``attention_mask``: (bs, L, L) bool, ``True`` where attention is allowed.
        ``position_ids``: (bs, L) long, restarting at 0 in every block.
        ``cate_to_token_mask_list``: per image, a (num_categories, L) bool tensor
        marking the tokens of each category, used to map a category back to its
        token span.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape
    device = input_ids.device

    is_special = torch.zeros((bs, num_token), dtype=torch.bool, device=device)
    for token_id in special_token_ids:
        is_special |= input_ids == token_id

    # Row-major order, so each row is processed in one contiguous sweep. The
    # per-row reset of ``previous_col`` is implicit: every row starts with [CLS]
    # at column 0, which takes the first branch and sets previous_col back to 0.
    idxs = torch.nonzero(is_special)

    attention_mask = torch.eye(num_token, dtype=torch.bool, device=device)[None].repeat(bs, 1, 1)
    position_ids = torch.zeros((bs, num_token), dtype=torch.long, device=device)
    cate_to_token_mask_list: List[List[Tensor]] = [[] for _ in range(bs)]

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col == 0 or col == num_token - 1:
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            block = slice(previous_col + 1, col + 1)
            attention_mask[row, block, block] = True
            position_ids[row, block] = torch.arange(0, col - previous_col, device=device)

            phrase_mask = torch.zeros(num_token, dtype=torch.bool, device=device)
            phrase_mask[previous_col + 1 : col] = True  # the separator itself is not part of the phrase
            cate_to_token_mask_list[row].append(phrase_mask)
        previous_col = col

    stacked = [
        torch.stack(masks, dim=0) if masks else torch.zeros((0, num_token), dtype=torch.bool, device=device)
        for masks in cate_to_token_mask_list
    ]
    return attention_mask, position_ids, stacked


def encode_text(
    tokenizer,
    bert: nn.Module,
    feat_map: nn.Module,
    captions: List[str],
    special_token_ids: List[int],
    max_text_len: int,
    sub_sentence_present: bool,
    device,
) -> Tuple[Dict[str, Tensor], object]:
    """Tokenize and encode a batch of captions.

    Returns:
        ``text_dict`` with ``encoded_text`` (bs, L, d_model), ``text_token_mask``
        (bs, L; ``True`` for real tokens), ``position_ids`` and
        ``text_self_attention_masks``; plus the raw tokenizer output, which the
        criterion needs to map category names back to token spans.
    """
    tokenized = tokenizer(captions, padding="longest", return_tensors="pt").to(device)

    text_self_attention_masks, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_token_ids
    )

    # A caption longer than the model's text budget is truncated rather than
    # rejected: with many categories in one prompt this is a normal occurrence.
    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        for key in ("input_ids", "attention_mask", "token_type_ids"):
            if key in tokenized:
                tokenized[key] = tokenized[key][:, :max_text_len]

    if sub_sentence_present:
        bert_inputs = {k: v for k, v in tokenized.items() if k != "attention_mask"}
        bert_inputs["attention_mask"] = text_self_attention_masks
        bert_inputs["position_ids"] = position_ids
    else:
        bert_inputs = dict(tokenized)

    bert_output = bert(**bert_inputs)
    encoded_text = feat_map(bert_output["last_hidden_state"])
    text_token_mask = tokenized.attention_mask.bool()

    if encoded_text.shape[1] > max_text_len:
        encoded_text = encoded_text[:, :max_text_len, :]
        text_token_mask = text_token_mask[:, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]

    text_dict = {
        "encoded_text": encoded_text,
        "text_token_mask": text_token_mask,
        "position_ids": position_ids,
        "text_self_attention_masks": text_self_attention_masks,
    }
    return text_dict, tokenized


__all__ = [
    "SPECIAL_TOKENS",
    "encode_text",
    "force_eager_attention",
    "generate_masks_with_special_tokens_and_transfer_map",
    "get_pretrained_language_model",
    "get_tokenizer",
]
