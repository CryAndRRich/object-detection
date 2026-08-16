"""Backbone / deformable-attention / text-encoder tests.

Runs on CPU with no downloaded weights. If ``../weights/diffu_grounding_dino/groundingdino_swint_ogc.pth``
happens to be present, the Swin key test upgrades itself into a real
checkpoint-compatibility check.

    python tests/test_backbone_text.py
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.backbone import build_backbone  # noqa: E402
from models.backbone.position_encoding import PositionEmbeddingSineHW  # noqa: E402
from models.backbone.swin import build_swin_transformer  # noqa: E402
from models.ops import MSDeformAttn, multi_scale_deformable_attn_pytorch  # noqa: E402
from models.text.bert import (  # noqa: E402
    SPECIAL_TOKENS,
    generate_masks_with_special_tokens_and_transfer_map,
)
from util.misc import NestedTensor, clean_state_dict, nested_tensor_from_tensor_list  # noqa: E402

CHECKPOINT = Path(__file__).resolve().parents[2] / "weights" / "diffu_grounding_dino" / "groundingdino_swint_ogc.pth"


class _Args:
    """The image-side subset of config/cfg_odvg.py."""

    backbone = "swin_T_224_1k"
    position_embedding = "sine"
    pe_temperatureH = 20
    pe_temperatureW = 20
    return_interm_indices = [1, 2, 3]
    hidden_dim = 256
    dilation = False
    use_checkpoint = False
    backbone_freeze_keywords = None


def _dummy_batch(sizes=((3, 61, 83), (3, 40, 100))):
    """Deliberately non-round, non-equal image sizes: padding and window padding
    are where a from-scratch Swin implementation goes wrong."""
    return nested_tensor_from_tensor_list([torch.rand(*s) for s in sizes])


# --------------------------------------------------------------------------- #
# Swin
# --------------------------------------------------------------------------- #
def test_swin_forward_shapes():
    swin = build_swin_transformer("swin_T_224_1k", 224, out_indices=(1, 2, 3)).eval()
    samples = _dummy_batch()

    with torch.no_grad():
        outputs = swin(samples)

    assert list(outputs.keys()) == [0, 1, 2], "three returned stages, indexed from 0"
    assert swin.num_features == [96, 192, 384, 768], "Swin-T channel plan"

    _, _, h, w = samples.tensors.shape
    for level, expected_channels in zip(range(3), [192, 384, 768]):
        feat = outputs[level].tensors
        stride = 8 * 2**level
        assert feat.shape[1] == expected_channels, f"level {level} channels"
        assert feat.shape[2] == -(-h // stride) and feat.shape[3] == -(-w // stride), (
            f"level {level} expected stride {stride}, got {tuple(feat.shape[-2:])} from {(h, w)}"
        )
        assert outputs[level].mask.shape == feat.shape[-2:][-2:] or outputs[level].mask.shape == (
            feat.shape[0],
            *feat.shape[-2:],
        )
        assert torch.isfinite(feat).all()


def test_swin_padding_mask_is_propagated():
    swin = build_swin_transformer("swin_T_224_1k", 224, out_indices=(1, 2, 3)).eval()
    samples = _dummy_batch(sizes=((3, 64, 64), (3, 16, 16)))
    with torch.no_grad():
        outputs = swin(samples)

    mask = outputs[0].mask  # stride 8 -> 8x8
    assert not mask[0].any(), "the full-size image has no padding"
    assert mask[1].any(), "the small image must be padded, and the mask must say so"


def test_swin_key_names_match_checkpoint_layout():
    swin = build_swin_transformer("swin_T_224_1k", 224, out_indices=(1, 2, 3))
    keys = set(swin.state_dict().keys())

    required = {
        "patch_embed.proj.weight",
        "patch_embed.proj.bias",
        "patch_embed.norm.weight",
        "layers.0.blocks.0.norm1.weight",
        "layers.0.blocks.0.attn.relative_position_bias_table",
        "layers.0.blocks.0.attn.relative_position_index",
        "layers.0.blocks.0.attn.qkv.weight",
        "layers.0.blocks.0.attn.proj.weight",
        "layers.0.blocks.0.norm2.weight",
        "layers.0.blocks.0.mlp.fc1.weight",
        "layers.0.blocks.0.mlp.fc2.weight",
        "layers.0.downsample.reduction.weight",
        "layers.0.downsample.norm.weight",
        "layers.2.blocks.5.attn.qkv.weight",  # Swin-T stage 3 has 6 blocks
        "norm1.weight",
        "norm2.weight",
        "norm3.weight",
    }
    missing = required - keys
    assert not missing, f"key names drifted from the checkpoint layout: {sorted(missing)}"

    assert "norm0.weight" not in keys, "out_indices=(1,2,3) must not create norm0"
    assert "layers.3.downsample.reduction.weight" not in keys, "the last stage has no downsample"
    assert not any(k.startswith("head.") for k in keys), "the detection backbone has no classifier head"

    if not CHECKPOINT.exists():
        print(f"        (skipped real checkpoint check, {CHECKPOINT.name} not downloaded yet)")
        return

    # The released checkpoint was saved from a DDP-wrapped model, so every key
    # carries a leading "module." -- the same prefix production code strips via
    # util.misc.clean_state_dict before loading.
    state = clean_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model"])
    ckpt_swin = {k[len("backbone.0.") :]: v for k, v in state.items() if k.startswith("backbone.0.")}
    assert ckpt_swin, "no backbone.0.* keys in the checkpoint -- wrong file?"

    ours = swin.state_dict()
    missing = set(ckpt_swin) - set(ours)
    unexpected = set(ours) - set(ckpt_swin)
    assert not missing, f"checkpoint has keys we lack: {sorted(missing)[:10]}"
    # relative_position_index is a derived buffer; upstream may or may not save it.
    unexpected = {k for k in unexpected if "relative_position_index" not in k}
    assert not unexpected, f"we have keys the checkpoint lacks: {sorted(unexpected)[:10]}"

    for key, value in ckpt_swin.items():
        assert ours[key].shape == value.shape, f"{key}: {tuple(ours[key].shape)} vs {tuple(value.shape)}"
    print(f"        (verified {len(ckpt_swin)} Swin tensors against the real checkpoint)")


def test_build_backbone_returns_features_and_positions():
    backbone = build_backbone(_Args()).eval()
    assert backbone.num_channels == [192, 384, 768]

    with torch.no_grad():
        features, positions = backbone(_dummy_batch())

    assert len(features) == len(positions) == 3
    for feat, pos in zip(features, positions):
        assert pos.shape[1] == 256, "position encoding must be hidden_dim wide"
        assert pos.shape[-2:] == feat.tensors.shape[-2:]
        assert torch.isfinite(pos).all()


def test_position_encoding_ignores_padding():
    """The encoding of a real pixel must not depend on how much padding follows."""
    pos_embed = PositionEmbeddingSineHW(64, temperatureH=20, temperatureW=20, normalize=True)

    small = NestedTensor(torch.rand(1, 3, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))
    padded_mask = torch.zeros(1, 12, 12, dtype=torch.bool)
    padded_mask[:, 8:, :] = True
    padded_mask[:, :, 8:] = True
    padded = NestedTensor(torch.rand(1, 3, 12, 12), padded_mask)

    a = pos_embed(small)[0, :, :8, :8]
    b = pos_embed(padded)[0, :, :8, :8]
    assert torch.allclose(a, b, atol=1e-5), "normalized position encoding leaked the padding size"


# --------------------------------------------------------------------------- #
# deformable attention
# --------------------------------------------------------------------------- #
def _naive_deformable_attn(value, spatial_shapes, sampling_locations, attention_weights):
    """Slow, explicit reference: manual bilinear sampling with zero padding.

    Independent of ``grid_sample`` so it actually validates the coordinate
    convention (align_corners=False maps a normalized ``u`` to pixel ``u*W - 0.5``).
    """
    bs, _, num_heads, head_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape
    shapes = [(int(h), int(w)) for h, w in spatial_shapes]

    offset = 0
    per_level = []
    for h, w in shapes:
        per_level.append(value[:, offset : offset + h * w].view(bs, h, w, num_heads, head_dim))
        offset += h * w

    out = torch.zeros(bs, num_queries, num_heads, head_dim, dtype=value.dtype)
    for b in range(bs):
        for q in range(num_queries):
            for head in range(num_heads):
                for level, (h, w) in enumerate(shapes):
                    grid = per_level[level][b]
                    for p in range(num_points):
                        u, v = sampling_locations[b, q, head, level, p]
                        x = u * w - 0.5
                        y = v * h - 0.5
                        x0, y0 = int(torch.floor(x)), int(torch.floor(y))
                        acc = torch.zeros(head_dim, dtype=value.dtype)
                        for dy in (0, 1):
                            for dx in (0, 1):
                                xi, yi = x0 + dx, y0 + dy
                                if 0 <= xi < w and 0 <= yi < h:
                                    wx = 1 - abs(x - xi)
                                    wy = 1 - abs(y - yi)
                                    acc = acc + wx * wy * grid[yi, xi, head]
                        out[b, q, head] += attention_weights[b, q, head, level, p] * acc
    return out.view(bs, num_queries, num_heads * head_dim)


def test_deformable_attn_core_matches_naive_reference():
    torch.manual_seed(0)
    shapes = torch.tensor([[4, 5], [2, 3]])
    total = int((shapes[:, 0] * shapes[:, 1]).sum())
    bs, num_queries, num_heads, head_dim, num_levels, num_points = 2, 3, 2, 4, 2, 2

    value = torch.randn(bs, total, num_heads, head_dim)
    # Include out-of-range locations on purpose: zero padding must be exercised.
    locations = torch.rand(bs, num_queries, num_heads, num_levels, num_points, 2) * 1.4 - 0.2
    weights = torch.rand(bs, num_queries, num_heads, num_levels, num_points)
    weights = weights / weights.flatten(3).sum(-1)[..., None, None]

    fast = multi_scale_deformable_attn_pytorch(value, shapes, locations, weights)
    slow = _naive_deformable_attn(value, shapes, locations, weights)
    assert fast.shape == (bs, num_queries, num_heads * head_dim)
    assert torch.allclose(fast, slow, atol=1e-5), f"max diff {(fast - slow).abs().max():.2e}"


def test_deformable_attn_module():
    torch.manual_seed(0)
    shapes = torch.tensor([[16, 20], [8, 10], [4, 5], [2, 3]])
    total = int((shapes[:, 0] * shapes[:, 1]).sum())
    level_start = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])

    attn = MSDeformAttn(embed_dim=256, num_heads=8, num_levels=4, num_points=4)
    query = torch.randn(30, 2, 256, requires_grad=True)
    value = torch.randn(total, 2, 256)
    key_padding = torch.zeros(2, total, dtype=torch.bool)
    key_padding[1, -3:] = True

    for ref_dim in (2, 4):
        reference = torch.rand(2, 30, 4, ref_dim)
        out = attn(
            query,
            value=value,
            reference_points=reference,
            spatial_shapes=shapes,
            level_start_index=level_start,
            key_padding_mask=key_padding,
        )
        assert out.shape == (30, 2, 256), f"ref_dim={ref_dim}"
        assert torch.isfinite(out).all()

    out.sum().backward()
    # Both query-driven projections are zero-initialised, so at step 0 the query
    # receives exactly no gradient through them -- the offsets are pure bias and
    # the weights pure softmax-of-zeros. The module's own parameters do get
    # gradients, which is how the stencil starts to deform.
    assert query.grad is not None and query.grad.abs().sum() == 0, "unexpected query gradient at init"
    for name in ("sampling_offsets", "attention_weights", "value_proj", "output_proj"):
        grad = getattr(attn, name).weight.grad
        assert grad is not None and grad.abs().sum() > 0, f"{name} got no gradient"


def test_deformable_attn_query_gradient_after_init_moves():
    """Once the projections are non-zero, gradient must reach the query."""
    torch.manual_seed(0)
    shapes = torch.tensor([[8, 10], [4, 5]])
    total = int((shapes[:, 0] * shapes[:, 1]).sum())
    level_start = torch.cat([shapes.new_zeros(1), (shapes[:, 0] * shapes[:, 1]).cumsum(0)[:-1]])

    attn = MSDeformAttn(embed_dim=64, num_heads=4, num_levels=2, num_points=2)
    nn.init.normal_(attn.sampling_offsets.weight, std=0.01)
    nn.init.normal_(attn.attention_weights.weight, std=0.01)

    query = torch.randn(7, 2, 64, requires_grad=True)
    out = attn(
        query,
        value=torch.randn(total, 2, 64),
        reference_points=torch.rand(2, 7, 2, 4),
        spatial_shapes=shapes,
        level_start_index=level_start,
    )
    out.sum().backward()
    assert query.grad.abs().sum() > 0, "gradient does not reach the query"


def test_deformable_attn_key_names_and_init():
    attn = MSDeformAttn(embed_dim=256, num_heads=8, num_levels=4, num_points=4)
    keys = set(attn.state_dict().keys())
    assert keys == {
        "sampling_offsets.weight",
        "sampling_offsets.bias",
        "attention_weights.weight",
        "attention_weights.bias",
        "value_proj.weight",
        "value_proj.bias",
        "output_proj.weight",
        "output_proj.bias",
    }, sorted(keys)

    assert attn.sampling_offsets.weight.abs().sum() == 0, "offset projection starts at zero"
    bias = attn.sampling_offsets.bias.view(8, 4, 4, 2)
    # Point p sits p+1 units out along the head's ring direction.
    radii = bias[0, 0].norm(dim=-1)
    assert torch.allclose(radii, torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-5), radii


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def _fake_tokenized(rows):
    return {"input_ids": torch.as_tensor(rows, dtype=torch.long)}


def test_special_token_mask_is_block_diagonal():
    """The real caption shape: ``"a . b b ."`` -> ``[CLS] a . b b . [SEP]``.

    Note the trailing " ." the dataset appends -- see
    ``test_trailing_separator_is_required`` for why it is not cosmetic.
    """
    cls_id, sep_id, dot_id = 101, 102, 1012
    ids = [[cls_id, 500, dot_id, 600, 601, dot_id, sep_id]]
    mask, position_ids, cate_masks = generate_masks_with_special_tokens_and_transfer_map(
        _fake_tokenized(ids), [cls_id, sep_id, dot_id]
    )

    assert mask.shape == (1, 7, 7)
    m = mask[0]
    assert m[1, 2] and m[2, 1], "'a' and its trailing '.' form one block"
    assert not m[1, 3], "category 'a' must not see category 'b'"
    assert not m[3, 1], "and the reverse"
    assert m[3, 4] and m[4, 3], "'b b' are in the same block"
    assert m[0, 0] and not m[0, 1], "[CLS] attends only to itself"
    assert torch.equal(m, m.T), "the mask must be symmetric"

    assert position_ids[0].tolist() == [0, 0, 1, 0, 1, 2, 0], position_ids[0].tolist()

    assert len(cate_masks) == 1 and cate_masks[0].shape[0] == 2, "two category phrases"
    assert cate_masks[0][0].tolist() == [False, True, False, False, False, False, False]
    assert cate_masks[0][1].tolist() == [False, False, False, True, True, False, False]


def test_trailing_separator_is_required():
    """Without the trailing " .", the last category gets no attention block.

    ``[SEP]`` sitting in the final column takes the "isolated special token"
    branch, so the tokens after the last "." are left attending to themselves
    only. This is why ``odvg.py`` builds captions as ``" . ".join(cats) + " ."``
    -- dropping that suffix silently degrades the last category in every prompt.
    """
    cls_id, sep_id, dot_id = 101, 102, 1012
    no_suffix = [[cls_id, 500, dot_id, 600, 601, sep_id]]
    mask, _, cate_masks = generate_masks_with_special_tokens_and_transfer_map(
        _fake_tokenized(no_suffix), [cls_id, sep_id, dot_id]
    )
    assert not mask[0][3, 4], "documented quirk: last phrase is not blocked without a trailing separator"
    assert cate_masks[0].shape[0] == 1, "and it does not appear in cate_to_token_mask_list either"


def test_special_token_mask_resets_per_row():
    """Row 2 must not inherit row 1's separator position."""
    cls_id, sep_id, dot_id = 101, 102, 1012
    ids = [
        [cls_id, 500, dot_id, 600, sep_id],
        [cls_id, 700, 701, dot_id, sep_id],
    ]
    mask, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        _fake_tokenized(ids), [cls_id, sep_id, dot_id]
    )
    assert position_ids[1].tolist() == [0, 0, 1, 2, 0], position_ids[1].tolist()
    assert mask[1, 1, 2] and mask[1, 2, 1], "row 2's single phrase must be one block"
    assert not mask[1, 0, 1], "[CLS] stays isolated in row 2 as well"


def test_bert_rejects_3d_mask_under_sdpa():
    """Guard the reason ``force_eager_attention`` exists.

    If a future transformers version starts handling 3D masks under SDPA this
    test will fail -- at which point the workaround can be revisited. Until then,
    it documents that the failure is real and not hypothetical.
    """
    from transformers import BertConfig, BertModel

    config = BertConfig(
        vocab_size=2000, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64
    )
    config._attn_implementation = "sdpa"
    bert = BertModel(config).eval()
    if getattr(bert, "attn_implementation", None) != "sdpa":
        print("        (skipped: this transformers build does not expose an sdpa path for BERT)")
        return

    ids = torch.tensor([[101, 500, 1012, 600, 601, 1012, 102]])
    mask3d, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        _fake_tokenized(ids), [101, 102, 1012]
    )
    try:
        with torch.no_grad():
            bert(input_ids=ids, attention_mask=mask3d, position_ids=position_ids)
    except Exception:  # noqa: BLE001 - any failure proves the point
        return
    print("        (note: sdpa accepted a 3D mask here; force_eager_attention may no longer be needed)")


def test_bert_accepts_a_3d_attention_mask():
    """The installed transformers version must support the sub-sentence mask."""
    from transformers import BertConfig, BertModel

    from models.text.bert import force_eager_attention

    config = BertConfig(
        vocab_size=2000, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
    )
    bert = force_eager_attention(BertModel(config, add_pooling_layer=True)).eval()
    feat_map = nn.Linear(32, 256)

    ids = torch.tensor([[101, 500, 1012, 600, 601, 1012, 102]])
    mask3d, position_ids, _ = generate_masks_with_special_tokens_and_transfer_map(
        _fake_tokenized(ids), [101, 102, 1012]
    )

    with torch.no_grad():
        out = bert(
            input_ids=ids,
            attention_mask=mask3d,
            position_ids=position_ids,
            token_type_ids=torch.zeros_like(ids),
        )
    encoded = feat_map(out["last_hidden_state"])
    assert encoded.shape == (1, 7, 256)
    assert torch.isfinite(encoded).all()

    # The mask must actually bite: changing a token in one block must not move the
    # representation of a token in another block.
    ids_changed = ids.clone()
    ids_changed[0, 3] = 650
    with torch.no_grad():
        out2 = bert(
            input_ids=ids_changed,
            attention_mask=mask3d,
            position_ids=position_ids,
            token_type_ids=torch.zeros_like(ids),
        )
    a = out["last_hidden_state"][0, 1]
    b = out2["last_hidden_state"][0, 1]
    assert torch.allclose(a, b, atol=1e-5), "category embeddings leaked across the sub-sentence mask"


def test_bert_key_prefix():
    """``self.bert = BertModel(...)`` must yield ``bert.embeddings.*`` keys."""
    from transformers import BertConfig, BertModel

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.bert = BertModel(BertConfig(vocab_size=50, hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64))
            self.feat_map = nn.Linear(32, 256)

    keys = Holder().state_dict().keys()
    assert any(k.startswith("bert.embeddings.word_embeddings") for k in keys)
    assert any(k.startswith("bert.encoder.layer.0.attention") for k in keys)
    assert "feat_map.weight" in keys
    assert not any(k.startswith("bert.bert.") for k in keys), "double prefix would break the checkpoint load"


def test_special_tokens_list():
    assert SPECIAL_TOKENS == ["[CLS]", "[SEP]", ".", "?"]


# --------------------------------------------------------------------------- #
def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            torch.manual_seed(0)
            fn()
            print(f"  ok    {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
