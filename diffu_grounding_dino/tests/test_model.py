"""End-to-end model tests: verification steps 3, 4 and 6 of the plan.

Covers a training forward pass with diffusion, the DDIM sampler (including the
"encode once" guarantee), the non-diffusion baseline path, and the loss.

    python tests/test_model.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.tiny import build_tiny_model, fake_batch  # noqa: E402
from util.vl_utils import build_caption, category_char_spans, create_positive_map  # noqa: E402


# --------------------------------------------------------------------------- #
# 3. training forward pass
# --------------------------------------------------------------------------- #
def test_diffusion_training_forward():
    model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
    model.train()
    samples, targets = fake_batch(cfg)

    out = model(samples, targets=targets)

    assert out["pred_boxes"].shape == (2, cfg.num_queries, 4), out["pred_boxes"].shape
    assert out["pred_logits"].shape == (2, cfg.num_queries, cfg.max_text_len)
    assert out["diffusion_t"].shape == (2,) and out["diffusion_t"].dtype == torch.long
    assert out["diffusion_loss_weight"].shape == (2,)
    assert torch.isfinite(out["pred_boxes"]).all(), "non-finite boxes"
    assert (out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all(), "boxes must be normalized"

    assert len(out["aux_outputs"]) == cfg.dec_layers - 1
    assert "interm_outputs" in out, "the encoder proposal branch must still be supervised"
    assert torch.isfinite(out["interm_outputs"]["pred_boxes"]).all()


def test_diffusion_training_requires_targets():
    model, _, _, cfg = build_tiny_model(use_diffusion=True)
    model.train()
    samples, targets = fake_batch(cfg)
    try:
        model(samples, captions=[t["caption"] for t in targets])
    except ValueError as exc:
        assert "targets" in str(exc)
        return
    raise AssertionError("training with diffusion and no targets must fail loudly")


def test_loss_is_finite_and_backpropagates():
    model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
    model.train()
    samples, targets = fake_batch(cfg)

    out = model(samples, targets=targets)
    loss_dict = criterion(
        out,
        targets,
        [t["cap_list"] for t in targets],
        [t["caption"] for t in targets],
        t_weight=out["diffusion_loss_weight"],
    )

    for key in ("loss_ce", "loss_bbox", "loss_giou", "loss_ce_interm", "loss_bbox_interm"):
        assert key in loss_dict, f"missing {key}; got {sorted(loss_dict)}"
    for key, value in loss_dict.items():
        assert torch.isfinite(value).all(), f"{key} is not finite: {value}"

    weight_dict = criterion.weight_dict
    total = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)
    total.backward()

    # The FiLM output projection is zero-initialised, so at step 0 gradient reaches
    # the injector's own weights but cannot flow *through* them into time_embed.
    inject_grads = [p.grad for n, p in model.named_parameters() if "time_inject" in n and p.grad is not None]
    assert inject_grads, "no gradient reached time_inject"
    assert any(g.abs().sum() > 0 for g in inject_grads), "time_inject received only zero gradients"

    time_grads = [p.grad for n, p in model.named_parameters() if "time_embed" in n and p.grad is not None]
    assert time_grads, "time_embed is not connected to the graph at all"
    assert all(g.abs().sum() == 0 for g in time_grads), "unexpected time_embed gradient at init"


def test_time_embed_learns_after_the_first_step():
    """Verification #5's gradient check: non-zero once the injector has moved."""
    model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
    model.train()
    samples, targets = fake_batch(cfg, num_boxes=(2, 2))

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)

    def step():
        optimizer.zero_grad()
        out = model(samples, targets=targets)
        loss_dict = criterion(
            out,
            targets,
            [t["cap_list"] for t in targets],
            [t["caption"] for t in targets],
            t_weight=out["diffusion_loss_weight"],
        )
        total = sum(loss_dict[k] * criterion.weight_dict[k] for k in loss_dict if k in criterion.weight_dict)
        total.backward()
        optimizer.step()
        return total.item()

    step()  # moves the zero-initialised FiLM projection off zero
    step()

    time_grads = [p.grad for n, p in model.named_parameters() if "time_embed" in n and p.grad is not None]
    assert any(g.abs().sum() > 0 for g in time_grads), "time_embed still has no gradient after two steps"


def test_timestep_weighting_changes_the_loss():
    """A weighted loss must differ from an unweighted one, or the flag does nothing."""
    model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
    model.train()
    torch.manual_seed(0)
    samples, targets = fake_batch(cfg)

    with torch.no_grad():
        out = model(samples, targets=targets)
    args = (out, targets, [t["cap_list"] for t in targets], [t["caption"] for t in targets])

    weighted = criterion(*args, t_weight=out["diffusion_loss_weight"])["loss_bbox"].item()
    unweighted = criterion(*args, t_weight=None)["loss_bbox"].item()

    weights = out["diffusion_loss_weight"]
    if abs(weights.max() - weights.min()) < 1e-6:
        return  # both sampled timesteps happened to share a weight
    assert abs(weighted - unweighted) > 1e-9, "t_weight had no effect on loss_bbox"


def test_criterion_ignores_t_weight_when_disabled():
    model, criterion, _, cfg = build_tiny_model(use_diffusion=False)
    assert criterion.use_timestep_weighting is False
    model.train()
    samples, targets = fake_batch(cfg)

    with torch.no_grad():
        out = model(samples, targets=targets)
    args = (out, targets, [t["cap_list"] for t in targets], [t["caption"] for t in targets])
    a = criterion(*args, t_weight=torch.tensor([0.1, 5.0]))["loss_bbox"].item()
    b = criterion(*args, t_weight=None)["loss_bbox"].item()
    assert a == b, "the baseline criterion must ignore t_weight entirely"


# --------------------------------------------------------------------------- #
# 4. DDIM sampling
# --------------------------------------------------------------------------- #
def _count_encode_calls(model):
    counter = {"encode": 0, "decode": 0}
    real_encode = model.transformer.encode
    real_decode = model.transformer.decode

    def encode(*args, **kwargs):
        counter["encode"] += 1
        return real_encode(*args, **kwargs)

    def decode(*args, **kwargs):
        counter["decode"] += 1
        return real_decode(*args, **kwargs)

    model.transformer.encode = encode
    model.transformer.decode = decode
    return counter


def test_ddim_sample_encodes_once_per_image():
    """The whole reason for the encode/decode split."""
    for steps in (1, 3, 10):
        model, _, _, cfg = build_tiny_model(use_diffusion=True, diff_sampling_timesteps=steps)
        model.eval()
        samples, targets = fake_batch(cfg)
        counter = _count_encode_calls(model)

        with torch.no_grad():
            out = model(samples, captions=[t["caption"] for t in targets])

        assert counter["encode"] == 1, f"{steps} steps: encoder ran {counter['encode']} times"
        assert counter["decode"] == steps, f"{steps} steps: decoder ran {counter['decode']} times"
        assert out["pred_boxes"].shape == (2, cfg.num_queries, 4)
        assert torch.isfinite(out["pred_boxes"]).all(), f"{steps} steps produced non-finite boxes"
        assert (out["pred_boxes"] >= 0).all() and (out["pred_boxes"] <= 1).all()


def test_ddim_trajectory_is_recorded():
    model, _, _, cfg = build_tiny_model(use_diffusion=True, diff_sampling_timesteps=3)
    model.eval()
    samples, targets = fake_batch(cfg)

    text_dict, tokenized = model._encode_text([t["caption"] for t in targets], torch.device("cpu"))
    srcs, masks, poss = model._prepare_image_features(samples)
    with torch.no_grad():
        enc = model.transformer.encode(srcs, masks, poss, text_dict)
        out = model.ddim_sample(enc, tokenized, torch.device("cpu"), return_trajectory=True)

    trajectory = out["trajectory"]
    assert len(trajectory) == 3
    assert [step["t"] for step in trajectory] == sorted([step["t"] for step in trajectory], reverse=True)
    assert trajectory[0]["t"] == model.diffusion.num_timesteps - 1
    # Successive steps should not be identical: the sampler is meant to move.
    assert not torch.allclose(trajectory[0]["pred_boxes"], trajectory[-1]["pred_boxes"])


def test_deterministic_sampler_is_reproducible():
    model, _, _, cfg = build_tiny_model(use_diffusion=True, diff_sampling_timesteps=3, diff_ddim_eta=0.0)
    model.eval()
    samples, targets = fake_batch(cfg)
    captions = [t["caption"] for t in targets]

    torch.manual_seed(7)
    with torch.no_grad():
        first = model(samples, captions=captions)["pred_boxes"]
    torch.manual_seed(7)
    with torch.no_grad():
        second = model(samples, captions=captions)["pred_boxes"]
    assert torch.allclose(first, second), "eta=0 sampling should be reproducible under a fixed seed"


# --------------------------------------------------------------------------- #
# 6. the baseline path must be untouched by the refactor
# --------------------------------------------------------------------------- #
def test_baseline_has_no_diffusion_modules():
    model, _, _, _ = build_tiny_model(use_diffusion=False)
    names = [n for n, _ in model.named_parameters()]
    assert not any("time_embed" in n or "time_inject" in n for n in names), "baseline must not gain diffusion params"
    assert model.diffusion is None
    assert model.transformer.time_embed is None
    assert model.transformer.decoder.time_inject is None


def test_baseline_forward_matches_encode_decode_split():
    """``forward()`` must equal ``encode()`` then ``decode()``, exactly."""
    model, _, _, cfg = build_tiny_model(use_diffusion=False)
    model.eval()
    samples, targets = fake_batch(cfg)
    captions = [t["caption"] for t in targets]

    with torch.no_grad():
        via_model = model(samples, captions=captions)["pred_boxes"]

        text_dict, _ = model._encode_text(captions, torch.device("cpu"))
        srcs, masks, poss = model._prepare_image_features(samples)
        hs, refs, _, _, _ = model.transformer.forward(srcs, masks, None, poss, None, None, text_dict)
        _, coords = model._decode_predictions(hs, refs, text_dict)

    assert torch.allclose(via_model, coords[-1], atol=1e-6)


def test_baseline_and_diffusion_share_head_weights():
    """Only the reference-point source should differ between the two paths."""
    baseline, _, _, _ = build_tiny_model(use_diffusion=False)
    diffusion, _, _, _ = build_tiny_model(use_diffusion=True)

    base_keys = set(baseline.state_dict())
    diff_keys = set(diffusion.state_dict())
    extra = diff_keys - base_keys
    assert extra, "the diffusion model must add parameters"
    assert all("time_embed" in k or "time_inject" in k or "diffusion." in k for k in extra), sorted(extra)[:10]
    assert not (base_keys - diff_keys), "diffusion must not drop any baseline parameter"


def test_diffusion_extra_keys_are_covered_by_finetune_ignore():
    """The documented ``--finetune_ignore time_ diffusion`` must cover every new key."""
    baseline, _, _, _ = build_tiny_model(use_diffusion=False)
    diffusion, _, _, _ = build_tiny_model(use_diffusion=True)

    extra = set(diffusion.state_dict()) - set(baseline.state_dict())
    ignore_keywords = ["time_", "diffusion"]
    uncovered = [k for k in extra if not any(kw in k for kw in ignore_keywords)]
    assert not uncovered, f"these keys would be reported as missing on a finetune: {uncovered}"


# --------------------------------------------------------------------------- #
# post-processing and the category/token map
# --------------------------------------------------------------------------- #
def test_postprocess_shapes():
    model, _, postprocessors, cfg = build_tiny_model(use_diffusion=True)
    model.eval()
    samples, targets = fake_batch(cfg)

    with torch.no_grad():
        out = model(samples, captions=[t["caption"] for t in targets])
    sizes = torch.stack([t["size"] for t in targets]).float()
    results = postprocessors["bbox"](out, sizes)

    assert len(results) == 2
    for result in results:
        assert result["scores"].shape == (cfg.num_select,)
        assert result["labels"].shape == (cfg.num_select,)
        assert result["boxes"].shape == (cfg.num_select, 4)
        assert (result["scores"] >= 0).all() and (result["scores"] <= 1).all()
        assert (result["labels"] < len(cfg.label_list)).all(), "label index outside the eval vocabulary"
        assert torch.isfinite(result["boxes"]).all()
        # An untrained model happily predicts boxes hanging off the image, and
        # neither this post-processor nor upstream's clips them; only assert the
        # magnitude is in the right ballpark, which catches a missing rescale.
        assert result["boxes"].abs().max() < 4 * max(sizes[0].tolist())


def test_postprocess_scaling_is_exact():
    """Catches the classic h/w swap: cxcywh -> xyxy -> absolute pixels."""
    _, _, postprocessors, cfg = build_tiny_model(use_diffusion=False)
    post = postprocessors["bbox"]

    num_categories = len(cfg.label_list)
    logits = torch.full((1, 2, cfg.max_text_len), -20.0)
    # Make query 0 confidently the first category by lighting up its tokens.
    token_slots = post.positive_map[0].nonzero().flatten()
    logits[0, 0, token_slots] = 20.0

    boxes = torch.tensor([[[0.5, 0.5, 0.5, 0.25], [0.1, 0.1, 0.05, 0.05]]])
    results = post({"pred_logits": logits, "pred_boxes": boxes}, torch.tensor([[100.0, 200.0]]))

    top = results[0]["boxes"][0]
    expected = torch.tensor([50.0, 37.5, 150.0, 62.5])  # w=200 on x, h=100 on y
    assert torch.allclose(top, expected, atol=1e-4), f"{top.tolist()} != {expected.tolist()}"
    assert results[0]["labels"][0].item() == 0
    assert num_categories > 0


def test_positive_map_is_substring_safe():
    """The car/carrot trap: a substring search maps every 'car' onto 'carrot'."""
    from tests.tiny import make_tiny_tokenizer

    tokenizer = make_tiny_tokenizer()
    cat_list = ["carrot", "car"]
    caption = build_caption(cat_list)
    assert caption == "carrot . car ."

    spans = category_char_spans(cat_list, caption)
    assert spans == [(0, 6), (9, 12)], spans
    assert caption[spans[1][0] : spans[1][1]] == "car"

    tokenized = tokenizer(caption, return_tensors="pt")
    pos_map = create_positive_map(tokenized, [0, 1], cat_list, caption, max_text_len=32)

    carrot_tokens = pos_map[0].nonzero().flatten().tolist()
    car_tokens = pos_map[1].nonzero().flatten().tolist()
    assert carrot_tokens and car_tokens, (carrot_tokens, car_tokens)
    assert not set(carrot_tokens) & set(car_tokens), (
        f"'car' and 'carrot' share token slots {carrot_tokens} / {car_tokens} -- "
        "the positive map fell back to substring search"
    )

    naive_start = caption.find("car")
    assert naive_start == 0, "sanity: a naive find() really would resolve 'car' to 'carrot'"


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
