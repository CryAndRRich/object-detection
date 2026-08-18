"""Verification: LoRA injection is a no-op at init, freezes the right params, trains
the right params, and survives the inject-before-resume-load ordering main.py relies
on (see the comment block around ``inject_lora`` there).

    python tests/test_lora.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import train_one_epoch  # noqa: E402
from models.lora import LoRALinear, inject_lora  # noqa: E402
from tests.tiny import build_tiny_model, fake_batch  # noqa: E402
from util.param_dicts import get_param_dict, match_name_keywords  # noqa: E402

TARGETS = ["backbone.0", "bert"]


class _Args:
    use_diffusion = False
    amp = False
    debug = False
    onecyclelr = False
    diff_warmup_iters = 0
    diff_warmup_freeze_keywords = []
    use_lora = True


def test_injection_is_a_noop_at_init():
    torch.manual_seed(0)
    model, criterion, _, cfg = build_tiny_model(use_diffusion=False)
    model.eval()  # disable any stochastic depth / dropout so the forward is deterministic

    samples, targets = fake_batch(cfg, num_boxes=(3, 0))
    captions = [t["caption"] for t in targets]
    with torch.no_grad():
        before = model(samples, targets=targets, captions=captions)

    n_wrapped = inject_lora(model, TARGETS, rank=4, alpha=8)
    assert n_wrapped > 0, "expected at least one nn.Linear under backbone.0/bert to wrap"

    with torch.no_grad():
        after = model(samples, targets=targets, captions=captions)

    assert torch.equal(before["pred_boxes"], after["pred_boxes"]), "lora_B=0 must be an exact no-op at init"
    assert torch.equal(before["pred_logits"], after["pred_logits"]), "lora_B=0 must be an exact no-op at init"
    print("  ok    test_injection_is_a_noop_at_init")


def test_freezing_is_correct():
    model, _, _, _ = build_tiny_model(use_diffusion=False)
    inject_lora(model, TARGETS, rank=4, alpha=8)

    for name, param in model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"{name} should be trainable"
        elif match_name_keywords(name, TARGETS):
            # a LoRALinear's own base.weight/base.bias, or any other backbone.0/bert
            # parameter untouched by injection (e.g. norm layers) -- both frozen.
            assert not param.requires_grad, f"{name} should be frozen under lora"
    print("  ok    test_freezing_is_correct")


def test_training_step_only_moves_lora_params():
    torch.manual_seed(0)
    model, criterion, _, cfg = build_tiny_model(use_diffusion=False)
    inject_lora(model, TARGETS, rank=4, alpha=8)
    cfg.use_lora = True
    cfg.lora_lr = 1e-2

    base_before = {
        n: p.clone()
        for n, p in model.named_parameters()
        if "lora_" not in n and match_name_keywords(n, TARGETS)
    }
    lora_before = {n: p.clone() for n, p in model.named_parameters() if "lora_" in n}

    param_dicts = get_param_dict(cfg, model, include_frozen=True)
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.lr)

    batch = fake_batch(cfg, num_boxes=(3, 2))
    train_one_epoch(model, criterion, [batch], optimizer, torch.device("cpu"), epoch=0, args=_Args())

    for n, p in model.named_parameters():
        if "lora_" not in n and match_name_keywords(n, TARGETS):
            assert torch.equal(base_before[n], p), f"frozen base param {n} changed after optimizer.step()"

    moved = [n for n, p in model.named_parameters() if "lora_" in n and not torch.equal(lora_before[n], p)]
    assert moved, "expected at least one lora_A/lora_B to move after a training step"
    print("  ok    test_training_step_only_moves_lora_params")


def test_resume_ordering_inject_before_load():
    """Mirrors main.py: a checkpoint saved by an already-lora-injected model must be
    loadable into a freshly built model that has LoRA injected BEFORE the load."""
    torch.manual_seed(0)
    model_a, _, _, cfg = build_tiny_model(use_diffusion=False)
    inject_lora(model_a, TARGETS, rank=4, alpha=8)
    with torch.no_grad():
        for n, p in model_a.named_parameters():
            if "lora_" in n:
                p.add_(0.01)  # simulate a training step having moved the adapters
    saved_state = model_a.state_dict()

    model_b, _, _, _ = build_tiny_model(use_diffusion=False)
    inject_lora(model_b, TARGETS, rank=4, alpha=8)  # inject BEFORE loading, like a --resume
    model_b.load_state_dict(saved_state, strict=True)  # strict: any key mismatch fails loudly

    backbone_a = {n: p for n, p in model_a.named_parameters() if n.startswith("backbone.0")}
    backbone_b = dict(model_b.named_parameters())
    for n, p in backbone_a.items():
        assert torch.equal(p, backbone_b[n]), f"{n} did not transfer correctly across the resume-ordering load"
    print("  ok    test_resume_ordering_inject_before_load")


def main():
    test_injection_is_a_noop_at_init()
    test_freezing_is_correct()
    test_training_step_only_moves_lora_params()
    test_resume_ordering_inject_before_load()
    print("4/4 passed")


if __name__ == "__main__":
    main()
