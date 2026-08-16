"""Optimizer parameter groups and the diffusion warm-up freeze.

Two related concerns live here.

**Parameter groups.** Backbone/BERT get a much smaller learning rate than the
rest of the model; the deformable-attention sampling offsets and the reference
point head get their own group as well. Mirrors the ``ddetr_in_mmdet`` scheme of
the config so numbers stay comparable with the non-diffusion baseline.

**Warm-up freeze.** The timestep modules start out as the identity (see
``models/diffusion/timestep.py``), so on the first steps the diffusion branch
feeds the decoder near-random reference points while the freshly initialised
conditioning has nothing to say. Letting gradients from that into a pretrained
Swin/BERT is what wrecks a finetune. So for the first ``diff_warmup_iters``
steps those two towers are frozen.

The freeze is driven by the *global step*, recomputed every iteration, which
makes it resume-safe: a job restarted at step 5000 with a 2000-step warm-up sees
an unfrozen model immediately, with no flag to persist. The frozen parameters
still enter the optimizer at construction time (``include_frozen=True``) so that
unfreezing later actually does something -- a parameter filtered out of the
param groups is gone for the whole run.
"""

import json
from typing import List

import torch.nn as nn


def match_name_keywords(name: str, keywords: List[str]) -> bool:
    return any(kw in name for kw in (keywords or []))


def get_param_dict(args, model: nn.Module, include_frozen: bool = True):
    """Build AdamW parameter groups.

    Args:
        include_frozen: keep parameters whose ``requires_grad`` is currently
            ``False``. Required when a warm-up will unfreeze them later.
    """
    param_dict_type = getattr(args, "param_dict_type", "default")
    assert param_dict_type in ("default", "ddetr_in_mmdet", "large_wd"), param_dict_type

    def selected(predicate):
        return [
            p
            for n, p in model.named_parameters()
            if predicate(n) and (include_frozen or p.requires_grad)
        ]

    if param_dict_type == "default":
        return [
            {"params": selected(lambda n: "backbone" not in n), "lr": args.lr},
            {"params": selected(lambda n: "backbone" in n), "lr": args.lr_backbone},
        ]

    if param_dict_type == "ddetr_in_mmdet":
        backbone_names = args.lr_backbone_names
        proj_names = args.lr_linear_proj_names
        return [
            {
                "params": selected(
                    lambda n: not match_name_keywords(n, backbone_names)
                    and not match_name_keywords(n, proj_names)
                ),
                "lr": args.lr,
            },
            {
                "params": selected(lambda n: match_name_keywords(n, backbone_names)),
                "lr": args.lr_backbone,
            },
            {
                # Named "_mult" in the config but used as an absolute learning
                # rate, both here and upstream. Kept as-is so that runs remain
                # numerically comparable to the published baseline.
                "params": selected(lambda n: match_name_keywords(n, proj_names)),
                "lr": args.lr_linear_proj_mult,
            },
        ]

    # large_wd: no weight decay on norms and biases
    def is_norm_or_bias(n):
        return match_name_keywords(n, ["norm", "bias"])

    return [
        {
            "params": selected(lambda n: "backbone" not in n and not is_norm_or_bias(n)),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        },
        {
            "params": selected(lambda n: "backbone" not in n and is_norm_or_bias(n)),
            "lr": args.lr,
            "weight_decay": 0.0,
        },
        {
            "params": selected(lambda n: "backbone" in n and not is_norm_or_bias(n)),
            "lr": args.lr_backbone,
            "weight_decay": args.weight_decay,
        },
        {
            "params": selected(lambda n: "backbone" in n and is_norm_or_bias(n)),
            "lr": args.lr_backbone,
            "weight_decay": 0.0,
        },
    ]


def apply_freeze_keywords(model: nn.Module, keywords: List[str]):
    """Permanently freeze every parameter whose name contains any keyword."""
    if not keywords:
        return
    for name, param in model.named_parameters():
        if match_name_keywords(name, keywords):
            param.requires_grad_(False)


def apply_diffusion_warmup(args, model: nn.Module, global_step: int) -> bool:
    """Freeze/unfreeze the warm-up towers according to ``global_step``.

    Idempotent and stateless -- safe to call every iteration and after a resume.
    Parameters frozen permanently by ``freeze_keywords`` are never revived.

    Returns:
        ``True`` while the warm-up freeze is active.
    """
    warmup_iters = int(getattr(args, "diff_warmup_iters", 0) or 0)
    keywords = getattr(args, "diff_warmup_freeze_keywords", None)
    if not getattr(args, "use_diffusion", False) or warmup_iters <= 0 or not keywords:
        return False

    permanent = getattr(args, "freeze_keywords", None) or []
    frozen = global_step < warmup_iters
    for name, param in model.named_parameters():
        if not match_name_keywords(name, keywords):
            continue
        if match_name_keywords(name, permanent):
            param.requires_grad_(False)
            continue
        param.requires_grad_(not frozen)
    return frozen


def describe_trainable(model: nn.Module, indent: int = 2) -> str:
    """JSON summary of trainable parameter counts, for the log."""
    return json.dumps(
        {n: p.numel() for n, p in model.named_parameters() if p.requires_grad},
        indent=indent,
    )
