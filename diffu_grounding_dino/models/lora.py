"""LoRA adapters for the pretrained GroundingDINO towers.

Alternative to full-finetune: freeze every parameter of a pretrained tower and add a
trainable low-rank update alongside each of its ``nn.Linear`` layers (Hu et al.,
LoRA). Scope here is deliberately narrow -- only ``backbone.0`` (Swin) and ``bert``
(BERT text encoder), the two towers that already share a naming convention with
``diff_warmup_freeze_keywords``/``lr_backbone_names`` elsewhere in this project.
Fusion, the deformable transformer, and the diffusion modules stay fully trainable
either way -- they either have no pretrained weight to protect (diffusion) or mix in
``nn.MultiheadAttention`` (no plain ``nn.Linear`` q/k/v to wrap).

"Freeze the tower" means ALL of its parameters, not just the ones inside an
``nn.Linear``: Swin also has a ``nn.Conv2d`` patch embedding, ``nn.LayerNorm``
throughout, and raw ``nn.Parameter`` tensors (relative position bias table,
optionally an absolute position embedding); BERT has ``nn.LayerNorm`` and
``nn.Embedding``. ``inject_lora`` freezes all of these explicitly -- see its
docstring.

Injection must happen at the right point relative to checkpoint loading -- see the
ordering comment in ``main.py`` around ``inject_lora``. In short: load a *plain*
pretrained checkpoint before injecting (key names must still be flat), but inject
*before* loading a checkpoint that was itself saved with LoRA already injected
(its keys are already ``...base.weight``/``...lora_A``).
"""

import math

import torch
import torch.nn as nn

from util.param_dicts import match_name_keywords


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a trainable low-rank update.

    ``lora_B`` is zero-initialised so the wrapped layer is a no-op at construction
    time -- output is bit-identical to the plain ``nn.Linear`` until training moves
    the adapter away from init.
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.scaling = alpha / rank
        # Match base.weight's device/dtype explicitly: injection runs AFTER
        # model.to(device) in main.py, and a bare torch.empty/torch.zeros here
        # would silently default to CPU -- fine on the CPU-only unit tests, but a
        # device-mismatch crash on the very first GPU forward pass.
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + update * self.scaling


def inject_lora(model: nn.Module, target_prefixes, rank: int, alpha: float, dropout: float = 0.0) -> int:
    """Replace every ``nn.Linear`` under ``target_prefixes`` with a ``LoRALinear``,
    and freeze every OTHER parameter under those prefixes too.

    The two towers this targets are not made of ``nn.Linear`` alone: Swin has a
    ``nn.Conv2d`` patch embedding, ``nn.LayerNorm`` throughout, and a raw
    ``relative_position_bias_table``/``absolute_pos_embed`` ``nn.Parameter``; BERT
    has ``nn.LayerNorm`` and ``nn.Embedding``. ``LoRALinear`` only freezes the
    ``nn.Linear`` it wraps -- left on its own, all of the above would stay fully
    trainable, which is not "backbone/bert frozen, only a low-rank adapter trains"
    but a partial, silently-incomplete freeze. So after wrapping the Linears, this
    also explicitly freezes every remaining parameter under ``target_prefixes``
    (skipping the ``lora_A``/``lora_B`` just added, which must stay trainable).

    Returns the number of ``nn.Linear`` layers wrapped. Mutates ``model`` in place.
    """
    if not target_prefixes:
        return 0

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and match_name_keywords(name, target_prefixes)
    ]
    for name, linear in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))

    for name, param in model.named_parameters():
        if "lora_" not in name and match_name_keywords(name, target_prefixes):
            param.requires_grad_(False)

    return len(targets)


__all__ = ["LoRALinear", "inject_lora"]
