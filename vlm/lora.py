"""LoRA tối giản cho finetune VLM (ZETA4.1) — không phụ thuộc `peft` (venv vLLM đã ghim, khỏi cài thêm).

`add_lora` bọc mọi `nn.Linear` có tên lá thuộc `targets` (trong phạm vi `scope`, vd chỉ LLM) bằng `LoRALinear`: y = W x + (α / r) · B A
drop(x); A ~ Kaiming, B = 0 ⇒ lúc khởi tạo ra y hệt model gốc. Chỉ A, B được train (tham số fp32, kể cả khi W bf16; dưới autocast
phép nhân LoRA chạy bf16 như phần còn lại, ngoài autocast thì fp32). `merge_lora` cộng
(α / r) · B A vào W rồi gỡ lớp bọc ⇒ model HF thường (lưu `save_pretrained` cho vLLM).
"""

import math

import torch
from torch import nn

__all__ = ["LoRALinear", "add_lora", "lora_parameters", "lora_state_dict", "load_lora", "merge_lora"]


def _autocast_on(device_type):
    try:
        return torch.is_autocast_enabled(device_type)
    except TypeError:                                  # torch < 2.4: không nhận device_type
        return torch.is_autocast_enabled() if device_type == "cuda" else torch.is_autocast_cpu_enabled()


class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.0):
        super().__init__()
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        y = self.base(x)
        z = self.drop(x)
        if not _autocast_on(z.device.type):            # autocast bf16: phép nhân tự chạy bf16, KHÔNG tạo / lưu bản sao fp32 của x
            z = z.to(self.lora_A.dtype)
        z = z @ self.lora_A.t() @ self.lora_B.t()
        return y + (self.scale * z).to(y.dtype)

    def delta(self):
        return self.scale * (self.lora_B @ self.lora_A)


def _parent(model, name):
    mod = model
    for p in name.split(".")[:-1]:
        mod = getattr(mod, p)
    return mod


def add_lora(model, targets, r, alpha, dropout=0.0, skip=()):
    """Đóng băng MỌI tham số rồi bọc các Linear có tên lá ∈ `targets`, trừ module có tên chứa một chuỗi trong `skip` (vd "visual" =
    tháp ảnh Qwen2.5-VL, cũng có gate / up / down_proj). -> số lớp đã bọc."""
    for p in model.parameters():
        p.requires_grad_(False)
    names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear) and n.rsplit(".", 1)[-1] in targets
             and not any(s in n for s in skip)]
    if not names:
        raise ValueError(f"không có nn.Linear nào tên {targets} (bỏ {skip}) để gắn LoRA")
    for n in names:
        setattr(_parent(model, n), n.rsplit(".", 1)[-1], LoRALinear(model.get_submodule(n), r, alpha, dropout))
    return len(names)


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))]


def lora_state_dict(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n.endswith(("lora_A", "lora_B"))}


def load_lora(model, sd):
    own = dict(model.named_parameters())
    miss = [k for k in own if k.endswith(("lora_A", "lora_B")) and k not in sd]
    extra = [k for k in sd if k not in own]
    if miss or extra:
        raise RuntimeError(f"LoRA không khớp model: thiếu {miss[:3]} ({len(miss)}), thừa {extra[:3]} ({len(extra)})")
    with torch.no_grad():
        for k, v in sd.items():
            own[k].copy_(v)


@torch.no_grad()
def merge_lora(model):
    """W += (α / r) B A (tính fp32 rồi về dtype của W), thay lớp bọc bằng Linear gốc. -> số lớp đã gộp."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for n in names:
        m = model.get_submodule(n)
        w = m.base.weight
        w.copy_((w.float() + m.delta().to(w.device)).to(w.dtype))
        setattr(_parent(model, n), n.rsplit(".", 1)[-1], m.base)
    return len(names)
