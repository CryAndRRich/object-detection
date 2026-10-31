"""Tiện ích nhiều GPU của finetune (chỉ phụ thuộc torch)."""

import torch
import torch.distributed as dist

__all__ = ["all_sum", "sync_grads"]


def all_sum(x, world):
    """Tổng qua các GPU (bản sao)."""
    if world > 1:
        x = x.clone()
        dist.all_reduce(x)
    return x


def sync_grads(params, world):
    """TB grad qua các GPU (một all_reduce gộp phẳng; grad None = 0). Gọi một lần mỗi bước tối ưu, sau cộng dồn; grad inf ở một GPU lan
    sang mọi GPU ⇒ GradScaler mọi rank cùng bỏ bước."""
    if world == 1:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for p, g in zip(params, torch._utils._unflatten_dense_tensors(flat, grads)):
        if p.grad is None:
            p.grad = g.clone()
        else:
            p.grad.copy_(g)
