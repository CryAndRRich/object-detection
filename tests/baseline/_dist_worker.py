"""Tiến trình con của `test_sync_grads_two_processes` (chỉ import torch + baseline.dist ⇒ khởi động nhanh)."""

import torch

from baseline.dist import sync_grads


def sync_worker(rank, path):
    torch.distributed.init_process_group("gloo", init_method=f"file://{path}", rank=rank, world_size=2)
    a, b = torch.nn.Parameter(torch.zeros(3)), torch.nn.Parameter(torch.zeros(2))
    a.grad = torch.full((3,), float(rank + 1))
    b.grad = None if rank == 0 else torch.tensor([2.0, float("inf")])   # grad None = 0; inf lan sang mọi rank
    sync_grads([a, b], 2)
    assert torch.equal(a.grad, torch.full((3,), 1.5)) and b.grad[0] == 1.0 and torch.isinf(b.grad[1])
    torch.distributed.destroy_process_group()
