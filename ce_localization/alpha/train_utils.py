"""Tiện ích train ALPHA: DDP, lịch lr kiểu detectron2, chia batch tái lập được, checkpoint .pth."""

import bisect
import datetime
import os
import re

import torch
import torch.distributed as dist

from ce_localization.utils.checkpoint import CheckpointManager

__all__ = ["setup_dist", "PthCheckpoints", "warmup_multistep", "epoch_batches", "alpha_group",
           "noise_seed"]


def setup_dist():
    """-> (rank, world, dev). torchrun đặt WORLD_SIZE/RANK/LOCAL_RANK; không có thì 1 tiến trình.
    (cùng khuôn `ce_localization/legacy/train.py:66-78`)"""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world == 1:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        dev = torch.device("cpu")
    dist.init_process_group("nccl" if dev.type == "cuda" else "gloo",
                            timeout=datetime.timedelta(hours=1))
    return dist.get_rank(), world, dev


class PthCheckpoints(CheckpointManager):
    """last.pth / best.pth (quy tắc Kaggle), pickle KHÔNG zip, ghi nguyên tử."""
    LAST, BEST = "last.pth", "best.pth"

    @staticmethod
    def _atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp, _use_new_zipfile_serialization=False)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)


def warmup_multistep(it, steps, gamma=0.1, warmup_iters=1000, warmup_factor=0.01):
    """Hệ số lr ở iteration `it` — `WarmupMultiStepLR` của detectron2 (warmup tuyến tính):
    trong warmup: factor·(1−α) + α với α = it / warmup_iters ; sau đó gamma^(số mốc đã qua)."""
    f = gamma ** bisect.bisect_right(list(steps), it)
    if it < warmup_iters:
        a = it / warmup_iters
        f *= warmup_factor * (1 - a) + a
    return f


def epoch_batches(n, batch_per_rank, rank, world, seed, epoch):
    """Danh sách batch (list chỉ số) của MỘT epoch cho rank này. Hoán vị seed theo (seed, epoch)
    nên resume giữa epoch khớp đúng; bỏ phần lẻ để mọi rank có cùng số batch (như drop_last)."""
    g = torch.Generator().manual_seed(seed * 1000003 + epoch)
    perm = torch.randperm(n, generator=g).tolist()
    per_step = batch_per_rank * world
    n_steps = n // per_step
    out = []
    for s in range(n_steps):
        chunk = perm[s * per_step:(s + 1) * per_step]
        out.append(chunk[rank * batch_per_rank:(rank + 1) * batch_per_rank])
    return out


def noise_seed(seed, it, rank):
    """Seed của t / nhiễu khuếch tán ở iteration `it` — tái lập khi resume."""
    return (seed * 1000003 + it * 101 + rank) % (2 ** 63 - 1)


_STAGE = re.compile(r"head\.stages\.(\d+)\.(\w+)")


def alpha_group(name):
    """Tên tham số ALPHA -> nhóm cho GradMonitor."""
    if name.startswith("backbone.fpn."):
        return "fpn"
    if name.startswith("backbone."):
        return "resnet"
    if name.startswith("memory."):
        return "memory." + name.split(".")[1]
    m = _STAGE.match(name)
    if m:
        i, sub = int(m.group(1)), m.group(2)
        if sub in ("roi_proj", "decoder"):
            return f"{sub}[{i}]"
        return f"heads[{i}]"
    return "khác"
