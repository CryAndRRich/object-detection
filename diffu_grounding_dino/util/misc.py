"""Misc helpers: nested tensors, distributed utilities, metric logging.

These are the standard DETR-family utilities, re-implemented here so the project
has no import dependency on any vendored reference repo.
"""

import datetime
import os
import pickle
import subprocess
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------- #
# numerics
# --------------------------------------------------------------------------- #
def _cpu_autocast_enabled() -> bool:
    """``is_autocast_enabled`` gained a device argument in torch 2.4."""
    try:
        return torch.is_autocast_enabled("cpu")
    except TypeError:
        return torch.is_autocast_cpu_enabled()


@contextmanager
def force_fp32():
    """Run a block outside autocast, whatever the caller enabled.

    Used for the two places where reduced precision is known to hurt: the
    diffusion schedule (whose buffers span ~10 orders of magnitude) and the
    decoder FFN.
    """
    if torch.is_autocast_enabled():
        with torch.autocast(device_type="cuda", enabled=False):
            yield
    elif _cpu_autocast_enabled():
        with torch.autocast(device_type="cpu", enabled=False):
            yield
    else:
        yield


# --------------------------------------------------------------------------- #
# math helpers
# --------------------------------------------------------------------------- #
def inverse_sigmoid(x: Tensor, eps: float = 1e-3) -> Tensor:
    """Logit function, the inverse of ``sigmoid``.

    ``x`` is expected in [0, 1]; it is clamped away from the open boundaries so
    the log never produces +/-inf. ``eps=1e-3`` matches GroundingDINO/Open-
    GroundingDino upstream; a smaller eps gives a larger logit magnitude at
    saturation (box-space boundaries, DDIM's noise-initialised boxes) and was not
    an intentional deviation.
    """
    x = x.clamp(min=0.0, max=1.0)
    x1 = x.clamp(min=eps)
    x2 = (1.0 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def interpolate(tensor, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """``F.interpolate`` that tolerates empty batches."""
    if tensor.numel() > 0:
        return F.interpolate(tensor, size, scale_factor, mode, align_corners)
    output_shape = _output_size(2, tensor, size, scale_factor)
    output_shape = list(tensor.shape[:-2]) + list(output_shape)
    return _new_empty_tensor(tensor, output_shape)


def _new_empty_tensor(x, shape):
    return torch.empty(shape, dtype=x.dtype, device=x.device)


def _output_size(dim, input, size, scale_factor):
    if size is not None:
        return size
    assert scale_factor is not None
    if not isinstance(scale_factor, (list, tuple)):
        scale_factor = [scale_factor] * dim
    return [int(input.shape[-dim + i] * scale_factor[i]) for i in range(dim)]


# --------------------------------------------------------------------------- #
# nested tensor (image batch + padding mask)
# --------------------------------------------------------------------------- #
class NestedTensor:
    """A batch of variable-sized images padded to a common shape.

    ``tensors``: (bs, 3, H, W); ``mask``: (bs, H, W) with ``True`` on padding.
    """

    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking: bool = False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        cast_mask = self.mask.to(device, non_blocking=non_blocking) if self.mask is not None else None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    @property
    def device(self):
        return self.tensors.device

    @property
    def shape(self):
        return self.tensors.shape

    def __repr__(self):
        return f"NestedTensor(tensors={tuple(self.tensors.shape)})"


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    """Pad a list of (3, Hi, Wi) tensors to a common (3, Hmax, Wmax)."""
    assert tensor_list[0].ndim == 3, "expected a list of CHW image tensors"

    max_size = [max(s) for s in zip(*[list(img.shape) for img in tensor_list])]
    batch_shape = [len(tensor_list)] + max_size
    b, _, h, w = batch_shape

    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
    mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        m[: img.shape[1], : img.shape[2]] = False
    return NestedTensor(tensor, mask)


def collate_fn(batch):
    """Collate ``(image, target)`` pairs into ``(NestedTensor, list_of_targets)``."""
    images, targets = list(zip(*batch))
    return nested_tensor_from_tensor_list(list(images)), list(targets)


def to_device(item, device):
    """Recursively move tensors inside lists/dicts to ``device``."""
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    if isinstance(item, list):
        return [to_device(v, device) for v in item]
    if isinstance(item, dict):
        return {k: to_device(v, device) for k, v in item.items()}
    return item


# --------------------------------------------------------------------------- #
# distributed
# --------------------------------------------------------------------------- #
def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_distributed(args):
    """Initialise ``torch.distributed`` from the usual environment variables."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = args.local_rank = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.local_rank = args.rank % torch.cuda.device_count()
        args.world_size = int(os.environ.get("SLURM_NTASKS", 1))
    else:
        print("Not using distributed mode")
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = args.gpu = 0
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    _disable_print_on_non_master(args.rank == 0)


def _disable_print_on_non_master(is_master: bool):
    import builtins

    builtin_print = builtins.print

    def print_wrapper(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print_wrapper


def _dist_device() -> str:
    """Device the active process group's collectives must run on.

    ``nccl`` (real multi-GPU training) only moves CUDA tensors; ``gloo`` (used
    here for CPU-only DDP smoke tests, see ``tests/test_ddp.py``) only moves CPU
    ones. Hardcoding ``"cuda"`` broke the moment anything ran distributed
    collectives on gloo -- this keeps the real training path byte-for-byte the
    same (still ``"cuda"`` whenever the backend is ``nccl``) while making the
    CPU debug path actually work instead of crashing on every sync point.
    """
    return "cuda" if dist.get_backend() == "nccl" else "cpu"


def all_gather(data):
    """All-gather arbitrary picklable objects across ranks."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    device = _dist_device()
    buffer = torch.ByteTensor(bytearray(pickle.dumps(data))).to(device)
    local_size = torch.tensor([buffer.numel()], device=device)
    size_list = [torch.zeros_like(local_size) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(s.item()) for s in size_list]
    max_size = max(size_list)

    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=device) for _ in size_list]
    if local_size.item() != max_size:
        padding = torch.empty((max_size - local_size.item(),), dtype=torch.uint8, device=device)
        buffer = torch.cat((buffer, padding), dim=0)
    dist.all_gather(tensor_list, buffer)

    return [pickle.loads(t.cpu().numpy().tobytes()[:size]) for size, t in zip(size_list, tensor_list)]


@torch.no_grad()
def reduce_dict(input_dict, average: bool = True):
    """Average (or sum) the scalar values of a dict across ranks."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    names = sorted(input_dict.keys())
    values = torch.stack([input_dict[k] for k in names], dim=0)
    dist.all_reduce(values)
    if average:
        values /= world_size
    return dict(zip(names, values))


# --------------------------------------------------------------------------- #
# checkpoint helpers
# --------------------------------------------------------------------------- #
def clean_state_dict(state_dict):
    """Strip a leading ``module.`` from DDP-saved checkpoints."""
    cleaned = type(state_dict)() if isinstance(state_dict, dict) else {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


def get_sha() -> str:
    """Short description of the current git state, for logging. Best-effort."""
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd, stderr=subprocess.DEVNULL).decode("ascii").strip()

    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        dirty = "has uncommitted changes" if _run(["git", "diff-index", "HEAD"]) else "clean"
        return f"sha: {sha}, status: {dirty}, branch: {branch}"
    except Exception:
        return "sha: N/A, status: unknown, branch: N/A"


# --------------------------------------------------------------------------- #
# metric logging
# --------------------------------------------------------------------------- #
class SmoothedValue:
    """Tracks a series of values, exposing a windowed median/avg plus a global avg."""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=_dist_device())
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0].item())
        self.total = t[1].item()

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max, value=self.value
        )


class MetricLogger:
    """Accumulates named scalars and prints a progress line every ``print_freq`` steps."""

    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f"meter {k} must be a scalar, got {type(v)}"
            self.meters[k].update(v)

    def add_meter(self, name: str, meter: SmoothedValue):
        self.meters[name] = meter

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"{type(self).__name__} has no attribute {attr}")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def log_every(self, iterable, print_freq: int, header: str = "", logger=None):
        printer = logger.info if logger is not None else print
        start = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space = ":" + str(len(str(len(iterable) if hasattr(iterable, "__len__") else 0))) + "d"
        parts = [
            header,
            "[{0" + space + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            parts.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(parts)
        mega_byte = 1024.0 * 1024.0

        total = len(iterable) if hasattr(iterable, "__len__") else 0
        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total - 1:
                eta_seconds = iter_time.global_avg * (total - i)
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                fields = dict(eta=eta, meters=str(self), time=str(iter_time), data=str(data_time))
                if torch.cuda.is_available():
                    fields["memory"] = torch.cuda.max_memory_allocated() / mega_byte
                printer(log_msg.format(i, total, **fields))
            end = time.time()

        total_time = time.time() - start
        printer(
            f"{header} Total time: {datetime.timedelta(seconds=int(total_time))} "
            f"({total_time / max(total, 1):.4f} s / it)"
        )


class BestMetricHolder:
    """Remembers the best value of a monitored metric and the epoch it came from."""

    def __init__(self, init_res: float = 0.0, better: str = "large"):
        assert better in ("large", "small")
        self.better = better
        self.best = init_res
        self.best_epoch = -1

    def update(self, value: float, epoch: int) -> bool:
        improved = value > self.best if self.better == "large" else value < self.best
        if improved:
            self.best = value
            self.best_epoch = epoch
        return improved

    def summary(self):
        return {"best": self.best, "best_epoch": self.best_epoch}

    def __str__(self):
        return f"best={self.best:.4f} @epoch{self.best_epoch}"
