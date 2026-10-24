"""Tiện ích train: DDP, lịch lr, chia batch tái lập được, checkpoint .pth ghi nguyên tử, đỉnh bộ nhớ GPU."""

import math
import datetime
import inspect
import os
import shutil

import torch
import torch.distributed as dist

__all__ = ["setup_dist", "PthCheckpoints", "lr_factor", "epoch_batches", "EpochBatches", "noise_seed", "gpu_peak", "fmt_gpu_peak"]


def setup_dist(device=None):
    """-> (rank, world, dev). torchrun đặt WORLD_SIZE/RANK/LOCAL_RANK; không có thì 1 tiến trình.
    (khuôn DDP của Diffusion Policy / CE-Loc gốc)

    `device="cpu"`: ép CPU kể cả khi có CUDA (nhiều tiến trình thì dùng gloo) — test chạy trên
    server có GPU dùng chung phải tất định và không phụ thuộc bộ nhớ GPU còn trống."""
    world = int(os.environ.get("WORLD_SIZE", "1"))
    use_cuda = torch.cuda.is_available() and device != "cpu"
    if world == 1:
        return 0, 1, torch.device(device or ("cuda" if use_cuda else "cpu"))
    local = int(os.environ["LOCAL_RANK"])
    if use_cuda:
        n = torch.cuda.device_count()
        if local >= n:
            raise SystemExit(f"LOCAL_RANK {local} nhưng chỉ thấy {n} GPU "
                             f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}): "
                             f"--nproc_per_node phải <= số GPU nhìn thấy")
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        dev = torch.device("cpu")
    # device_id (torch >= 2.3): NCCL gắn đúng GPU của tiến trình — barrier() khỏi đoán thiết bị (và khỏi cảnh báo)
    kw = ({"device_id": dev} if dev.type == "cuda" and "device_id" in inspect.signature(dist.init_process_group).parameters
          else {})
    dist.init_process_group("nccl" if dev.type == "cuda" else "gloo", timeout=datetime.timedelta(hours=1), **kw)
    return dist.get_rank(), world, dev


class PthCheckpoints:
    """`last.pth` / `best.pth` (quy tắc Kaggle), pickle KHÔNG zip, GHI NGUYÊN TỬ: ghi ra `*.tmp`, fsync, rồi `os.replace` sang tên thật —
    job bị giết đúng lúc đang ghi (ssh rớt, `kill`, OOM trên GPU dùng chung) thì file cũ vẫn nguyên. `last` = trạng thái mới nhất (đủ
    để `--resume`), `best` = bản chép của `last` khi chỉ số chọn cải thiện."""
    LAST, BEST = "last.pth", "best.pth"

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @property
    def last_path(self):
        return os.path.join(self.save_dir, self.LAST)

    @property
    def best_path(self):
        return os.path.join(self.save_dir, self.BEST)

    def has_last(self):
        return os.path.exists(self.last_path)

    def save(self, state, is_best):
        """Ghi `last.pth`; nếu `is_best` thì CHÉP sang `best.pth` (khỏi serialize ~700 MB hai lần; chép cũng qua `.tmp`)."""
        tmp = self.last_path + ".tmp"
        torch.save(state, tmp, _use_new_zipfile_serialization=False)
        with open(tmp, "rb") as f:          # đẩy xuống đĩa trước khi đổi tên
            os.fsync(f.fileno())
        os.replace(tmp, self.last_path)     # nguyên tử trên cùng một filesystem
        if is_best:
            tmp = self.best_path + ".tmp"
            shutil.copyfile(self.last_path, tmp)
            os.replace(tmp, self.best_path)

    def save_snapshot(self, state, name):
        """Ghi nguyên tử `<save_dir>/snapshots/<name>` (weight giữa chừng, không optimizer) -> đường dẫn."""
        path = os.path.join(self.save_dir, "snapshots", name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path + ".tmp", _use_new_zipfile_serialization=False)
        with open(path + ".tmp", "rb") as f:
            os.fsync(f.fileno())
        os.replace(path + ".tmp", path)
        return path

    def load_last(self, map_location="cpu"):
        return torch.load(self.last_path, map_location=map_location, weights_only=False)


def gpu_peak(dev, reset=True):
    """-> (đỉnh bộ nhớ tensor, đỉnh bộ nhớ allocator giữ) GB trên `dev` từ lần reset trước | None (CPU); `reset` bắt đầu cửa sổ mới.
    Chỉ đọc bộ đếm của allocator (không đồng bộ GPU)."""
    if dev.type != "cuda":
        return None
    out = torch.cuda.max_memory_allocated(dev) / 2 ** 30, torch.cuda.max_memory_reserved(dev) / 2 ** 30
    if reset:
        torch.cuda.reset_peak_memory_stats(dev)
    return out


def fmt_gpu_peak(peak):
    return "" if peak is None else f" | GPU đỉnh {peak[0]:.2f} GB (giữ {peak[1]:.2f})"


def lr_factor(it, tr):
    """Hệ số lr: cosine theo ITERATION tới `cosine_t_max` (CosineAnnealingLR của CE-Loc gốc), nhân warmup tuyến tính (it + 1) / W trong
    `cosine_warmup_iters` W iter đầu (vắng = không warmup — DELTA4 / 5.x)."""
    if tr.get("schedule", "cosine") != "cosine":
        raise ValueError(f"training.schedule {tr.get('schedule')!r}: chỉ còn cosine")
    f = 0.5 * (1 + math.cos(math.pi * min(it, tr["cosine_t_max"]) / tr["cosine_t_max"]))
    w = tr.get("cosine_warmup_iters") or 0
    return f * min(1.0, (it + 1) / w) if w else f


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


class EpochBatches:
    """batch_sampler cho DataLoader worker BỀN (`persistent_workers`): đặt (epoch, batch bắt đầu) trước mỗi `iter(loader)` — cùng thứ tự
    `epoch_batches`, khỏi dựng lại tiến trình worker mỗi epoch."""

    def __init__(self, n, batch_per_rank, rank, world, seed):
        self.args, self.epoch, self.start = (n, batch_per_rank, rank, world, seed), 0, 0

    def set(self, epoch, start=0):
        self.epoch, self.start = epoch, start

    def __iter__(self):
        return iter(epoch_batches(*self.args, self.epoch)[self.start:])

    def __len__(self):
        n, bpr, _, world, _ = self.args
        return n // (bpr * world) - self.start


def noise_seed(seed, it, rank):
    """Seed của t / nhiễu khuếch tán ở iteration `it` — tái lập khi resume."""
    return (seed * 1000003 + it * 101 + rank) % (2 ** 63 - 1)
