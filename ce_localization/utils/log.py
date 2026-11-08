"""Tiện ích log: thời gian thực tế, thông tin môi trường chạy.

Mọi job chạy lâu phải in tiến độ kèm thời gian THỰC TẾ và ETA — im lặng vài phút không
phân biệt được với treo.
"""

import os
import socket
import sys
import time
from datetime import datetime

import torch

__all__ = ["fmt_time", "run_env", "print_banner"]


def fmt_time(seconds):
    """3661 -> '1h01m01s'."""
    seconds = int(max(seconds, 0))
    h, m, s = seconds // 3600, (seconds % 3600) // 60, seconds % 60
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def run_env(cfg, dev):
    """Nơi chạy + GPU + phiên bản + lệnh — mọi kết quả train phải ghi kèm những thứ này."""
    return {
        "experiment": cfg.get("experiment", "?"),
        "description": cfg.get("description", ""),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": str(dev), "torch": torch.__version__,
        "python": sys.version.split()[0], "hostname": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "command": " ".join(sys.argv), "cwd": os.getcwd(),
    }


def print_banner(title, env):
    print("=" * 78, flush=True)
    print(f"  {title}", flush=True)
    print("-" * 78, flush=True)
    for k, v in env.items():
        print(f"  {k:22s} {v}", flush=True)
    print("=" * 78, flush=True)


class Progress:
    """Log tiến độ suy luận dùng chung (baseline ZETA): `update(done, prev)` in "[name done/total] x ms/ảnh | còn ~ETA" khi số ảnh xong
    vượt một bội số của `every` (hoặc xong hết). `every` 0 / None = im lặng."""

    def __init__(self, name, total, every=500, log=print):
        self.name, self.total, self.every, self.log, self.t0 = name, total, every, log, time.time()

    def update(self, done, prev=None):
        prev = done - 1 if prev is None else prev
        if self.every and done and (done // self.every != prev // self.every or done == self.total):
            el = time.time() - self.t0
            self.log(f"    [{self.name} {done:5d}/{self.total}] {el / done * 1000:.0f} ms/ảnh | còn ~{fmt_time(el / done * (self.total - done))}")
