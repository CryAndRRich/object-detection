"""Chép tensor CPU lên thiết bị không bắt CPU chờ GPU."""

import torch

__all__ = ["to_device"]


def to_device(t, dev):
    """Tensor CPU -> `dev` KHÔNG đồng bộ: CUDA chép từ bộ nhớ ghim (chép từ bộ nhớ thường bắt CPU chờ GPU xong mọi việc đang xếp hàng;
    khối ghim được bộ cấp phát host của PyTorch giữ tới khi lệnh chép xong)."""
    dev = torch.device(dev)
    return t.pin_memory().to(dev, non_blocking=True) if dev.type == "cuda" else t.to(dev)
