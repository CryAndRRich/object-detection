#!/usr/bin/env python3
"""Giữ trước G GiB GPU cho CẢ job rồi chạy script trong CÙNG tiến trình (người dùng 2026-11-08: server dùng chung — job đang chạy
mà bộ nhớ dùng tạm giảm thì người khác chen vào, tới đỉnh sau thì OOM).

    python gpu_hold.py G script.py [tham số...]        # thường gọi qua run_on_free_gpu.py --reserve-gib G

Cấp một khối G GiB rồi trả về CACHE của PyTorch (không trả driver): nvidia-smi vẫn thấy job giữ G GiB suốt thời gian chạy, mọi cấp
phát sau của job lấy từ khối đó. `torch.cuda.empty_cache` bị vô hiệu (nó trả cache về driver = nhả chỗ giữ). Job vượt G thì PyTorch
tự xin thêm như thường (lúc OOM nó mới nhả cache rồi thử lại). G = đỉnh đo được (log "GPU đỉnh") + một khoảng nhỏ (khối < 1 MiB
nằm ở pool riêng, ngoài khối giữ).
KHÔNG dùng cho vLLM: vLLM tự giữ trước `vllm.gpu_mem_gib`, và nó sinh tiến trình con (spawn) nên cache của tiến trình này vô ích.
`PYTORCH_CUDA_ALLOC_CONF` phải đặt TRƯỚC lần chạm CUDA đầu tiên ⇒ đặt mặc định `expandable_segments:True` như train / eval / finetune.
"""

import os
import runpy
import sys

__all__ = ["hold", "main"]


def _keep_cache():
    """Thay `torch.cuda.empty_cache`: không trả cache về driver (giữ chỗ)."""


def hold(gib, log=print):
    """Giữ `gib` GiB trên GPU hiện tại (cache PyTorch). -> số GiB đang giữ (0 nếu không giữ)."""
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gib <= 0:
        return 0.0
    import torch
    if not torch.cuda.is_available():
        log(f"[giữ GPU] không có CUDA -> bỏ qua giữ {gib:g} GiB", flush=True)
        return 0.0
    try:
        x = torch.empty(int(gib * 2 ** 30), dtype=torch.uint8, device="cuda")
    except torch.cuda.OutOfMemoryError:       # torch cũ (server 1) không có torch.OutOfMemoryError
        free = torch.cuda.mem_get_info()[0] / 2 ** 30
        log(f"[giữ GPU] KHÔNG đủ chỗ giữ {gib:g} GiB (trống {free:.1f} GiB) -> thoát 1 (run_on_free_gpu thử lại nếu còn lượt)",
            flush=True)
        raise SystemExit(1)
    del x
    torch.cuda.empty_cache = _keep_cache
    got = torch.cuda.memory_reserved() / 2 ** 30
    log(f"[giữ GPU] giữ {got:.1f} GiB cho cả job (empty_cache vô hiệu)", flush=True)
    return got


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 2:
        raise SystemExit("cách dùng: gpu_hold.py G script.py [tham số...]")
    hold(float(argv[0]))
    script = argv[1]
    sys.argv = [script] + argv[2:]
    sys.path[0] = os.path.dirname(os.path.abspath(script))          # như `python script.py`
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
