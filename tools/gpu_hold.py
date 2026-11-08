#!/usr/bin/env python3
"""Giữ trước G GiB GPU cho CẢ job rồi chạy script trong CÙNG tiến trình (người dùng 2026-11-08: server dùng chung — job đang chạy
mà bộ nhớ dùng tạm giảm thì người khác chen vào, tới đỉnh sau thì OOM).

    python gpu_hold.py G script.py [tham số...]                    # thường gọi qua run_on_free_gpu.py --reserve-gib G
    python gpu_hold.py --grab M G script.py [...]                  # GOM DẦN: chờ tới khi giữ đủ G, luôn chừa M GiB cho người khác
    python gpu_hold.py --grab M --release G script.py [...]        # gom đủ G rồi NHẢ hết trước khi chạy (job vLLM)

Cấp một khối G GiB rồi trả về CACHE của PyTorch (không trả driver): nvidia-smi vẫn thấy job giữ G GiB suốt thời gian chạy, mọi cấp
phát sau của job lấy từ khối đó. `torch.cuda.empty_cache` bị vô hiệu (nó trả cache về driver = nhả chỗ giữ). Job vượt G thì PyTorch
tự xin thêm như thường (lúc OOM nó mới nhả cache rồi thử lại). G = đỉnh đo được (log "GPU đỉnh") + một khoảng nhỏ (khối < 1 MiB
nằm ở pool riêng, ngoài khối giữ).

`--grab M` (người dùng 2026-11-08: GPU đang bị job khác chiếm, không đủ G ngay): mỗi `--poll` giây đọc bộ nhớ trống, xin thêm
(trống − M) GiB (tới đủ G), giữ phần đã có; đủ G thì chạy job. `expandable_segments` ⇒ các khối gom dần nằm trong MỘT segment ảo liền
⇒ nhả vào cache rồi dùng lại được cho tensor lớn bất kỳ. `--release`: gom đủ rồi nhả hết về driver và chạy script ở tiến trình CON
(vLLM đo bộ nhớ trống lúc khởi động và tự giữ `vllm.gpu_mem_gib`; khe hở giữa nhả và vLLM giữ chỉ vài giây).
KHÔNG dùng chế độ thường cho vLLM: nó sinh tiến trình con (spawn) nên cache của tiến trình này vô ích.
`PYTORCH_CUDA_ALLOC_CONF` phải đặt TRƯỚC lần chạm CUDA đầu tiên ⇒ đặt mặc định `expandable_segments:True` như train / eval / finetune.
"""

import os
import runpy
import subprocess
import sys
import time

__all__ = ["hold", "grab", "main"]

GIB = 2 ** 30
STEP = 0.25                 # GiB: không xin mẩu nhỏ hơn


def _keep_cache():
    """Thay `torch.cuda.empty_cache`: không trả cache về driver (giữ chỗ)."""


def _torch_or_none(gib, log):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if gib <= 0:
        return None
    import torch
    if not torch.cuda.is_available():
        log(f"[giữ GPU] không có CUDA -> bỏ qua giữ {gib:g} GiB", flush=True)
        return None
    return torch


def _keep(torch, log, how):
    torch.cuda.empty_cache = _keep_cache
    got = torch.cuda.memory_reserved() / GIB
    log(f"[giữ GPU] giữ {got:.1f} GiB cho cả job ({how}; empty_cache vô hiệu)", flush=True)
    return got


def hold(gib, log=print):
    """Giữ `gib` GiB trên GPU hiện tại (cache PyTorch). -> số GiB đang giữ (0 nếu không giữ)."""
    torch = _torch_or_none(gib, log)
    if torch is None:
        return 0.0
    try:
        x = torch.empty(int(gib * GIB), dtype=torch.uint8, device="cuda")
    except torch.cuda.OutOfMemoryError:       # torch cũ (server 1) không có torch.OutOfMemoryError
        free = torch.cuda.mem_get_info()[0] / GIB
        log(f"[giữ GPU] KHÔNG đủ chỗ giữ {gib:g} GiB (trống {free:.1f} GiB) -> thoát 1 (run_on_free_gpu thử lại nếu còn lượt)",
            flush=True)
        raise SystemExit(1)
    del x
    return _keep(torch, log, "một lần")


def grab(gib, margin, poll=10.0, release=False, log=print, sleep=time.sleep, clock=time.time, heartbeat=600.0):
    """Gom dần tới `gib` GiB, luôn chừa `margin` GiB trống. -> GiB đang giữ (0 nếu không CUDA / `release`)."""
    torch = _torch_or_none(gib, log)
    if torch is None:
        return 0.0
    blocks, held, t0 = [], 0.0, clock()
    last = t0
    log(f"[gom GPU] cần {gib:g} GiB, chừa {margin:g} GiB cho người khác, đọc lại mỗi {poll:g} s", flush=True)
    while held < gib:
        free = torch.cuda.mem_get_info()[0] / GIB
        take = min(free - margin, gib - held)
        if take >= min(STEP, gib - held) and take > 0:
            try:
                blocks.append(torch.empty(int(take * GIB), dtype=torch.uint8, device="cuda"))
                held += take
                log(f"[gom GPU] +{take:.1f} -> giữ {held:.1f} / {gib:g} GiB (trống còn ~{free - take:.1f}) | "
                    f"{(clock() - t0) / 60:.0f} phút", flush=True)
                last = clock()
            except torch.cuda.OutOfMemoryError:   # người khác vừa lấy giữa lúc đọc và lúc xin
                pass
        if held >= gib:
            break
        if clock() - last >= heartbeat:
            log(f"[gom GPU] vẫn chờ: giữ {held:.1f} / {gib:g} GiB, trống {free:.1f} | {(clock() - t0) / 60:.0f} phút", flush=True)
            last = clock()
        sleep(poll)
    del blocks
    log(f"[gom GPU] đủ {held:.1f} GiB sau {(clock() - t0) / 60:.0f} phút", flush=True)
    if release:
        torch.cuda.empty_cache()
        log("[gom GPU] nhả hết cho job (vLLM tự giữ)", flush=True)
        return 0.0
    return _keep(torch, log, "gom dần")


def _run_here(script, args):
    sys.argv = [script] + args
    sys.path[0] = os.path.dirname(os.path.abspath(script))          # như `python script.py`
    runpy.run_path(script, run_name="__main__")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    margin, release, poll = None, False, 10.0
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--grab":
            margin = float(argv.pop(0))
        elif flag == "--poll":
            poll = float(argv.pop(0))
        elif flag == "--release":
            release = True
        else:
            raise SystemExit(f"cờ lạ {flag}")
    if len(argv) < 2:
        raise SystemExit("cách dùng: gpu_hold.py [--grab M [--release] [--poll S]] G script.py [tham số...]")
    gib, script, args = float(argv[0]), argv[1], argv[2:]
    if release and margin is None:
        raise SystemExit("--release chỉ đi với --grab")
    if margin is None:
        hold(gib)
    else:
        grab(gib, margin, poll=poll, release=release)
    if release:                                   # tiến trình con: vLLM spawn, không thừa hưởng gì từ tiến trình này
        raise SystemExit(subprocess.run([sys.executable, script] + args).returncode)
    _run_here(script, args)


if __name__ == "__main__":
    main()
