#!/usr/bin/env python3
"""Giữ trước G GiB GPU cho CẢ job rồi chạy script trong CÙNG tiến trình (người dùng 2026-11-08: server dùng chung — job đang chạy
mà bộ nhớ dùng tạm giảm thì người khác chen vào, tới đỉnh sau thì OOM).

    python gpu_hold.py G script.py [tham số...]                    # thường gọi qua run_on_free_gpu.py --reserve-gib G
    python gpu_hold.py --grab M G script.py [...]                  # GOM DẦN: chờ tới khi giữ đủ G, luôn chừa M GiB cho người khác
    python gpu_hold.py --grab M --release G script.py [...]        # gom đủ G rồi NHẢ hết trước khi chạy (job vLLM)
    python gpu_hold.py --keeper DIR --grab M [--need N] G           # NGƯỜI GIỮ CHỖ cho cả chuỗi (chạy nền), `DIR/stop` để thoát;
                                                                   # --need: ready khi đủ N, vẫn gom tiếp tới G (G lớn = lấy hết)
    python gpu_hold.py --grab M --handoff DIR G script.py [...]    # bước GPU trong chuỗi: lấy chỗ từ người giữ, xong trả lại

Cấp một khối G GiB rồi trả về CACHE của PyTorch (không trả driver): nvidia-smi vẫn thấy job giữ G GiB suốt thời gian chạy, mọi cấp
phát sau của job lấy từ khối đó. `torch.cuda.empty_cache` bị vô hiệu (nó trả cache về driver = nhả chỗ giữ). Job vượt G thì PyTorch
tự xin thêm như thường (lúc OOM nó mới nhả cache rồi thử lại). G = đỉnh đo được (log "GPU đỉnh") + một khoảng nhỏ (khối < 1 MiB
nằm ở pool riêng, ngoài khối giữ).

`--grab M` (người dùng 2026-11-08: GPU đang bị job khác chiếm, không đủ G ngay): mỗi `--poll` giây đọc bộ nhớ trống, xin thêm
(trống − M) GiB (tới đủ G), giữ phần đã có; đủ G thì chạy job. `expandable_segments` ⇒ các khối gom dần nằm trong MỘT segment ảo liền
⇒ nhả vào cache rồi dùng lại được cho tensor lớn bất kỳ. `--release`: gom đủ rồi nhả hết về driver và chạy script ở tiến trình CON
(vLLM đo bộ nhớ trống lúc khởi động và tự giữ `vllm.gpu_mem_gib`; khe hở giữa nhả và vLLM giữ chỉ vài giây).
`--keeper DIR` (người dùng 2026-11-08: giữ chỗ suốt CẢ chuỗi pilot, kể cả lúc merge / chấm chạy CPU): tiến trình riêng gom tới G,
ghi `DIR/ready` khi đủ (xoá lúc nhả — chuỗi chờ `ready` trước MỖI bước GPU); bước GPU (`--handoff DIR`) khởi tạo CUDA xong thì tạo `DIR/want` -> người giữ nhả, ghi `DIR/freed` -> bước gom
(khe hở ~1 s); bước thoát (xoá `want`) -> người giữ gom lại. Bước bị `kill -9` không xoá được `want` -> xoá tay.
`want` = "PID GiB": người giữ chỉ nhả phần bước cần (GiB), GIỮ LẠI phần dư suốt bước (người dùng 2026-11-09: bước vLLM chỉ cần
~20 / 32 GiB, nhả hết thì người khác chen vào phần dư); `want` chỉ có PID -> nhả hết.
KHÔNG dùng chế độ thường cho vLLM: nó sinh tiến trình con (spawn) nên cache của tiến trình này vô ích.
`PYTORCH_CUDA_ALLOC_CONF` phải đặt TRƯỚC lần chạm CUDA đầu tiên ⇒ đặt mặc định `expandable_segments:True` như train / eval / finetune.
"""

import os
import runpy
import subprocess
import sys
import time

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


def _gather(torch, gib, margin, poll, log, sleep, clock, heartbeat, stop=lambda: False, tag="gom GPU"):
    """Xin dần tới `gib` GiB (chừa `margin` GiB trống) cho tới đủ hoặc `stop()`. -> (khối đang giữ, GiB)."""
    blocks, held, t0 = [], 0.0, clock()
    last = t0
    while held < gib and not stop():
        free = torch.cuda.mem_get_info()[0] / GIB
        take = min(free - margin, gib - held)
        if take >= min(STEP, gib - held) and take > 0:
            try:
                blocks.append(torch.empty(int(take * GIB), dtype=torch.uint8, device="cuda"))
                held += take
                log(f"[{tag}] +{take:.1f} -> giữ {held:.1f} / {gib:g} GiB (trống còn ~{free - take:.1f}) | "
                    f"{(clock() - t0) / 60:.0f} phút", flush=True)
                last = clock()
            except torch.cuda.OutOfMemoryError:   # người khác vừa lấy giữa lúc đọc và lúc xin
                pass
        if held >= gib:
            break
        if clock() - last >= heartbeat:
            log(f"[{tag}] vẫn chờ: giữ {held:.1f} / {gib:g} GiB, trống {free:.1f} | {(clock() - t0) / 60:.0f} phút", flush=True)
            last = clock()
        sleep(poll)
    return blocks, held


def _wait(cond, poll, sleep):
    while not cond():
        sleep(poll)


def _take_over(handoff, poll, log, sleep, ask=None):
    """Bước GPU trong chuỗi có người giữ chỗ (`keeper`): xin người giữ nhả `ask` GiB (None = hết), chờ nhả xong; lúc thoát thì trả
    lại lượt."""
    import atexit
    want, freed, pid = (os.path.join(handoff, f) for f in ("want", "freed", "keeper.pid"))
    open(want, "w").write(f"{os.getpid()}" + ("" if ask is None else f" {ask:g}"))
    atexit.register(lambda: os.path.exists(want) and os.remove(want))
    alive = lambda: os.path.exists(pid) and _pid_alive(int(open(pid).read() or 0))
    log(f"[gom GPU] xin người giữ chỗ ({handoff}) nhả {'hết' if ask is None else f'{ask:g} GiB'}", flush=True)
    _wait(lambda: os.path.exists(freed) or not alive(), poll, sleep)
    log("[gom GPU] người giữ chỗ đã nhả" if os.path.exists(freed) else "[gom GPU] không có người giữ chỗ -> gom như thường",
        flush=True)


def _asked(path):
    """Nội dung `want` -> (PID, GiB bước cần | None = nhả hết)."""
    try:
        parts = open(path).read().split()
    except OSError:
        return "?", None
    return (parts[0] if parts else "?"), (float(parts[1]) if len(parts) > 1 else None)


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
        return pid > 0
    except OSError:
        return False


def grab(gib, margin, poll=10.0, release=False, log=print, sleep=time.sleep, clock=time.time, heartbeat=600.0, handoff=None):
    """Gom dần tới `gib` GiB, luôn chừa `margin` GiB trống. -> GiB đang giữ (0 nếu không CUDA / `release`).
    `handoff` = thư mục của `keeper`: CUDA khởi tạo xong mới xin người giữ chỗ nhả (khe hở ~1 lần đọc), rồi gom như thường."""
    torch = _torch_or_none(gib, log)
    if torch is None:
        return 0.0
    log(f"[gom GPU] cần {gib:g} GiB, chừa {margin:g} GiB cho người khác, đọc lại mỗi {poll:g} s", flush=True)
    if handoff:
        torch.cuda.mem_get_info()                  # khởi tạo CUDA trước (vài giây) rồi mới xin nhả
        # xin đủ gom `gib` (chừa `margin`) + khoảng cho CUDA context: vLLM (`release`) còn tiến trình chính + EngineCore, mỗi cái ~0,5
        _take_over(handoff, min(poll, 1.0), log, sleep, ask=gib + margin + (2.0 if release else 0.5))
        poll = min(poll, 1.0)
    t0 = clock()
    blocks, held = _gather(torch, gib, margin, poll, log, sleep, clock, heartbeat)
    del blocks
    log(f"[gom GPU] đủ {held:.1f} GiB sau {(clock() - t0) / 60:.0f} phút", flush=True)
    if release:
        torch.cuda.empty_cache()
        log("[gom GPU] nhả hết cho job (vLLM tự giữ)", flush=True)
        return 0.0
    return _keep(torch, log, "gom dần")


def keeper(d, gib, margin, poll=10.0, log=print, sleep=time.sleep, clock=time.time, heartbeat=600.0, reclaim_margin=1.0,
           need=None):
    """Người giữ chỗ cho CẢ chuỗi job: gom tới `gib` (chừa `margin`), ghi `d/ready` khi giữ được `need` (mặc định = `gib`) rồi vẫn
    gom tiếp tới `gib` (người dùng 2026-11-09: "còn bao nhiêu lấy hết, chừa 1 GB" ⇒ `gib` lớn, `need` = bước cần); bước GPU tạo
    `d/want` thì nhả hết, ghi `d/freed`, chờ bước đó xong (xoá `want`) rồi gom lại. `d/stop` -> thoát.

    Gom LẠI sau một bước GPU là lấy về chỗ của chính chuỗi (bước vừa thoát trả ra) ⇒ chỉ chừa `reclaim_margin` (như bước GPU) và đọc
    mỗi giây — KHÔNG chừa lại `margin`: người khác lấn vào phần trống trong lúc bước chạy thì chừa 8 GiB sẽ làm chuỗi kẹt hàng giờ
    dù vừa có đủ (pilot ZETA8 2026-11-09: sau bước 1 chỉ gom lại 31 / 37, chờ 98 phút)."""
    import torch
    os.makedirs(d, exist_ok=True)
    f = lambda n: os.path.join(d, n)
    for n in ("ready", "freed", "want", "stop"):
        if os.path.exists(f(n)):
            os.remove(f(n))
    open(f("keeper.pid"), "w").write(str(os.getpid()))
    real_empty = torch.cuda.empty_cache
    stop, want = (lambda: os.path.exists(f("stop"))), (lambda: os.path.exists(f("want")))
    need = gib if need is None else min(need, gib)
    log(f"[giữ chỗ] {d}: gom tới {gib:g} GiB (ready khi đủ {need:g}), chừa {margin:g} GiB", flush=True)
    m, p = margin, poll
    busy = lambda: stop() or want()
    kept, kheld = [], 0.0                          # phần giữ lại trong lúc bước GPU chạy
    while not stop():
        blocks, held = (_gather(torch, need - kheld, m, p, log, sleep, clock, heartbeat, stop=busy, tag="giữ chỗ")
                        if need > kheld else ([], 0.0))
        blocks, held, kept = kept + blocks, kheld + held, []
        if held >= need and not os.path.exists(f("ready")):
            open(f("ready"), "w").close()
            log(f"[giữ chỗ] đủ {held:.1f} GiB -> ready", flush=True)
        if held >= need and gib > held:            # phần thêm: gom tiếp, không chặn ready, dừng ngay khi bước GPU xin
            more, extra = _gather(torch, gib - held, m, p, log, sleep, clock, float("inf"), stop=busy, tag="giữ chỗ thêm")
            blocks += more
            held += extra
            del more                               # chỉ `blocks` giữ khối ⇒ `del blocks` nhả hết
        _wait(lambda: stop() or want(), min(poll, 1.0), sleep)
        pid, ask = _asked(f("want"))
        del blocks
        real_empty()
        if os.path.exists(f("ready")):
            os.remove(f("ready"))
        if stop():
            break
        kheld = 0.0 if ask is None else held - ask
        if kheld >= STEP:                          # giữ lại phần dư (cấp lại ngay sau khi nhả: khe hở vài µs)
            try:
                kept = [torch.empty(int(kheld * GIB), dtype=torch.uint8, device="cuda")]
            except torch.cuda.OutOfMemoryError:
                kept, kheld = [], 0.0
        else:
            kheld = 0.0
        open(f("freed"), "w").close()
        log(f"[giữ chỗ] nhả {held - kheld:.1f} GiB cho bước GPU (PID {pid}), giữ lại {kheld:.1f}", flush=True)
        _wait(lambda: stop() or not want(), min(poll, 1.0), sleep)
        os.remove(f("freed"))
        m, p = min(margin, reclaim_margin), min(poll, 1.0)
        log(f"[giữ chỗ] bước GPU xong -> gom lại (chừa {m:g} GiB, đọc mỗi {p:g} s)", flush=True)
    del kept
    for n in ("ready", "freed", "keeper.pid"):
        if os.path.exists(f(n)):
            os.remove(f(n))
    log("[giữ chỗ] thoát", flush=True)


def _run_here(script, args):
    sys.argv = [script] + args
    sys.path[0] = os.path.dirname(os.path.abspath(script))          # như `python script.py`
    runpy.run_path(script, run_name="__main__")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    margin, release, poll, handoff, keep, need = None, False, 10.0, None, None, None
    while argv and argv[0].startswith("--"):
        flag = argv.pop(0)
        if flag == "--grab":
            margin = float(argv.pop(0))
        elif flag == "--poll":
            poll = float(argv.pop(0))
        elif flag == "--release":
            release = True
        elif flag == "--handoff":
            handoff = argv.pop(0)
        elif flag == "--keeper":
            keep = argv.pop(0)
        elif flag == "--need":
            need = float(argv.pop(0))
        else:
            raise SystemExit(f"cờ lạ {flag}")
    if keep:
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return keeper(keep, float(argv[0]), 8.0 if margin is None else margin, poll=poll, need=need)
    if len(argv) < 2:
        raise SystemExit("cách dùng: gpu_hold.py [--grab M [--release] [--poll S]] G script.py [tham số...]")
    gib, script, args = float(argv[0]), argv[1], argv[2:]
    if release and margin is None:
        raise SystemExit("--release chỉ đi với --grab")
    if margin is None:
        hold(gib)
    else:
        grab(gib, margin, poll=poll, release=release, handoff=handoff)
    if release:                                   # tiến trình con: vLLM spawn, không thừa hưởng gì từ tiến trình này
        raise SystemExit(subprocess.run([sys.executable, script] + args).returncode)
    _run_here(script, args)


if __name__ == "__main__":
    main()
