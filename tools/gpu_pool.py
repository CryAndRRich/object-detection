#!/usr/bin/env python3
"""Bể GPU nhiều tầng cho nhiều làn chạy song song trên MỘT GPU dùng chung (người dùng 2026-11-16, ablation DELTA9 server 2).

    python gpu_pool.py root DIR TOTAL [--margin M] [--poll S]   # GỐC: gom dần tới TOTAL GiB (chừa M), cấp cho các làn
    python gpu_pool.py lane DIR NEED [--poll S]                 # LÀN: DIR = <thư mục gốc>/<tên làn>; xin NEED GiB từ gốc
    bước GPU của làn: run_on_free_gpu.py --gpu X --grab-margin-gib 1 --handoff DIR --reserve-gib g -- script ...

Cây: gốc giữ TOTAL = tổng (đỉnh lớn nhất mỗi làn) + 1–2 GiB dư; làn xin gốc đúng NEED của mình rồi giữ suốt làn; mỗi bước của
làn xin làn (giao thức `gpu_hold.py --keeper`: `DIR/want` "PID GiB" -> `DIR/freed`, bước thoát xoá `want`). Bước xin nhiều hơn
làn đang giữ -> làn xin THÊM phần thiếu từ gốc (`want.<làn>-top`), bước xong thì trả lại gốc.

Giao thức với gốc (thư mục gốc): khách ghi `want.<id>` = "PID GiB"; gốc phục vụ TỪNG yêu cầu một theo thứ tự tạo file: đủ chỗ thì
nhả đúng GiB đó về driver, ghi `freed.<id>`, chờ khách gom xong (`got.<id>`) rồi mới xét yêu cầu sau ("1 xong thì 2 mới xin");
chưa đủ thì gom tiếp. Khách thoát (xoá `want.<id>` / PID chết) -> gốc gom lại phần đó.
Gốc KHÔNG gom trong lúc khách đang gom phần vừa nhả, và trong `HOLD_OFF` s sau khi một làn bàn giao / nhận lại chỗ từ bước
(`hold_off`) — tránh lấy mất chỗ của chính chuỗi (server 2 2026-11-16: keeper làn bên lấy chỗ bước 9.12 vừa nhả ⇒ OOM).
Giữ bằng các khúc `CHUNK` GiB (tensor sống) ⇒ nhả đúng phần cần. `DIR/stop` -> nhả hết, thoát (làn: trả cả gốc).
Bước GPU nên chạy `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`: khối giữ là MỘT segment cudaMalloc, model nằm trong đó ⇒
`emptyCache` của `cudnn.benchmark` (C++) không trả được nó về driver (bật expandable thì trả được phần trống ⇒ mất chỗ).
"""

import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_hold import GIB, _asked, _pid_alive  # noqa: E402

CHUNK = 0.25                # GiB / khúc giữ
HOLD_OFF = 3.0              # s: gốc / làn không gom sau khi một làn bàn giao / nhận lại chỗ từ bước
ROOT_WAIT = 120.0           # s: khách chờ gốc khởi động (pool.pid) trước khi tự gom
REGATHER_WAIT = 120.0       # s: làn gom lại chưa đủ sau ngần này -> vẫn ready


class Holder:
    """Giữ bộ nhớ GPU thành các khúc `CHUNK` GiB (tensor sống); nhả đúng phần cần về driver."""

    def __init__(self, torch):
        self.t, self.chunks = torch, []

    @property
    def gib(self):
        return len(self.chunks) * CHUNK

    def take(self, gib, margin):
        """Xin thêm tối đa `gib` từ driver, chừa `margin` GiB trống -> GiB vừa xin."""
        n = int(min(gib, self.t.cuda.mem_get_info()[0] / GIB - margin) / CHUNK + 1e-9)
        got = 0
        for _ in range(max(n, 0)):
            try:
                self.chunks.append(self.t.empty(int(CHUNK * GIB), dtype=self.t.uint8, device="cuda"))
            except self.t.cuda.OutOfMemoryError:        # người khác vừa lấy
                break
            got += 1
        return got * CHUNK

    def give(self, gib):
        """Nhả `gib` (làm tròn lên khúc) về driver -> GiB đã nhả."""
        n = min(len(self.chunks), int(-(-round(gib / CHUNK, 6) // 1)))
        if n:
            del self.chunks[len(self.chunks) - n:]
            self.t.cuda.empty_cache()
        return n * CHUNK


def _rm(*paths):
    for p in paths:
        if os.path.exists(p):
            os.remove(p)


def _touch(path):
    with open(path, "w"):
        pass


def _fresh(path, clock, sec):
    try:
        return clock() - os.path.getmtime(path) < sec
    except OSError:
        return False


def root(d, total, margin=1.0, poll=1.0, log=print, sleep=time.sleep, clock=time.time, heartbeat=600.0, torch=None):
    """GỐC của cây (xem docstring module)."""
    if torch is None:
        import torch
    os.makedirs(d, exist_ok=True)
    f = lambda n: os.path.join(d, n)
    for p in glob.glob(f("want.*")) + glob.glob(f("freed.*")) + glob.glob(f("got.*")) + [f("stop"), f("hold_off")]:
        _rm(p)
    with open(f("pool.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    h, granted, serving, t_log = Holder(torch), {}, None, clock()
    log(f"[bể] {d}: gom tới {total:g} GiB, chừa {margin:g} GiB", flush=True)
    while not os.path.exists(f("stop")):
        for cid, (pid, gib) in list(granted.items()):                     # khách đã thoát -> gom lại phần của họ
            if not os.path.exists(f(f"want.{cid}")) or not _pid_alive(pid):
                granted.pop(cid)
                _rm(f(f"want.{cid}"), f(f"freed.{cid}"), f(f"got.{cid}"))
                log(f"[bể] {cid} trả {gib:g} GiB -> gom lại", flush=True)
        if serving:
            cid, t0 = serving
            if os.path.exists(f(f"got.{cid}")) or cid not in granted or clock() - t0 > 300:
                serving = None
                log(f"[bể] {cid} đã nhận | bể giữ {h.gib:.2f} GiB, đã cấp "
                    f"{sum(g for _, g in granted.values()):g}", flush=True)
        if not serving:                                                  # yêu cầu kế (theo thứ tự tạo file)
            reqs = sorted((os.path.getmtime(p), p.split("want.", 1)[1]) for p in glob.glob(f("want.*"))
                          if p.split("want.", 1)[1] not in granted)
            if reqs:
                cid = reqs[0][1]
                pid, gib = _asked(f(f"want.{cid}"))
                pid = int(pid) if str(pid).isdigit() else 0
                gib = h.gib if gib is None else gib
                if not _pid_alive(pid):
                    _rm(f(f"want.{cid}"))
                elif h.gib >= gib - 1e-6:
                    h.give(gib)
                    granted[cid] = (pid, gib)
                    serving = (cid, clock())
                    _touch(f(f"freed.{cid}"))
                    log(f"[bể] cấp {gib:g} GiB cho {cid} (PID {pid}) | bể còn {h.gib:.2f}", flush=True)
        target = total - sum(g for _, g in granted.values())
        if not serving and h.gib < target - CHUNK / 2 and not _fresh(f("hold_off"), clock, HOLD_OFF):
            got = h.take(target - h.gib, margin)
            if got:
                log(f"[bể] +{got:g} -> giữ {h.gib:.2f} / {target:g} GiB (chưa cấp)", flush=True)
                t_log = clock()
        if clock() - t_log >= heartbeat:
            log(f"[bể] giữ {h.gib:.2f} / {target:g} GiB chưa cấp | đã cấp {dict((c, g) for c, (_, g) in granted.items())} | "
                f"chờ: {[p.split('want.', 1)[1] for p in glob.glob(f('want.*')) if p.split('want.', 1)[1] not in granted]}",
                flush=True)
            t_log = clock()
        sleep(poll)
    h.give(h.gib)
    _rm(f("pool.pid"))
    log("[bể] dừng, nhả hết", flush=True)


def _ask_parent(pd, cid, gib, h, stop, log, sleep, poll):
    """Xin `gib` từ gốc `pd` dưới tên `cid`, gom phần gốc vừa nhả -> True (False: dừng giữa chừng)."""
    def alive():
        try:
            return _pid_alive(int(open(os.path.join(pd, "pool.pid")).read() or 0))
        except (OSError, ValueError):
            return False
    for _ in range(int(ROOT_WAIT / poll)):                               # gốc vừa khởi động chưa ghi pool.pid: CHỜ, đừng tự gom
        if alive() or stop():                                           # (server 2 2026-11-16: làn 1 tự gom -> gốc không tính
            break                                                       # phần làn 1, giữ thừa 10,5 GiB -> các làn kẹt cả đêm)
        sleep(poll)
    tmp = os.path.join(pd, f".tmp_want_{cid}")
    with open(tmp, "w") as fh:
        fh.write(f"{os.getpid()} {gib:g}")
    os.replace(tmp, os.path.join(pd, f"want.{cid}"))                    # ghi nguyên tử: gốc không đọc nửa file
    log(f"[làn] xin gốc {gib:g} GiB ({cid})", flush=True)
    while not os.path.exists(os.path.join(pd, f"freed.{cid}")):
        if stop():
            return False
        if not alive():
            log("[làn] gốc không chạy -> tự gom", flush=True)
            break
        sleep(poll)
    want = h.gib + gib
    while h.gib < want - CHUNK / 2:
        if stop():
            return False
        if not h.take(want - h.gib, 0.3):
            sleep(0.2)
    _touch(os.path.join(pd, f"got.{cid}"))
    log(f"[làn] nhận {gib:g} GiB từ gốc -> giữ {h.gib:.2f}", flush=True)
    return True


def lane(d, need, poll=1.0, log=print, sleep=time.sleep, clock=time.time, torch=None):
    """LÀN: xin `need` GiB từ gốc (thư mục cha của `d`), cấp cho từng bước theo giao thức `gpu_hold.py --keeper`."""
    if torch is None:
        import torch
    d = os.path.abspath(d)
    pd, me = os.path.dirname(d), os.path.basename(d)
    os.makedirs(d, exist_ok=True)
    f = lambda n: os.path.join(d, n)
    for n in ("want", "freed", "ready", "stop"):
        _rm(f(n))
    with open(f("keeper.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    stop = lambda: os.path.exists(f("stop"))
    h, top = Holder(torch), 0.0
    if _ask_parent(pd, me, need, h, stop, log, sleep, poll):
        _touch(f("ready"))
        log(f"[làn {me}] ready, giữ {h.gib:.2f} GiB", flush=True)
    while not stop():
        if not os.path.exists(f("want")):
            if h.gib < need - CHUNK / 2 and not _fresh(os.path.join(pd, "hold_off"), clock, HOLD_OFF):
                h.take(need - h.gib, 0.3)                               # đã ready khi còn thiếu -> gom tiếp lúc rảnh
            sleep(poll)
            continue
        pid, ask = _asked(f("want"))
        ask = h.gib if ask is None else ask
        if ask > h.gib + 1e-6:                                           # bước cần hơn làn đang giữ -> xin thêm từ gốc
            top = CHUNK * -(-(ask - h.gib) // CHUNK)
            if not _ask_parent(pd, f"{me}-top", top, h, stop, log, sleep, poll):
                break
        _rm(f("ready"))
        _touch(os.path.join(pd, "hold_off"))
        gave = h.give(ask)
        _touch(f("freed"))
        log(f"[làn {me}] nhả {gave:g} GiB cho bước (PID {pid}), giữ lại {h.gib:.2f}", flush=True)
        while os.path.exists(f("want")) and not stop():
            sleep(poll)
        _rm(f("freed"))
        _touch(os.path.join(pd, "hold_off"))
        t0 = clock()
        while h.gib < need - CHUNK / 2 and not stop():                   # gom lại phần của làn (bước vừa trả)
            if _fresh(os.path.join(pd, "hold_off"), clock, HOLD_OFF) or not h.take(need - h.gib, 0.3):
                if clock() - t0 > REGATHER_WAIT:                         # không chờ đủ mãi (server 2 2026-11-16: 3 làn cùng
                    log(f"[làn {me}] gom lại mới {h.gib:.2f} / {need:g} GiB sau {REGATHER_WAIT:g} s -> vẫn ready "
                        "(bước tự gom phần thiếu; làn gom tiếp lúc rảnh)", flush=True)  # chờ đủ -> kẹt chéo 8 giờ)
                    break
                sleep(0.2)
        if top:                                                          # trả phần xin thêm cho gốc
            _rm(os.path.join(pd, f"want.{me}-top"))
            top = 0.0
        _touch(f("ready"))
        log(f"[làn {me}] bước xong -> giữ lại {h.gib:.2f} GiB, ready", flush=True)
    h.give(h.gib)
    _rm(os.path.join(pd, f"want.{me}"), os.path.join(pd, f"want.{me}-top"), f("ready"), f("keeper.pid"))
    log(f"[làn {me}] dừng, trả gốc", flush=True)


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if len(a) < 3 or a[0] not in ("root", "lane"):
        raise SystemExit(__doc__)
    role, d, gib, rest = a[0], a[1], float(a[2]), a[3:]
    opt = {k.lstrip("-"): float(v) for k, v in zip(rest[::2], rest[1::2])}
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:False")
    log = lambda *x, **k: print(time.strftime("%m-%d %H:%M:%S"), *x, **k)
    if role == "root":
        root(d, gib, margin=opt.get("margin", 1.0), poll=opt.get("poll", 1.0), log=log)
    else:
        lane(d, gib, poll=opt.get("poll", 1.0), log=log)


if __name__ == "__main__":
    main()
