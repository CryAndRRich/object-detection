#!/usr/bin/env python3
"""Chạy MỘT script Python trên GPU có NHIỀU BỘ NHỚ TRỐNG NHẤT của server (dùng chung).

Dùng chung cho mọi sub-project. Chạy từ thư mục của project, script chạy trong CHÍNH thư
mục hiện tại đó:

    cd object-detection/ce_localization
    python ../tools/run_on_free_gpu.py -- train.py --save-dir checkpoints/x
    python ../tools/run_on_free_gpu.py --gpu 1 -- eval.py --ckpt ...      # ép GPU

Ba quy tắc, mỗi quy tắc từng gây sự cố thật:
  1. Chọn theo BỘ NHỚ TRỐNG, và CHỈ theo nó. Lọc theo utilization trước (2026-09-11) đã
     gửi job sang GPU còn 1.238 MiB trong khi GPU khác trống 15.842 MiB. Utilization chỉ
     được IN ra để đọc log.
  2. KHÔNG có ngưỡng bộ nhớ tối thiểu. Hai lần thử (ngưỡng cứng; ngưỡng đoán từ tên
     script) đều làm job bị bỏ qua hoặc chờ vô ích hàng giờ. OOM thì torch báo rõ và
     `--retries` thử lại, mỗi lần đọc lại nvidia-smi.
  3. Job bị `kill` (rc < 0) là người dùng CỐ Ý dừng -> KHÔNG retry (2026-09-04: wrapper
     từng hồi sinh chính job vừa bị kill).
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def query_gpus():
    """-> [(index, free_mib, total_mib, util_percent)], [] nếu không có nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return []
    if out.returncode != 0:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        idx, used, total, util = (int(x.strip()) for x in line.split(","))
        gpus.append((idx, total - used, total, util))
    return gpus


def pick_gpu(gpus):
    """GPU có NHIỀU BỘ NHỚ TRỐNG NHẤT. Một tiêu chí duy nhất (quy tắc 1)."""
    return max(gpus, key=lambda g: g[1])


def run_once(cmd, forced_gpu=None):
    """Chọn GPU rồi chạy `cmd` trong thư mục hiện tại -> return code."""
    env = os.environ.copy()
    if forced_gpu is not None:
        chosen = forced_gpu
        print(f"dùng GPU {chosen} (ép bằng --gpu)", flush=True)
    else:
        gpus = query_gpus()
        if not gpus:
            print("không có nvidia-smi -> chạy thẳng, không đặt CUDA_VISIBLE_DEVICES",
                  flush=True)
            print(f"running: {' '.join(cmd)}", flush=True)
            return subprocess.run(cmd).returncode
        print("current GPU state:", flush=True)
        for idx, free, total, util in gpus:
            print(f"  GPU {idx}: free {free:6d} MiB / {total} MiB, utilization {util:3d}%")
        chosen, free, _, util = pick_gpu(gpus)
        print(f"chose GPU {chosen} (free {free} MiB, utilization {util}%) -- most free "
              f"memory", flush=True)
    env["CUDA_VISIBLE_DEVICES"] = str(chosen)
    print(f"running: CUDA_VISIBLE_DEVICES={chosen} {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu", type=int, default=None, help="ép một GPU, bỏ tự chọn")
    ap.add_argument("--retries", type=int, default=3,
                    help="số lần thử lại khi job chết (GPU có thể bị chiếm giữa lúc đọc "
                         "nvidia-smi và lúc cấp phát)")
    ap.add_argument("--retry-wait", type=int, default=60, help="giây chờ trước mỗi lần thử")
    ap.add_argument("target", nargs=argparse.REMAINDER,
                    help="script + tham số của nó, đặt sau --")
    a = ap.parse_args()

    target = a.target[1:] if a.target and a.target[0] == "--" else a.target
    if not target:
        ap.error("chưa có script — đặt sau --, vd: -- train.py --save-dir checkpoints/x")
    cmd = [sys.executable] + target

    for attempt in range(1, a.retries + 2):
        rc = run_once(cmd, a.gpu)
        if rc == 0:
            raise SystemExit(0)
        if rc < 0:                                          # quy tắc 3
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = f"signal {-rc}"
            print(f"\nJob stopped by {name} (rc={rc}) -- CỐ Ý dừng, KHÔNG retry.", flush=True)
            raise SystemExit(128 + (-rc))
        if attempt > a.retries:
            print(f"EXHAUSTED {a.retries} retries -- bỏ cuộc. Traceback là CUDA OOM thì GPU "
                  f"đang bị chiếm (chờ, --gpu <id>, hoặc hạ batch); lỗi khác thì phải sửa "
                  f"code, retry không giúp.", flush=True)
            raise SystemExit(rc)
        print(f"\n[retry {attempt}/{a.retries}] job thoát với code {rc}. Thường là OOM trên "
              f"server dùng chung, nhưng mã thoát không phân biệt được với lỗi code — nếu "
              f"cả {a.retries} lần đều hỏng thì đọc traceback. Chờ {a.retry_wait}s...\n",
              flush=True)
        time.sleep(a.retry_wait)


if __name__ == "__main__":
    main()
