"""Run any diffu2seg script on the server's freest GPU, chosen via nvidia-smi.

PORTED (rewritten, not imported) from
count_editing/CE-LocModel/tools/run_on_free_gpu.py -- sub-projects under
object-detection/ do not import from each other. The three hard-won lessons in
that file are reproduced here because they cost real debugging time and would be
re-learned identically:

  1. Query nvidia-smi for every GPU: free memory + utilization.
  2. Pick the GPU with the MOST FREE MEMORY. That is the ONLY criterion.
  3. Set CUDA_VISIBLE_DEVICES=<chosen> and run the given script.

The server has 3x A30 24 GB shared with other people's jobs, so defaulting to
GPU 0 is not viable.

                    NO MINIMUM FREE-MEMORY THRESHOLD

Tried twice upstream, failed twice:
  - A hard-coded 15000 MiB floor skipped jobs within one second even when a GPU
    had 10,607 MiB free -- plenty for them.
  - Inferring the floor from the script name misclassified a visualisation tool
    as a training job, and it waited 2 hours for a GPU it never needed.

Guessing memory from a command line is wrong somewhere, and when it is wrong the
job never runs -- far worse than an OOM, which torch reports clearly and which
costs seconds. Pick the freest GPU and run.

                  UTILIZATION IS PRINTED, NEVER USED TO CHOOSE

An upstream version filtered on utilization first (keep GPUs below 50 % busy,
then take the most free memory). On 2026-09-11 GPU 0 had 1,238 MiB free at 0 %
while GPU 2 had 15,842 MiB free at 72 %. The filter shrank the pool to GPU 0, so
the job went to the gpu with the LEAST free memory -- the exact opposite of
"freest". Utilization is a snapshot of the last sampling period, not a claim on
memory; memory is what jobs run out of.

                        DIFFU2SEG-SPECIFIC NOTE

VRAM here is modest and does not scale with the sweep: the affinity A is
(4096, 4096) fp32 = 0.07 GB at grid_r=64, and f is (441, 4096) = 7 MB. SD2 in
fp16 dominates. A GPU with a few GB free is enough -- one more reason not to
gate on a threshold.

Usage -- the script goes immediately after --, its arguments follow it:

    python tools/run_on_free_gpu.py -- tools/check_attention_separates.py --split val
    python tools/run_on_free_gpu.py -- tools/check_plaplacian_vs_p2.py --limit 100
    python tools/run_on_free_gpu.py -- tools/run_stage1.py --split val

To force a specific GPU: --gpu <index> (placed BEFORE --).

⚠️ HF_HOME must still be exported by the caller; this wrapper does not set it.
"""

import argparse
import os
import subprocess
import sys
import time


def query_gpus():
    """``[(index, free_mib, total_mib, util_percent), ...]`` from nvidia-smi.

    Returns ``[]`` when nvidia-smi is absent (a dev machine with no GPU) -- the
    caller then runs directly without setting CUDA_VISIBLE_DEVICES.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=False,
        )
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
    """The GPU with the MOST FREE MEMORY. One criterion -- see module docstring."""
    return max(gpus, key=lambda g: g[1])


def _pick_and_run(args, cmd, repo_root):
    if args.gpu is not None:
        chosen = args.gpu
        print(f"using GPU {chosen} (forced via --gpu)", flush=True)
    else:
        gpus = query_gpus()
        if not gpus:
            print("nvidia-smi not found -- running directly, not setting "
                  "CUDA_VISIBLE_DEVICES", flush=True)
            print(f"running: {' '.join(cmd)}", flush=True)
            return subprocess.run(cmd, cwd=repo_root).returncode

        print("current GPU state:", flush=True)
        for idx, free, total, util in gpus:
            print(f"  GPU {idx}: free {free:6d} MiB / {total} MiB, "
                  f"utilization {util:3d}%")
        chosen, free, _total, util = pick_gpu(gpus)
        print(f"chose GPU {chosen} (free {free} MiB, utilization {util}%) -- "
              f"most free memory", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(chosen)
    if "HF_HOME" not in env:
        print("⚠️  HF_HOME not set -- SD2 will download to ~/.cache, which is NOT "
              "writable on this server. Export it first:\n"
              "    export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache", flush=True)

    print(f"running: CUDA_VISIBLE_DEVICES={chosen} {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=repo_root).returncode


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, default=None,
                        help="force a GPU index, skipping auto-detection")
    parser.add_argument(
        "--retries", type=int, default=3,
        help="retries when the job dies. On a shared server the GPU can be taken "
             "between reading nvidia-smi and allocating; a retry re-reads "
             "nvidia-smi and may pick a different GPU.")
    parser.add_argument("--retry-wait", type=int, default=60,
                        help="seconds to wait before each retry")
    parser.add_argument(
        "target", nargs=argparse.REMAINDER,
        help="the script plus its arguments, after --, e.g. "
             "-- tools/check_attention_separates.py --split val")
    args = parser.parse_args()

    target = args.target
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("no script given -- put it after --, e.g. "
                     "-- tools/check_attention_separates.py --split val")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable] + target

    for attempt in range(1, args.retries + 2):
        rc = _pick_and_run(args, cmd, repo_root)
        if rc == 0:
            raise SystemExit(0)

        # A NEGATIVE rc means the child was killed by a signal (-15 SIGTERM from
        # `kill`, -9 SIGKILL, -2 SIGINT). That is A DELIBERATE STOP BY THE USER,
        # not a broken job -- retrying does the opposite of what they asked.
        #
        # This happened upstream (2026-09-04): the user killed a job's PID, the
        # wrapper saw rc=-15, assumed OOM, and RESTARTED the job they had just
        # killed -- a process that "came back to life" running already-loaded old
        # code, unstoppable by killing that same PID again.
        if rc < 0:
            import signal
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = f"signal {-rc}"
            print(f"\nJob stopped by {name} (rc={rc}) -- an INTENTIONAL stop, not a "
                  f"failure, so NOT retrying.", flush=True)
            raise SystemExit(128 + (-rc))

        if attempt > args.retries:
            print(f"EXHAUSTED {args.retries} retries, every one exited != 0 -- giving "
                  f"up. If the traceback above is CUDA out of memory, the GPU is "
                  f"continuously occupied (wait and rerun, or force --gpu <id>). Any "
                  f"other error: retrying cannot help -- fix the code.", flush=True)
            raise SystemExit(rc)

        print(f"\n[retry {attempt}/{args.retries}] job exited with code {rc}. On a "
              f"shared server the usual cause is OOM, but the exit code cannot "
              f"distinguish OOM from a code bug -- IF all {args.retries} attempts "
              f"fail, read the traceback above, it is very likely a real bug. "
              f"Waiting {args.retry_wait}s, then re-reading GPU state...\n", flush=True)
        time.sleep(args.retry_wait)


if __name__ == "__main__":
    main()
