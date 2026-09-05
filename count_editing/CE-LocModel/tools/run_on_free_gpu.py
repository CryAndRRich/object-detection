"""Run any Python script (train.py, eval.py, tools/*.py) on the server's freest
GPU, selected automatically via nvidia-smi.

Logic ported from object-detection/diffu_grounding_dino/tools/run_train.py — same
reason: this server is shared with other people and jobs, so defaulting to GPU 0
is not viable. Unlike the original (which wraps a fixed main.py), this script
wraps ANY Python script passed to it, because CE-LocModel has several entry
points that need to run on the server (train.py, eval.py, tools/*.py).

  1. Query nvidia-smi for every GPU: free memory + utilization.
  2. Pick the FREEST GPU: prefer GPUs not busy computing (utilization below a
     threshold); within that group, take the one with the most free memory.
  3. Set CUDA_VISIBLE_DEVICES=<chosen gpu> and run the given Python script with
     all remaining arguments.

THERE IS NO MINIMUM FREE-MEMORY THRESHOLD. It was tried twice and failed both times:

  - Version 1 hard-coded 15000 MiB for every job. On 2026-08-31 variant (d) and
    both eval jobs were skipped within one second, even though GPU 1 had 10,607
    MiB free — plenty to run them.
  - Version 2 inferred the threshold from the script name + arguments. On
    2026-09-04, tools/visualize_predictions.py was misclassified as a training job
    (10,580 MiB) purely because its name did not contain "eval", then waited for a
    GPU for 2 hours for nothing.

The lesson: guessing memory needs from a command line is always wrong somewhere,
and when it is wrong the consequence (the job never runs) is far worse than what
it was guarding against (an OOM, which torch reports clearly and which costs only
a few seconds). Just pick the freest GPU and run; if VRAM really is insufficient
the OOM traceback says so directly, and --retries will try again on a possibly
different GPU (each attempt re-reads nvidia-smi).

Usage — the Python script goes immediately after --, and all of ITS arguments
follow it:

    python tools/run_on_free_gpu.py -- train.py --config config/experiment_a.yaml
    python tools/run_on_free_gpu.py -- eval.py --ckpt checkpoints/experiment_a/best.pth --split test
    python tools/run_on_free_gpu.py -- tools/profile_and_memory.py --batch-size 8
    python tools/run_on_free_gpu.py -- tools/overfit_one.py --steps 300

To force a specific GPU: --gpu <index> (placed BEFORE --).
"""

import argparse
import os
import subprocess
import time
import sys

UTIL_BUSY_THRESHOLD = 50  # percent; a GPU above this is considered busy computing


def query_gpus():
    """``[(index, free_mib, total_mib, util_percent), ...]`` from nvidia-smi.

    Returns ``[]`` when nvidia-smi is absent (a dev machine with no GPU) -- in that
    case just run directly, without setting CUDA_VISIBLE_DEVICES.
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
    out = out.stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total, util = (int(x.strip()) for x in line.split(","))
        gpus.append((idx, total - used, total, util))
    return gpus


def pick_gpu(gpus):
    """The freest GPU: ``(gpu, had_idle)``.

    Prefers compute-idle GPUs (``utilization < UTIL_BUSY_THRESHOLD``); within the
    chosen group, takes the one with the most free memory. No GPU is ever excluded
    for lack of memory — see the module docstring.
    """
    idle = [g for g in gpus if g[3] < UTIL_BUSY_THRESHOLD]
    pool = idle if idle else gpus
    return max(pool, key=lambda g: g[1]), bool(idle)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", type=int, default=None,
                        help="force a specific GPU index, skipping auto-detection")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="how many times to retry when the job dies. On a shared server the GPU can "
        "be taken by another job between reading nvidia-smi and actually allocating -- a "
        "retry re-reads nvidia-smi and may therefore pick a different GPU.",
    )
    parser.add_argument("--retry-wait", type=int, default=60,
                        help="seconds to wait before each retry")
    parser.add_argument(
        "target", nargs=argparse.REMAINDER,
        help="the Python script to run plus its arguments, placed after --, e.g. "
        "-- train.py --config config/experiment_a.yaml",
    )
    args = parser.parse_args()

    target = args.target
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("no script given — put it after --, e.g. "
                     "-- train.py --config config/experiment_a.yaml")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable] + target

    for attempt in range(1, args.retries + 2):
        rc = _pick_and_run(args, cmd, repo_root)
        if rc == 0:
            raise SystemExit(0)
        # A NEGATIVE rc means the child was killed by a signal (-15 SIGTERM from
        # `kill`, -9 SIGKILL, -2 SIGINT from Ctrl+C). That is A DELIBERATE STOP BY
        # THE USER, not a broken job -- retrying here does the opposite of what
        # they asked for.
        #
        # This actually happened (2026-09-04): the user `kill`ed the viz job's PID,
        # this wrapper saw rc=-15, assumed OOM, and RESTARTED the job they had just
        # killed. The result was a process that "came back to life" from nowhere,
        # running the already-loaded old code, and could not be stopped by killing
        # that same PID again.
        if rc < 0:
            import signal
            try:
                name = signal.Signals(-rc).name
            except ValueError:
                name = f"signal {-rc}"
            print(f"\nJob stopped by {name} (rc={rc}) -- this is an INTENTIONAL stop, "
                  f"not a failure, so NOT retrying.", flush=True)
            raise SystemExit(128 + (-rc))
        if attempt > args.retries:
            print(f"EXHAUSTED {args.retries} retries, every one exited != 0 -- giving up. If the "
                  f"traceback above is CUDA out of memory, the GPU is continuously occupied (wait "
                  f"and rerun, force --gpu <id>, or lower --batch_size). If it is any other error, "
                  f"retrying cannot help -- fix the code.", flush=True)
            raise SystemExit(rc)
        print(f"\n[retry {attempt}/{args.retries}] job exited with code {rc}. On a shared server "
              f"the usual cause is OOM (the GPU was taken between reading nvidia-smi and "
              f"allocating), but the exit code cannot distinguish OOM from a code bug -- IF all "
              f"{args.retries} attempts fail, read the traceback above, it is very likely a real "
              f"bug. Waiting {args.retry_wait}s, then re-reading GPU state...\n", flush=True)
        time.sleep(args.retry_wait)


def _pick_and_run(args, cmd, repo_root):
    if args.gpu is not None:
        chosen_index = args.gpu
        print(f"using GPU {chosen_index} (forced via --gpu)", flush=True)
    else:
        gpus = query_gpus()
        if not gpus:
            print("nvidia-smi not found -- running directly, not setting CUDA_VISIBLE_DEVICES",
                  flush=True)
            print(f"running: {' '.join(cmd)}", flush=True)
            return subprocess.run(cmd, cwd=repo_root).returncode
        print("current GPU state:", flush=True)
        for idx, free, total, util in gpus:
            print(f"  GPU {idx}: free {free:6d} MiB / {total} MiB, utilization {util:3d}%")
        (chosen_index, free, total, util), had_idle = pick_gpu(gpus)
        if had_idle:
            print(f"chose GPU {chosen_index} (free {free} MiB, utilization {util}%)", flush=True)
        else:
            print(f"chose GPU {chosen_index} (free {free} MiB, utilization {util}%) -- EVERY GPU "
                  f"is busy computing (>{UTIL_BUSY_THRESHOLD}%), so the job will contend for "
                  f"resources and run slower, but it will run.", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(chosen_index)

    print(f"running: CUDA_VISIBLE_DEVICES={chosen_index} {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=repo_root).returncode


if __name__ == "__main__":
    main()
