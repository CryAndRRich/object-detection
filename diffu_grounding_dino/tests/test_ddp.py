"""Verification step 8: does a 2-process training step actually parallelize?

Spawns 2 real processes and runs one ``engine.train_one_epoch`` step through a
``DistributedDataParallel``-wrapped model, using the ``gloo`` backend on CPU
instead of ``nccl``/CUDA -- same code path ``main.py`` uses for a real 2xT4
Kaggle run (``DistributedSampler``, DDP gradient all-reduce,
``utils.reduce_dict`` loss averaging), just swapped to a backend that runs
without a GPU. This is what catches a wiring bug (a diffusion module that
silently isn't on the gradient path, an unused-parameter DDP hang, ranks
diverging after a step) *before* burning Kaggle GPU-hours finding out.

    python tests/test_ddp.py

Skips (does not fail) if ``gloo`` is unavailable, which should not happen on
any normal CPU-only PyTorch install.
"""

import os
import socket
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import train_one_epoch  # noqa: E402
from tests.tiny import build_tiny_model, fake_batch  # noqa: E402

WORLD_SIZE = 2
TIMEOUT_S = 120


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _Args:
    use_diffusion = True
    amp = False
    debug = False
    onecyclelr = False
    diff_warmup_iters = 0
    diff_warmup_freeze_keywords = []


def _worker(rank: int, world_size: int, port: int, result_queue, tmpdir: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    try:
        torch.manual_seed(0)  # identical init on every rank, like real DDP construction
        model, criterion, _, cfg = build_tiny_model(use_diffusion=True)
        ddp_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=1e-2)

        torch.manual_seed(rank)  # each rank gets a different shard, like DistributedSampler would
        batch = fake_batch(cfg, num_boxes=(3, 2))
        loader = [batch]

        train_one_epoch(ddp_model, criterion, loader, optimizer, torch.device("cpu"), epoch=0, args=_Args())

        # Saved to a file and only the path put on the queue -- putting the state
        # dict's tensors directly on a multiprocessing.Queue relies on torch's
        # shared-memory file-descriptor IPC (resource_sharer), which some
        # sandboxed/containerized hosts (observed on Kaggle) don't support,
        # raising FileNotFoundError from rebuild_storage_fd. A plain file avoids
        # that IPC path entirely and works the same everywhere.
        state = {k: v.clone() for k, v in ddp_model.module.state_dict().items()}
        state_path = os.path.join(tmpdir, f"rank{rank}.pt")
        torch.save(state, state_path)
        result_queue.put((rank, "ok", state_path))
    except Exception as exc:  # noqa: BLE001
        result_queue.put((rank, "error", repr(exc)))
    finally:
        dist.destroy_process_group()


def test_ddp_training_step_keeps_ranks_in_sync():
    if not dist.is_gloo_available():
        print("  skip  test_ddp_training_step_keeps_ranks_in_sync (gloo backend unavailable)")
        return

    port = _free_port()
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    with tempfile.TemporaryDirectory() as tmpdir:
        procs = [
            ctx.Process(target=_worker, args=(rank, WORLD_SIZE, port, result_queue, tmpdir))
            for rank in range(WORLD_SIZE)
        ]
        for p in procs:
            p.start()

        results = [result_queue.get(timeout=TIMEOUT_S) for _ in range(WORLD_SIZE)]
        for p in procs:
            p.join(timeout=TIMEOUT_S)

        errors = [(rank, msg) for rank, status, msg in results if status == "error"]
        if errors:
            raise AssertionError(f"rank(s) raised during DDP training step: {errors}")
        for p in procs:
            assert p.exitcode == 0, f"rank process exited with code {p.exitcode}"

        states = {rank: torch.load(path, map_location="cpu", weights_only=False) for rank, _, path in results}
    state0, state1 = states[0], states[1]
    assert state0.keys() == state1.keys(), "rank 0/1 ended up with different parameter sets"
    for key in state0:
        assert torch.allclose(state0[key], state1[key], atol=1e-6), (
            f"rank 0 and rank 1 diverged on {key} after one DDP step -- gradient "
            "all-reduce did not keep the two processes in sync"
        )
    print("  ok    test_ddp_training_step_keeps_ranks_in_sync")


def main():
    test_ddp_training_step_keeps_ranks_in_sync()
    print("1/1 passed")


if __name__ == "__main__":
    main()
