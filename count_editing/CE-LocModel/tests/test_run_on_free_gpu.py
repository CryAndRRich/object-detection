"""`pick_gpu` must choose on FREE MEMORY and nothing else.

This exists because of a real failure on 2026-09-11. The picker filtered on
utilization first (keep GPUs below 50 % busy, then take the most free memory
within that group). nvidia-smi reported:

    GPU 0: free  1238 MiB, utilization  0%
    GPU 1: free  6016 MiB, utilization 66%
    GPU 2: free 15842 MiB, utilization 72%

Only GPU 0 passed the utilization filter, so the pool had one element and the
"most free memory" step had nothing left to decide. The job was sent to the GPU
with the LEAST free memory of the three -- the exact opposite of the script's
name and its stated purpose.
"""

import importlib.util
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_on_free_gpu", os.path.join(_HERE, "tools", "run_on_free_gpu.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# (index, free_mib, total_mib, util_percent) -- the shape query_gpus returns.
REAL_2026_09_11 = [(0, 1238, 24576, 0), (1, 6016, 24576, 66), (2, 15842, 24576, 72)]


def test_the_actual_failure_is_fixed():
    assert _load().pick_gpu(REAL_2026_09_11)[0] == 2, (
        "picked a GPU other than the one with the most free memory -- this is the "
        "2026-09-11 regression, where a utilization filter sent the job to the GPU "
        "with 1,238 MiB free while 15,842 MiB sat idle on another")


def test_busy_gpu_wins_when_it_has_the_memory():
    """The negative control: utilization must not be able to veto a choice."""
    m = _load()
    assert m.pick_gpu([(0, 1000, 24576, 0), (1, 24000, 24576, 99)])[0] == 1


def test_idle_gpu_wins_when_it_has_the_memory():
    m = _load()
    assert m.pick_gpu([(0, 24000, 24576, 99), (1, 1000, 24576, 0)])[0] == 0


def test_returns_a_bare_tuple_not_a_pair():
    """It used to return `(gpu, had_idle)`; the call site unpacks 4 fields now."""
    g = _load().pick_gpu(REAL_2026_09_11)
    assert len(g) == 4 and all(isinstance(x, int) for x in g)


def test_no_utilization_threshold_survives_anywhere():
    """A constant left behind is an invitation to reintroduce the filter."""
    src = open(os.path.join(_HERE, "tools", "run_on_free_gpu.py")).read()
    assert "UTIL_BUSY_THRESHOLD" not in src
    code = "\n".join(l.split("#")[0] for l in src.splitlines())
    body = code.split("def pick_gpu")[1].split("def ")[0]
    assert "g[3]" not in body and "util" not in body.replace("utilization", ""), (
        "pick_gpu reads the utilization field again -- it must choose on g[1] "
        "(free memory) alone")
