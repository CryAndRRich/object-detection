"""tools/run_on_free_gpu.py phải chọn GPU theo BỘ NHỚ TRỐNG và CHỈ theo nó.

Sự cố 2026-09-11: lọc theo utilization trước đã gửi job sang GPU còn 1.238 MiB trong khi
GPU khác trống 15.842 MiB."""

import importlib.util
import os

# ============================================================================
# từ test_run_on_free_gpu.py
# ============================================================================

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



# ============================================================================
# --reserve-gib / tools/gpu_hold.py (người dùng 2026-11-08: giữ trước bộ nhớ cho cả job)
# ============================================================================

import json
import subprocess
import sys
import types

import pytest


def _hold_mod():
    spec = importlib.util.spec_from_file_location("gpu_hold", os.path.join(_HERE, "tools", "gpu_hold.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_cmd_reserve_goes_through_gpu_hold():
    m = _load()
    assert m.build_cmd(["train.py", "--x", "1"]) == [sys.executable, "train.py", "--x", "1"]
    cmd = m.build_cmd(["train.py", "--x", "1"], 40.5)
    assert cmd[0] == sys.executable and cmd[1].endswith(os.path.join("tools", "gpu_hold.py")) and os.path.isfile(cmd[1])
    assert cmd[2:] == ["40.5", "train.py", "--x", "1"]


def test_gpu_hold_runs_script_like_python(tmp_path):
    """Script chạy như `python script.py`: argv, __name__, sys.path[0] = thư mục script, mã thoát giữ nguyên."""
    out = tmp_path / "out.json"
    (tmp_path / "helper_mod.py").write_text("X = 7\n")
    (tmp_path / "s.py").write_text(
        "import json, sys\nimport helper_mod\n"
        f"json.dump({{'argv': sys.argv, 'name': __name__, 'p0': sys.path[0], 'x': helper_mod.X}}, open({str(out)!r}, 'w'))\n"
        "if __name__ == '__main__':\n    sys.exit(3)\n")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "tools", "gpu_hold.py"), "0", str(tmp_path / "s.py"), "a", "--b", "3"],
                       cwd=str(_HERE), capture_output=True, text=True)
    assert r.returncode == 3, r.stderr
    d = json.loads(out.read_text())
    assert d == {"argv": [str(tmp_path / "s.py"), "a", "--b", "3"], "name": "__main__", "p0": str(tmp_path), "x": 7}


def _fake_torch(oom=False):
    t = types.ModuleType("torch")
    t.uint8 = "u8"
    oom_t = type("OutOfMemoryError", (RuntimeError,), {})
    st = {"reserved": 0, "emptied": 0}

    def empty(n, dtype=None, device=None):
        if oom:
            raise oom_t("oom")
        st["reserved"] = n
        return object()

    def empty_cache():
        st["emptied"] += 1
        st["reserved"] = 0

    t.empty = empty
    t.cuda = types.SimpleNamespace(is_available=lambda: True, memory_reserved=lambda: st["reserved"], empty_cache=empty_cache,
                                   mem_get_info=lambda: (3 * 2 ** 30, 48 * 2 ** 30), OutOfMemoryError=oom_t)
    return t, st


def test_hold_reserves_and_disables_empty_cache(monkeypatch):
    t, st = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", t)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    got = _hold_mod().hold(40, log=lambda *a, **k: None)
    assert st["reserved"] == 40 * 2 ** 30 and got == 40
    t.cuda.empty_cache()                                   # code của job gọi empty_cache -> không nhả chỗ giữ
    assert st["reserved"] == 40 * 2 ** 30 and st["emptied"] == 0
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_hold_oom_exits_1_for_retry(monkeypatch):
    t, _ = _fake_torch(oom=True)
    monkeypatch.setitem(sys.modules, "torch", t)
    msgs = []
    with pytest.raises(SystemExit) as e:
        _hold_mod().hold(40, log=lambda *a, **k: msgs.append(a[0]))
    assert e.value.code == 1 and "KHÔNG đủ" in msgs[0]


def test_hold_zero_is_noop(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)       # không được import torch khi G = 0
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    assert _hold_mod().hold(0) == 0.0
