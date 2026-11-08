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


# ============================================================================
# --grab-margin-gib (người dùng 2026-11-08: GPU bị job khác chiếm -> gom dần, chừa lề cho người khác)
# ============================================================================

def _grab_torch(frees):
    """torch giả: lần đọc thứ i thấy `frees[i]` GiB trống (trước khi trừ phần đã giữ)."""
    t = types.ModuleType("torch")
    t.uint8 = "u8"
    oom_t = type("OutOfMemoryError", (RuntimeError,), {})
    st = {"live": 0, "cached": 0, "reads": 0, "emptied": 0, "allocs": []}

    def mem_get_info():
        i = min(st["reads"], len(frees) - 1)
        st["reads"] += 1
        return (frees[i] * 2 ** 30 - st["live"] - st["cached"], 48 * 2 ** 30)

    class Block:
        def __init__(self, n):
            self.n = n
            st["live"] += n

        def __del__(self):
            st["live"] -= self.n
            st["cached"] += self.n

    def empty(n, dtype=None, device=None):
        st["allocs"].append(n / 2 ** 30)
        return Block(n)

    def empty_cache():
        st["emptied"] += 1
        st["cached"] = 0

    t.empty = empty
    t.cuda = types.SimpleNamespace(is_available=lambda: True, memory_reserved=lambda: st["cached"] + st["live"],
                                   empty_cache=empty_cache, mem_get_info=mem_get_info, OutOfMemoryError=oom_t)
    return t, st


def test_grab_waits_and_keeps_margin(monkeypatch):
    """Trống 23,5 -> 23,5 -> 34 -> 48: gom 20 rồi thêm khi job khác nhả, luôn chừa 3,5 GiB; đủ 40 thì giữ trong cache."""
    t, st = _grab_torch([23.5, 23.5, 34.0, 48.0])
    monkeypatch.setitem(sys.modules, "torch", t)
    sleeps = []
    got = _hold_mod().grab(40, 3.5, poll=10, log=lambda *a, **k: None, sleep=sleeps.append, clock=lambda: 0.0)
    assert st["allocs"] == [20.0, 10.5, 9.5] and got == 40
    assert len(sleeps) == 3                                # đọc 4 lần, ngủ giữa các lần
    assert st["live"] == 0 and st["cached"] == 40 * 2 ** 30  # nhả vào cache, không về driver
    t.cuda.empty_cache()
    assert st["cached"] == 40 * 2 ** 30 and st["emptied"] == 0


def test_grab_release_returns_memory(monkeypatch):
    t, st = _grab_torch([48.0])
    monkeypatch.setitem(sys.modules, "torch", t)
    got = _hold_mod().grab(35, 3.5, release=True, log=lambda *a, **k: None, sleep=lambda s: None, clock=lambda: 0.0)
    assert got == 0.0 and st["allocs"] == [35.0] and st["emptied"] == 1 and st["cached"] == 0


def test_grab_skips_tiny_and_survives_race(monkeypatch):
    """Trống chỉ hơn lề 0,1 GiB -> không xin mẩu vụn; OOM giữa lúc đọc và lúc xin -> đọc lại, không thoát."""
    t, st = _grab_torch([3.6, 20.0, 20.0])
    calls = {"n": 0}
    real = t.empty

    def flaky(n, dtype=None, device=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise t.cuda.OutOfMemoryError("race")
        return real(n, dtype, device)

    t.empty = flaky
    monkeypatch.setitem(sys.modules, "torch", t)
    got = _hold_mod().grab(10, 3.5, log=lambda *a, **k: None, sleep=lambda s: None, clock=lambda: 0.0)
    assert got == 10 and st["allocs"] == [10.0] and calls["n"] == 2


def test_build_cmd_grab_flags():
    m = _load()
    cmd = m.build_cmd(["finetune.py", "--x"], 40, 3.5)
    assert cmd[2:] == ["--grab", "3.5", "40", "finetune.py", "--x"]
    cmd = m.build_cmd(["run.py"], 35, 3.5, release=True)
    assert cmd[2:] == ["--grab", "3.5", "--release", "35", "run.py"]


def test_gpu_hold_release_runs_child(tmp_path):
    """--release: script chạy ở tiến trình con như `python script.py`, mã thoát giữ nguyên (G = 0 -> không chạm torch)."""
    (tmp_path / "s.py").write_text("import sys\nsys.exit(5 if sys.argv[1:] == ['a'] else 9)\n")
    r = subprocess.run([sys.executable, os.path.join(_HERE, "tools", "gpu_hold.py"), "--grab", "3", "--release", "0",
                        str(tmp_path / "s.py"), "a"], cwd=str(_HERE), capture_output=True, text=True)
    assert r.returncode == 5, r.stderr


def test_keeper_hands_off_to_step_and_regrabs(monkeypatch, tmp_path):
    """Người giữ chỗ: gom đủ -> ready; bước GPU xin (want) -> nhả hết + freed; bước xong -> gom lại -> ready; stop -> thoát."""
    import threading
    import time as _t
    t, st = _grab_torch([48.0])
    monkeypatch.setitem(sys.modules, "torch", t)
    mod = _hold_mod()
    d = str(tmp_path / "k")
    th = threading.Thread(target=mod.keeper, args=(d, 37, 8), kwargs=dict(poll=0.01, log=lambda *a, **k: None), daemon=True)
    th.start()

    def until(c, sec=5):
        end = _t.time() + sec
        while not c():
            assert _t.time() < end, "quá giờ"
            _t.sleep(0.01)

    p = lambda n: os.path.join(d, n)
    until(lambda: os.path.exists(p("ready")))
    assert st["live"] == 37 * 2 ** 30
    open(p("want"), "w").write("123")
    until(lambda: os.path.exists(p("freed")))
    assert st["live"] == 0 and st["cached"] == 0 and not os.path.exists(p("ready"))   # nhả về driver cho bước GPU
    os.remove(p("want"))
    until(lambda: os.path.exists(p("ready")))
    assert st["live"] == 37 * 2 ** 30 and not os.path.exists(p("freed"))
    open(p("stop"), "w").close()
    th.join(5)
    assert not th.is_alive() and st["live"] == 0 and not os.path.exists(p("keeper.pid"))


def test_take_over_waits_for_keeper_then_cleans_want(tmp_path):
    """Bước GPU: tạo want, chờ freed; người giữ chỗ không còn sống -> không chờ mãi."""
    mod = _hold_mod()
    d = tmp_path / "k"
    d.mkdir()
    (d / "keeper.pid").write_text(str(os.getpid()))
    calls = {"n": 0}

    def sleep(_):
        calls["n"] += 1
        if calls["n"] == 3:
            (d / "freed").write_text("")
    mod._take_over(str(d), 1.0, lambda *a, **k: None, sleep)
    assert (d / "want").exists() and calls["n"] == 3
    (d / "freed").unlink()
    (d / "keeper.pid").write_text("999999999")             # người giữ chết -> thôi chờ
    mod._take_over(str(d), 1.0, lambda *a, **k: None, lambda _: None)


def test_build_cmd_handoff():
    cmd = _load().build_cmd(["finetune.py"], 37, 1, handoff="/x/k")
    assert cmd[2:] == ["--grab", "1", "--handoff", "/x/k", "37", "finetune.py"]
