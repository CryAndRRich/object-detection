"""`tools/latency.py` — một định nghĩa latency 1 ảnh cho mọi method: tập đo = pilot VLM, warm-up khác tập đo, đọc trước file, từng ảnh
một, lỗi API giữ hàng nhưng ra khỏi TB; trọn luồng ZETA (BootPlace / HiddenObjects tí hon), VLM (backend giả), CE-Loc (SLOW)."""

import importlib.util
import json
import os

import pytest

from tests.baseline.helpers import fake_ce130, fake_ckpt, fake_clip, tiny_cfg
from tests.fakes import fake_cocount_review

_spec = importlib.util.spec_from_file_location("latency_tool", os.path.join(os.path.dirname(__file__), "..", "tools", "latency.py"))
lat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lat)


def test_pick_is_vlm_pilot_and_warmup_disjoint():
    from vlm.runner import select
    samples = [{"image_id": f"im{i}"} for i in range(30)]
    meas, warm = lat.pick(samples, 5, 3)
    assert meas == select(samples, 0, 5) and len(warm) == 3
    assert not {s["image_id"] for s in warm} & {s["image_id"] for s in meas}
    with pytest.raises(ValueError, match="< n"):
        lat.pick(samples[:6], 5, 3)


def test_measure_one_by_one_prefetch_first_errors_out_of_mean():
    ev, t = [], [0.0]

    def clock():
        return t[0]

    def run(s):
        ev.append(("run", s["image_id"]))
        t[0] += s["cost"]
        if s["image_id"] == "bad":
            raise RuntimeError("HTTP 503")
        return {"boxes": [[0, 0, 1, 1]] * 3, "n_valid": 3}

    meas = [{"image_id": "a", "cost": 2.0}, {"image_id": "bad", "cost": 9.0}, {"image_id": "b", "cost": 4.0}]
    warm = [{"image_id": "w", "cost": 100.0}]
    rows = lat.measure(run, meas, warm, prefetch=lambda s: ev.append(("pre", s["image_id"])), sync=lambda: ev.append(("sync",)),
                       clock=clock)
    assert [e for e in ev if e[0] == "pre"] == [("pre", x) for x in ("w", "a", "bad", "b")]
    assert ev.index(("run", "w")) > ev.index(("pre", "b"))                    # đọc trước MỌI file rồi mới chạy
    assert [r["sec"] for r in rows] == [2.0, 9.0, 4.0] and rows[0]["n_boxes"] == 3 and rows[0]["n_valid"] == 3
    assert rows[1]["error"] == "HTTP 503" and rows[1]["n_boxes"] == 0
    sm = lat.summarize({"ce130": rows, "cocount": rows[:1]})
    assert sm["per_dataset"]["ce130"] == {"n": 3, "n_error": 1, "latency_s": 3.0} and sm["latency_s"] == pytest.approx(8 / 3)


@pytest.fixture
def data(tmp_path):
    _, samples_root, ti = fake_ce130(str(tmp_path / "data"))
    croot = str(tmp_path / "cocount")
    review = fake_cocount_review(croot)
    return ["--turn-index", ti, "--samples-root", samples_root, "--cocount-root", croot, "--cocount-review", review,
            "--n", "2", "--warmup", "1", "--out-root", str(tmp_path / "lat")]


def _read(tmp_path, name):
    with open(tmp_path / "lat" / f"{name}.json") as f:
        return json.load(f)


@pytest.mark.parametrize("run", ["zeta1", "zeta3"])
def test_zeta_flow(tmp_path, monkeypatch, data, run):
    """Trọn luồng `zeta`: 2 ảnh đo + 1 warm-up / tập, đúng K box, CLIP text tính MỘT lần / tập (không mỗi ảnh)."""
    calls = []
    monkeypatch.setattr("baseline.methods.hiddenobjects.encode_class_names",
                        lambda names, *a, **k: calls.append(list(names)) or fake_clip(names))
    cfg_path, cfg = tiny_cfg(tmp_path, run)
    ck = fake_ckpt(cfg, tmp_path / "init.pth")
    lat.main(["zeta", "--config", cfg_path, "--ckpt", str(ck), "--device", "cpu", *data])
    r = _read(tmp_path, f"{run}_tiny_init")
    assert r["kind"] == "local" and r["method"] == "zeta" and r["n_per_dataset"] == 2 and len(r["images"]) == 4
    assert all(x["n_boxes"] == cfg["predict"]["k"] and x["sec"] > 0 for x in r["images"])
    assert set(r["per_dataset"]) == {"ce130", "cocount"} and r["latency_s"] > 0
    if run == "zeta3":
        assert len(calls) == 1 + 2                       # nạp checkpoint công bố (lstsq) + một lần / tập


def test_oracleloc_flow(tmp_path, data):
    """OracleLoc (không weight): `zeta` không --ckpt, CPU 1 tiến trình, MỘT tiêu chí, đủ K box; tên file = <config>_<tiêu chí>."""
    import glob
    cfg_path, cfg = tiny_cfg(tmp_path, "zeta7")
    lat.main(["zeta", "--config", cfg_path, "--criterion", "euclid_px-free-k1", *data])
    files = glob.glob(str(tmp_path / "lat" / "*_euclid_px-free-k1.json"))
    assert len(files) == 1
    r = json.load(open(files[0]))
    assert r["criterion"] == "euclid_px-free-k1" and r["ckpt"] is None and r["device"].startswith("cpu") and len(r["images"]) == 4
    assert all(x["n_boxes"] == cfg["predict"]["k"] and x["sec"] > 0 for x in r["images"])


def test_vlm_flow(tmp_path, monkeypatch, data):
    """Trọn luồng `vlm` với backend giả: mỗi request 1 ảnh, box đã parse + fill đủ K, API ⇒ kind api."""
    from tests.vlm.test_runner import CFG, FakeBackend
    import yaml
    fb = FakeBackend()
    sizes = []

    class One(FakeBackend):
        def batch(self, jobs):
            sizes.append(len(jobs))
            return fb.batch(jobs)
    monkeypatch.setattr("vlm.backends.make_backend", lambda cfg, log=print: One())
    monkeypatch.setattr("vlm.env.load_env", lambda *a, **k: [])                     # không đụng vlm/.env thật
    p = tmp_path / "vlm.yaml"
    with open(p, "w") as f:
        yaml.safe_dump({"base": os.path.abspath(CFG), "k": 5, "icl": 0}, f)
    lat.main(["vlm", "--config", str(p), *data])
    r = _read(tmp_path, "gpt-6-luna")
    assert r["kind"] == "api" and r["gpu"] is None and sizes == [1] * 6               # (1 warm-up + 2 đo) × 2 tập, từng ảnh
    assert all(x["n_boxes"] == 5 and x["n_valid"] == 3 for x in r["images"])


@pytest.mark.slow
def test_latency_celoc_flow(tmp_path, monkeypatch):
    """SLOW: train DELTA8 tí hon 1 iter -> `celoc` trên CE-130 + CoCount giả: box 30 = eval.n_samples, về pixel ảnh gốc."""
    import ce_localization.train as ta
    from tests.ce_localization.helpers import _fake_text_table, _fake_turn_index, _patch_clip, _run_train, run_cfg
    from tests.fakes import fake_cocount, fake_review
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, _ = run_cfg(tmp_path, base, "delta8")
    w = str(tmp_path / "w")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", w, "--max-iter", "1"])
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    _patch_clip(monkeypatch)
    cc = fake_cocount(str(tmp_path / "cc"))
    rv = fake_review(cc, str(tmp_path / "rv.jsonl"))
    lat.main(["celoc", "--ckpt", os.path.join(w, "last.pth"), "--device", "cpu", "--turn-index", os.path.join(base, "turn_index.json"),
              "--samples-root", os.path.join(base, "samples"),
              "--cocount-root", cc, "--cocount-review", rv, "--n", "1", "--warmup", "1", "--out-root", str(tmp_path / "lat")])
    r = _read(tmp_path, "w_last")
    assert r["K"] == 3 and len(r["images"]) == 2 and all(x["n_boxes"] == 3 for x in r["images"]) and r["sampler"] == "mock"


# ----------------------------------------------------------------------------- bộ nhớ GPU (`--mem`)

def test_proc_tree_and_query_apps(monkeypatch):
    ps = "  1     0\n 10     1\n 11    10\n 12    11\n 20     1\n 13    10\n"
    assert lat.proc_tree(10, ps) == {10, 11, 12, 13} and lat.proc_tree(99, ps) == {99}

    class R:
        stdout = "11, 1024\n12, [N/A]\n30, 512\n11, 256\nNo running processes found\n"
    monkeypatch.setattr(lat.subprocess, "run", lambda *a, **k: R())
    assert lat.query_apps() == {11: 1280.0, 30: 512.0}                 # cùng PID trên 2 GPU cộng lại; [N/A] bỏ


def test_gpumem_window_counts_only_own_tree():
    """Đỉnh cửa sổ = max các mẫu giữa begin / end của cây tiến trình mình (job khác không tính); cây dựng lại định kỳ (tiến trình con
    sinh sau, như vLLM EngineCore)."""
    seq = iter([{1: 1024.0, 99: 9000.0}, {1: 1024.0, 2: 3072.0}, {1: 1024.0, 2: 1024.0}, {1: 1024.0}, {1: 512.0}])
    trees = iter([{1}, {1, 2}])
    m = lat.GpuMem(query=lambda: next(seq), tree=lambda r: next(trees), root=1, tree_every=1)
    assert m.load() == 1.0 and m.seen                                     # 1024 MiB = 1 GiB; PID 99 (người khác) không tính
    m.begin()                                                             # mẫu 2: 4096 (cây mới có PID 2)
    m.tree = lambda r: {1, 2}
    m.sample()                                                            # 2048
    out = m.end()                                                         # 1024
    assert out == {"proc_gib": 4.0}


def test_gpumem_window_resets():
    vals = iter([{1: 4096.0}, {1: 2048.0}, {1: 1024.0}, {1: 1024.0}])
    m = lat.GpuMem(query=lambda: next(vals), tree=lambda r: {1}, root=1)
    m.begin(); r1 = m.end()
    m.begin(); r2 = m.end()
    assert r1 == {"proc_gib": 4.0} and r2 == {"proc_gib": 1.0}


def test_gpumem_thread_samples_in_background():
    import time as _t
    n = []
    m = lat.GpuMem(poll=0.01, query=lambda: n.append(1) or {1: 100.0}, tree=lambda r: {1}, root=1).start()
    _t.sleep(0.1)
    m.stop()
    assert len(n) >= 3 and not m.th.is_alive()


def test_summarize_mem_and_set_dotted():
    rows = {"ce130": [{"image_id": "a", "proc_gib": 2.0, "alloc_gib": 1.0, "reserved_gib": 1.5},
                      {"image_id": "b", "proc_gib": 3.0, "alloc_gib": 0.5, "reserved_gib": 1.5}],
            "cocount": [{"image_id": "c", "proc_gib": 2.5, "alloc_gib": 1.2, "reserved_gib": 2.0}]}
    s = lat.summarize_mem(rows, 1.7)
    assert (s["load_gib"], s["peak_proc_gib"], s["peak_proc_image"], s["peak_alloc_gib"], s["peak_alloc_image"]) == (1.7, 3.0, "b", 1.2, "c")
    assert s["per_dataset"]["ce130"]["peak_alloc_image"] == "a" and s["peak_reserved_image"] == "c"
    assert "peak_alloc_gib" not in lat.summarize_mem({"x": [{"image_id": "v", "proc_gib": 9.0}]}, 8.0)   # vLLM: chỉ proc
    shared = {"kv_cache_gib": 14.0, "gpu_mem_gib": 34.0}
    cfg = {"vllm": shared, "engine": {"max_num_seqs": 64}}
    done = lat.set_dotted(cfg, ["vllm.kv_cache_gib=2", "engine.max_num_seqs=1", "engine.enforce_eager=true"])
    assert cfg["vllm"] == {"kv_cache_gib": 2, "gpu_mem_gib": 34.0} and shared["kv_cache_gib"] == 14.0      # không sửa dict gốc
    assert cfg["engine"] == {"max_num_seqs": 1, "enforce_eager": True} and done["engine.max_num_seqs"] == 1
    with pytest.raises(SystemExit):
        lat.set_dotted(cfg, ["vllm.kv_cache_gib"])


def test_zeta_mem_flow(tmp_path, monkeypatch, data):
    """Trọn luồng `zeta --mem` (CPU, nvidia-smi giả): JSON ở <out-root>/mem/, mỗi ảnh đo có proc_gib, đỉnh + ảnh gây đỉnh; không đè
    JSON latency; API / OracleLoc bị từ chối."""
    import itertools
    mib = itertools.count(1024, 256)
    monkeypatch.setattr(lat, "query_apps", lambda: {os.getpid(): float(next(mib))})
    cfg_path, cfg = tiny_cfg(tmp_path, "zeta1")
    ck = fake_ckpt(cfg, tmp_path / "init.pth")
    lat.main(["zeta", "--config", cfg_path, "--ckpt", str(ck), "--device", "cpu", "--mem", "--mem-poll", "0.01", *data])
    assert not (tmp_path / "lat" / "zeta1_tiny_init.json").exists()
    with open(tmp_path / "lat" / "mem" / "zeta1_tiny_init.json") as f:
        r = json.load(f)
    m = r["mem"]
    assert m["load_gib"] >= 1.0 and m["pid_seen"] and m["poll_s"] == 0.01 and "peak_alloc_gib" not in m     # CPU: không torch.cuda
    assert len(r["images"]) == 4 and all(x["proc_gib"] > m["load_gib"] for x in r["images"])
    assert m["peak_proc_gib"] == max(x["proc_gib"] for x in r["images"]) and m["peak_proc_image"] in {x["image_id"] for x in r["images"]}
    assert set(m["per_dataset"]) == {"ce130", "cocount"}
    cfg7, _ = tiny_cfg(tmp_path, "zeta7")
    with pytest.raises(SystemExit, match="--mem"):
        lat.main(["zeta", "--config", cfg7, "--criterion", "euclid_px-free-k1", "--mem", *data])


def test_vlm_mem_overrides_reach_backend(tmp_path, monkeypatch, data):
    """`vlm --vllm-set` đổi config TRƯỚC khi dựng backend (KV / số chuỗi batch 1), ghi vào info; prompt_hash không đổi."""
    from tests.vlm.test_runner import CFG, FakeBackend
    import yaml
    seen = []
    monkeypatch.setattr("vlm.backends.make_backend", lambda cfg, log=print: seen.append(cfg) or FakeBackend())
    monkeypatch.setattr("vlm.env.load_env", lambda *a, **k: [])
    p = tmp_path / "vlm.yaml"
    with open(p, "w") as f:
        yaml.safe_dump({"base": os.path.abspath(CFG), "k": 5, "icl": 0, "engine": {"max_num_seqs": 64}}, f)
    lat.main(["vlm", "--config", str(p), *data])
    h0 = _read(tmp_path, "gpt-6-luna")["prompt_hash"]
    lat.main(["vlm", "--config", str(p), "--vllm-set", "vllm.kv_cache_gib=2", "engine.max_num_seqs=1", "--name", "ov", *data])
    r = _read(tmp_path, "ov")
    assert seen[-1]["vllm"]["kv_cache_gib"] == 2 and seen[-1]["engine"]["max_num_seqs"] == 1
    lat.main(["vlm", "--config", str(p), "--model", "w/merged", "--vllm-set", "engine.max_num_seqs=1", "--name", "m", *data])
    assert seen[-1]["model"] == "w/merged" and _read(tmp_path, "m")["model"] == "w/merged"
    assert r["overrides"] == {"vllm.kv_cache_gib": 2, "engine.max_num_seqs": 1} and r["prompt_hash"] == h0
    with pytest.raises(SystemExit, match="prompt_hash"):                              # khoá trong hash ⇒ dừng
        lat.main(["vlm", "--config", str(p), "--vllm-set", "max_tokens=7", *data])
    with pytest.raises(SystemExit, match="--mem"):                                    # backend openrouter = API
        lat.main(["vlm", "--config", str(p), "--mem", *data])
