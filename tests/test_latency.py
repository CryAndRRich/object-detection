"""`tools/latency.py` — một định nghĩa latency 1 ảnh cho mọi method: tập đo = pilot VLM, warm-up khác tập đo, đọc trước file, từng ảnh
một, lỗi API giữ hàng nhưng ra khỏi TB; trọn luồng ZETA (BootPlace / HiddenObjects tí hon), VLM (backend giả), CE-Loc (SLOW)."""

import importlib.util
import json
import os

import pytest

from tests.baseline.helpers import fake_ce130, fake_ckpt, fake_cocount, fake_clip, tiny_cfg

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
    review = fake_cocount(croot)
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


def test_latency_celoc_flow(tmp_path, monkeypatch):
    """SLOW: train DELTA8 tí hon 1 iter -> `celoc` trên CE-130 + CoCount giả: box 30 = eval.n_samples, về pixel ảnh gốc."""
    import ce_localization.train as ta
    from tests.ce_localization.helpers import (_fake_cocount, _fake_review, _fake_text_table, _fake_turn_index, _patch_clip,
                                               _run_train, run_cfg)
    base = str(tmp_path / "d")
    os.makedirs(base)
    _fake_turn_index(base)
    c_path, _ = run_cfg(tmp_path, base, "delta8")
    w = str(tmp_path / "w")
    _run_train(monkeypatch, ["--config", c_path, "--save-dir", w, "--max-iter", "1"])
    monkeypatch.setattr(ta, "build_text_table", _fake_text_table)
    _patch_clip(monkeypatch)
    cc = _fake_cocount(str(tmp_path / "cc"))
    rv = _fake_review(cc, str(tmp_path / "rv.jsonl"))
    lat.main(["celoc", "--ckpt", os.path.join(w, "last.pth"), "--device", "cpu", "--turn-index", os.path.join(base, "turn_index.json"),
              "--samples-root", os.path.join(base, "samples"), "--ce130-root", os.path.join(base, "all_phase2_V2"),
              "--cocount-root", cc, "--cocount-review", rv, "--n", "1", "--warmup", "1", "--out-root", str(tmp_path / "lat")])
    r = _read(tmp_path, "w_last")
    assert r["K"] == 3 and len(r["images"]) == 2 and all(x["n_boxes"] == 3 for x in r["images"]) and r["sampler"] == "mock"
