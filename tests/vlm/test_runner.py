"""ZETA5 / ZETA6 trọn luồng với backend giả (dữ liệu giả, không mạng): pilot ổn định + là tập con của all (all dùng lại câu trả lời pilot),
resume, đổi prompt ⇒ hỏi lại, thử lại khi 0 box rồi box cả ảnh, lỗi API không ghi + dừng sau N lỗi liên tiếp, --max-cost, chấm box
score cao nhất (chính) / min C-NLL trong K (tham khảo) bằng code chấm ZETA, --score-only, CLI --dry-run."""

import argparse
import json
import os

import pytest

from ce_localization.utils.config import load_config
from vlm import run as vlm_run
from vlm.backends import Fatal, result
from vlm.prompt import prompt_hash
from vlm.runner import run_dataset, select
from tests.baseline.helpers import fake_ce130, fake_cocount

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vlm", "config", "openrouter", "gpt-6-luna.yaml")


class FakeBackend:
    """Trả `candidates` pixel canvas (3 box, score TĂNG dần ⇒ box cuối = confidence cao nhất) cho mọi ảnh, trừ ảnh trong `garbage` (không box) / `errors` (lỗi API); cost cố định / lần gọi."""

    def __init__(self, garbage=(), errors=(), cost=0.001, fatal=False):
        self.garbage, self.errors, self.cost, self.fatal = set(garbage), set(errors), cost, fatal
        self.calls = []

    def batch(self, jobs):
        if self.fatal:
            raise Fatal("402 hết tiền")
        out = []
        for j in jobs:
            self.calls.append((j["image_id"], j["temperature"]))
            if j["image_id"] in self.errors or "*" in self.errors:
                out.append(result(error="HTTP 503", sec=0.1))
            elif j["image_id"] in self.garbage:
                out.append(result("sorry, no idea", 100, 5, 0, self.cost, 0.1))
            else:
                text = json.dumps({"candidates": [{"name": "x", "box": [10 * i, 10 * i, 10 * i + 20, 10 * i + 30], "score": 0.5 + i / 10}
                                                  for i in range(3)]})
                out.append(result(text, 1000, 80, 10, self.cost, 0.2))
        return out


@pytest.fixture
def env(tmp_path):
    _, samples_root, ti = fake_ce130(str(tmp_path / "data"))
    croot = str(tmp_path / "cocount")
    review = fake_cocount(croot)
    cfg = load_config(CFG)
    cfg.update(k=5, chunk=2, icl=0, _path=CFG)                     # few-shot tác giả (ảnh thật) test riêng ở test_parse_prompt

    def ns(dataset="ce130"):
        return argparse.Namespace(dataset=dataset, split="test_all", turn_index=ti, samples_root=samples_root, cocount_root=croot,
                                  cocount_review=review, cocount_min_obj=5, obj_size=["mean", "exemplar"], seed=0, limit=None, out=None)
    return cfg, ns, tmp_path / "out"


def _lines(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def test_pilot_then_all_reuses_answers_and_scores(env):
    cfg, ns, out = env
    ph = prompt_hash(cfg)
    be = FakeBackend()
    d = str(out / "ce130")
    info, spent = run_dataset(cfg, ns(), be, d, "pilot2", ph, pilot=2, log=lambda *a: None)
    first = [c[0] for c in be.calls]
    assert len(first) == 2 and spent == pytest.approx(0.002)
    assert sorted(os.listdir(d)) == ["pilot2.json", "pilot2.jsonl", "responses.jsonl"]
    rows = _lines(os.path.join(d, "pilot2.jsonl"))
    resp = {r["image_id"]: r for r in _lines(os.path.join(d, "responses.jsonl"))}
    assert len(rows) == 2
    for r in rows:                                                         # 3 box (không lặp vòng), box[0] = score cao nhất (cuối model trả)
        W, H = resp[r["image_id"]]["canvas_wh"]
        sx, sy = r["orig_wh"][0] / W, r["orig_wh"][1] / H
        b = r["boxes"]
        assert r["n_valid"] == 3 and [c["score"] for c in b] == pytest.approx([0.7, 0.6, 0.5]) and b[0]["name"] == "x"
        assert b[0]["box"] == pytest.approx([20 * sx, 20 * sy, 40 * sx, 50 * sy]) and len(b[0]["box_512"]) == 4
        if "cnll" in b[0]:
            assert r["cnll_top"] == b[0]["cnll"] and r["cnll_min"] == min(c["cnll"] for c in b)
        assert r["text"] == resp[r["image_id"]]["text"] and r["cost"] == pytest.approx(0.001)
    with open(os.path.join(d, "pilot2.json")) as f:
        js = json.load(f)
    assert js["main"] == "top_score" and js["n_noscore"] == 0 and js["selected"]["n_short"] == 2 and js["selected"]["n_fail"] == 0
    assert js["top_score"]["K"] == 1 and js["min_cnll_5"]["K"] == 5 and "cnll" in js["top_score"]["results"]["inpainted"]
    top = [r["cnll_top"] for r in rows if r["cnll_top"] is not None]
    if top:
        assert js["top_score"]["results"]["inpainted"]["cnll"] == pytest.approx(sum(top) / len(top), abs=1e-3)
    assert info["full_estimate"]["est_full_cost"] == pytest.approx(0.001 * 4)

    run_dataset(cfg, ns(), be, d, "test", ph, log=lambda *a: None)
    assert len(be.calls) == 4 and not set(first) & {c[0] for c in be.calls[2:]}             # all chỉ hỏi 2 ảnh còn lại
    run_dataset(cfg, ns(), be, d, "test", ph, log=lambda *a: None)
    assert len(be.calls) == 4                                                              # resume: không hỏi lại
    cfg2 = {**cfg, "sampling": {"temperature": 0.1}}
    run_dataset(cfg2, ns(), be, d, "test", prompt_hash(cfg2), log=lambda *a: None)
    assert len(be.calls) == 8                                                              # đổi sampling -> hỏi lại hết


def test_pilot_selection_stable_and_subset():
    samples = [{"image_id": f"im{i}"} for i in range(50)]
    a, b = select(samples, 0, 10), select(list(reversed(samples)), 0, 10)
    assert {s["image_id"] for s in a} == {s["image_id"] for s in b} and len(a) == 10
    assert [s["image_id"] for s in a] == [s["image_id"] for s in samples if s in a]          # giữ thứ tự gốc
    assert select(samples, 0, 20)[:0] == [] and {s["image_id"] for s in a} <= {s["image_id"] for s in select(samples, 0, 20)}
    assert select(samples, 0, None) == samples


def test_retry_then_full_image_fallback(env):
    cfg, ns, out = env
    d = str(out / "cc")
    be = FakeBackend(garbage={"INTRA_FOO_TOM1_TOM2_00008_00006_0_200_positive"})
    run_dataset(cfg, ns("cocount"), be, d, "test", prompt_hash(cfg), log=lambda *a: None)
    bad = [c for c in be.calls if c[0] == "INTRA_FOO_TOM1_TOM2_00008_00006_0_200_positive"]
    assert [t for _, t in bad] == [None, 0.7, 0.7]                                          # 1 + retries (2), lần thử lại nhiệt 0.7
    rec = {r["image_id"]: r for r in _lines(os.path.join(d, "responses.jsonl"))}
    assert rec[bad[0][0]]["n_valid"] == 0 and rec[bad[0][0]]["attempts"] == 3 and rec[bad[0][0]]["cost"] == pytest.approx(0.003)
    rows = {r["image_id"]: r for r in _lines(os.path.join(d, "test.jsonl"))}
    W, H = rec[bad[0][0]]["orig_wh"]
    assert rows[bad[0][0]]["n_valid"] == 0 and rows[bad[0][0]]["boxes"][0]["box"] == [0, 0, W, H]
    assert rows[bad[0][0]]["boxes"][0]["fallback"] == "full_image" and len(rows[bad[0][0]]["boxes"]) == 1
    with open(os.path.join(d, "test.json")) as f:
        assert json.load(f)["selected"]["n_fail"] == 1


def test_api_errors_not_saved_and_stop(env):
    cfg, ns, out = env
    d = str(out / "err")
    logs = []
    info, _ = run_dataset({**cfg, "max_api_errors": 2}, ns(), FakeBackend(errors={"*"}), d, "test", prompt_hash(cfg), log=logs.append)
    assert _lines(os.path.join(d, "responses.jsonl")) == [] and "lỗi API liên tiếp" in info["this_run"]["stopped"]
    assert not os.path.exists(os.path.join(d, "test.jsonl")) and any("chưa chấm" in m for m in logs)
    info, _ = run_dataset(cfg, ns(), FakeBackend(fatal=True), d, "test", prompt_hash(cfg), log=lambda *a: None)
    assert "402" in info["this_run"]["stopped"]


def test_max_cost_stops_and_score_only(env):
    cfg, ns, out = env
    d = str(out / "cost")
    ph = prompt_hash(cfg)
    info, spent = run_dataset(cfg, ns(), FakeBackend(cost=1.0), d, "test", ph, max_cost=1.5, log=lambda *a: None)
    assert spent == pytest.approx(2.0) and len(_lines(os.path.join(d, "responses.jsonl"))) == 2   # dừng sau khối đầu (chunk 2)
    assert "max-cost" in info["this_run"]["stopped"]
    run_dataset(cfg, ns(), None, d, "test", ph, log=lambda *a: None)                         # score-only còn thiếu -> không chấm
    assert not os.path.exists(os.path.join(d, "test.jsonl"))
    run_dataset(cfg, ns(), FakeBackend(), d, "test", ph, score=False, log=lambda *a: None)
    with open(os.path.join(d, "test.json")) as f:                                            # chưa chấm: chỉ tiền / token
        assert "top_score" not in json.load(f)
    run_dataset(cfg, ns(), None, d, "test", ph, log=lambda *a: None)
    with open(os.path.join(d, "test.json")) as f:
        assert "top_score" in json.load(f) and os.path.exists(os.path.join(d, "test.jsonl"))


def test_cli_dry_run_needs_no_key(env, capsys, monkeypatch):
    _, ns, out = env
    n = ns()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    vlm_run.main(["--config", CFG, "--pilot", "2", "--dataset", "ce130", "--dry-run", "--turn-index", n.turn_index, "--samples-root",
                  n.samples_root, "--out-root", str(out)])
    txt = capsys.readouterr().out
    assert "--- SYSTEM" in txt and "Canvas size: " in txt and "Object(s) to insert: cup" in txt
    assert "Output ONLY JSON matching the schema.\n" in txt and "Schema:" not in txt                    # schema qua API
    if os.path.exists(os.path.join(os.path.dirname(CFG), "..", "..", "..", "data", "all_phase2_V2")):   # ảnh ví dụ tác giả có ở máy này
        assert "few-shot 6 lượt" in txt and "Your seashell boxes are too small" in txt
    assert not os.path.exists(out)


def test_data_parallel_shards_disjoint_merge_then_score(env):
    """`vllm.data_parallel`: 2 bản chia ảnh theo hash (không trùng, không sót), mỗi bản ghi file riêng; bản chưa gộp vẫn được tính
    là đã xong (resume); gộp rồi chấm như chạy một bản."""
    from baseline.score import load_data
    from vlm.runner import in_shard, merge_shards
    cfg, ns, out = env
    ph = prompt_hash(cfg)
    d = str(out / "dp")
    ids = [s["image_id"] for s in load_data(ns())[0]]
    assert sorted(i for k in (0, 1) for i in ids if in_shard(i, (k, 2))) == sorted(ids)
    calls = []
    for k in (0, 1):
        be = FakeBackend()
        run_dataset(cfg, ns(), be, d, f"test_shard{k}", ph, score=False, shard=(k, 2), log=lambda *a: None)
        calls.append({c[0] for c in be.calls})
        assert all(in_shard(i, (k, 2)) for i in calls[k])
    assert not calls[0] & calls[1] and calls[0] | calls[1] == set(ids)
    be = FakeBackend()
    run_dataset(cfg, ns(), be, d, "test", ph, score=False, log=lambda *a: None)       # shard chưa gộp vẫn tính là xong
    assert not be.calls
    assert merge_shards(d) == len(ids) and sorted(os.listdir(d)) == sorted(["responses.jsonl", "test.json", "test_shard0.json",
                                                                               "test_shard1.json"])
    info, _ = run_dataset(cfg, ns(), None, d, "test", ph, log=lambda *a: None)
    assert info["selected"]["n"] == len(ids) and os.path.exists(os.path.join(d, "test.jsonl"))


def test_launch_shards_one_gpu_each_and_merge(env, monkeypatch):
    """Tiến trình cha: n con `--shard i/n`, CUDA_VISIBLE_DEVICES thứ i; con xong ⇒ gộp `responses_shard*.jsonl`."""
    cfg, ns, out = env
    d = out / "root" / cfg["name"] / "ce130_test_all"
    d.mkdir(parents=True)
    seen = []

    class P:
        def __init__(self, cmd, env):
            i = int(cmd[cmd.index("--shard") + 1].split("/")[0])
            seen.append((cmd[cmd.index("--shard") + 1], env["CUDA_VISIBLE_DEVICES"]))
            (d / f"responses_shard{i}.jsonl").write_text(json.dumps({"image_id": f"x{i}", "prompt_hash": "h"}) + "\n")
            self.returncode = 0

        def poll(self):
            return 0
    monkeypatch.setattr(vlm_run.subprocess, "Popen", P)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    a = vlm_run.parse_args(["--config", CFG, "--all", "--dataset", "ce130", "--out-root", str(out / "root")])
    assert vlm_run.launch_shards(cfg, a, ["--config", CFG], ["ce130"], 2, stagger=0) == 0
    assert seen == [("0/2", "0"), ("1/2", "1")] and sorted(os.listdir(d)) == ["responses.jsonl"]
    assert [r["image_id"] for r in _lines(d / "responses.jsonl")] == ["x0", "x1"]


def test_batch_mode_first_small_then_budget_capped(env):
    """Backend Batch: lô đầu `batch.first` ảnh (đo giá), sau đó lô theo `batch.chunk` nhưng không vượt `--max-cost` ước trước khi
    gửi; state_dir = <out_dir>/batch; mỗi job mang `attempt`."""
    cfg, ns, out = env
    cfg = {**cfg, "batch": {"first": 2, "chunk": 100}}
    d = str(out / "batch")

    class B(FakeBackend):
        is_batch = True

        def __init__(self, **kw):
            super().__init__(**kw)
            self.sizes, self.state = [], None

        def set_state_dir(self, p):
            self.state = p

        def batch(self, jobs):
            assert all("attempt" in j for j in jobs)
            self.sizes.append(len(jobs))
            return super().batch(jobs)
    be = B(cost=0.5)
    info, spent = run_dataset(cfg, ns(), be, d, "test", prompt_hash(cfg), max_cost=2.0, log=lambda *a: None)
    assert be.state == os.path.join(d, "batch") and be.sizes == [2, 2] and spent == pytest.approx(2.0)   # 2 × $0,5 rồi ($2 − $1) / $0,5
    assert "max-cost" in info["this_run"]["stopped"]
