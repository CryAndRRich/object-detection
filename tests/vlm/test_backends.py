"""ZETA5 / ZETA6 backend với HTTP giả (không mạng): OpenRouter payload (reasoning / provider / ảnh data URL / schema tác giả + require_parameters,
bỏ tham số model không nhận, model không có structured_outputs = Fatal),
đọc usage.cost + token suy nghĩ, thử lại 429 theo Retry-After, 402 = Fatal; `.env` (env var ưu tiên, không in giá trị); bộ nhớ vLLM."""

import json

import pytest

from vlm.backends import Fatal
from vlm.backends.openrouter import OpenRouter
from vlm.backends.vllm_local import memory_utilization
from vlm.env import load_env, mask, require_key

JOB = {"image_id": "a", "temperature": None,
       "messages": [{"role": "user", "text": "hello", "image": "QUJD"}]}


class Resp:
    def __init__(self, status, body, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}
        self.text = json.dumps(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class Session:
    def __init__(self, posts, models=None):
        self.posts, self.sent, self.models = list(posts), [], models or []

    def get(self, url, headers=None, timeout=None):
        if url.endswith("/models"):
            return Resp(200, {"data": self.models})
        return Resp(200, {"data": {"usage": 1.25, "limit_remaining": 8.75}})

    def post(self, url, headers=None, json=None, timeout=None):
        self.sent.append((url, headers, json))
        return self.posts.pop(0)


ALL = ("structured_outputs", "temperature", "reasoning")


def _or(monkeypatch, posts, supported=ALL, **o):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-1234567890")
    cfg = {"model": "openai/gpt-6-luna", "k": 30, "max_tokens": 100, "reasoning": "none", "sampling": {"temperature": 0.2},
           "openrouter": {"concurrency": 2, "provider": {"data_collection": "deny"}, **o}}
    sess = Session(posts, [{"id": "openai/gpt-6-luna", "supported_parameters": list(supported), "pricing": {"prompt": "1e-7",
                                                                                                             "completion": "5e-7"}}])
    logs, sleeps = [], []
    return OpenRouter(cfg, log=logs.append, sleep=sleeps.append, session=sess), sess, logs, sleeps


OK = {"id": "gen-1", "choices": [{"message": {"content": '{"boxes": [[1,2,3,4]]}'}, "finish_reason": "stop"}],
      "usage": {"prompt_tokens": 1200, "completion_tokens": 300, "completion_tokens_details": {"reasoning_tokens": 120}, "cost": 0.00042}}


def test_openrouter_payload_usage_and_retry(monkeypatch):
    be, sess, logs, sleeps = _or(monkeypatch, [Resp(429, {"error": "rate"}, {"Retry-After": "3"}), Resp(200, OK)], detail="original")
    r = be.batch([JOB])[0]
    assert r["text"] == '{"boxes": [[1,2,3,4]]}' and r["cost"] == 0.00042 and r["in"] == 1200 and r["out"] == 300 and r["reasoning"] == 120
    assert sleeps == [3.0] and len(sess.sent) == 2
    url, hdr, body = sess.sent[-1]
    assert url.endswith("/chat/completions") and hdr["Authorization"] == "Bearer sk-or-test-1234567890"
    assert body["reasoning"] == {"effort": "none"} and body["temperature"] == 0.2
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True}
    img = body["messages"][0]["content"][1]["image_url"]
    assert body["messages"][0]["content"][0] == {"type": "text", "text": "hello"}                 # text trước ảnh (docs OpenRouter)
    assert img == {"url": "data:image/jpeg;base64,QUJD", "detail": "original"} and body["response_format"]["type"] == "json_schema"
    js = body["response_format"]["json_schema"]                                                # schema tác giả, maxItems = k
    assert js["name"] == "insertion_layout" and js["schema"]["properties"]["candidates"]["maxItems"] == 30
    assert not any("sk-or-test-1234567890" in m for m in logs) and any("…7890" in m for m in logs)


def test_latency_excludes_retry_wait(monkeypatch):
    import time
    be, _, _, _ = _or(monkeypatch, [Resp(503, {}), Resp(200, OK)])
    be.sleep = lambda sec: time.sleep(0.3)                                                  # chờ thử lại thật 0,3 s
    r = be.batch([JOB])[0]
    assert r["sec"] >= 0.3 and r["latency"] is not None and r["latency"] < 0.2 and r["gen_id"] == "gen-1"


def test_openrouter_unsupported_params_dropped_and_fatal(monkeypatch):
    be, sess, logs, _ = _or(monkeypatch, [Resp(200, OK)], supported=("structured_outputs", "reasoning"))   # Luna: không temperature
    be.batch([{**JOB, "temperature": 0.7}])
    assert "temperature" not in sess.sent[0][2] and any("không nhận `temperature`" in m for m in logs)
    with pytest.raises(Fatal, match="không có structured_outputs"):
        _or(monkeypatch, [], supported=("temperature",))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-proj-abcdefghijklmnop")                    # key OpenAI dán nhầm
    with pytest.raises(Fatal, match="không phải key OpenRouter"):
        OpenRouter({"model": "x", "openrouter": {}}, log=lambda *a: None, session=Session([], []))
    be, _, _, _ = _or(monkeypatch, [Resp(402, {"error": "credits"})])
    with pytest.raises(Fatal, match="402"):
        be.batch([JOB])
    be, sess, _, sleeps = _or(monkeypatch, [Resp(402, {"error": {"metadata": {"reason": "in_flight_budget_exhausted"}}}), Resp(200, OK)])
    assert be.batch([JOB])[0]["cost"] == 0.00042 and len(sleeps) == 1                        # 402 in-flight: chờ rồi thử lại
    be, _, _, sleeps = _or(monkeypatch, [Resp(503, {})] * 7)
    r = be.batch([JOB])[0]
    assert r["error"] and not r["text"] and len(sleeps) == 6
    with pytest.raises(Fatal, match="không có model"):
        OpenRouter({"model": "x/none", "openrouter": {}}, log=lambda *a: None, session=Session([], []))


def test_env_loader(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# c\nOPENROUTER_API_KEY='sk-file-abcdef'\nexport OTHER_API_KEY=g-file-123456\nEMPTY=\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-zzzzzz")
    monkeypatch.delenv("OTHER_API_KEY", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    assert load_env(str(p)) == ["OPENROUTER_API_KEY", "OTHER_API_KEY", "EMPTY"]
    assert require_key("OPENROUTER_API_KEY") == "sk-env-zzzzzz" and require_key("OTHER_API_KEY") == "g-file-123456"
    with pytest.raises(SystemExit, match="EMPTY") as e:
        require_key("EMPTY")
    assert "sk-" not in str(e.value) and mask("sk-env-zzzzzz") == "…zzzz" and load_env(str(tmp_path / "none")) == []
    monkeypatch.setenv("BAD_KEY", "sk-or-v1-abc def\u200bghij")                          # dán lỗi: dấu cách + ký tự vô hình
    with pytest.raises(SystemExit, match=r"vị trí \[\(12, 'U\+0020'\), \(16, 'U\+200B'\)\]") as e:
        require_key("BAD_KEY")
    assert "abc" not in str(e.value) and "biến môi trường" in str(e.value)


def test_vllm_memory_utilization():
    assert memory_utilization(10, 12 * 1024, 24 * 1024) == pytest.approx(10 / 24)
    with pytest.raises(RuntimeError, match="GPU trống 9.5 GiB < cần 10.0 GiB"):
        memory_utilization(10, 9.5 * 1024, 24 * 1024)


def test_vllm_free_memory_takes_tightest_visible_gpu(monkeypatch):
    import subprocess
    from vlm.backends import vllm_local
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="14000, 15360\n9000, 15360\n")
    monkeypatch.setattr(vllm_local.subprocess, "run", fake_run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert vllm_local.gpu_free_total_mib() == (9000.0, 15360.0) and seen["cmd"][2] == "0,1"     # tensor parallel: card trống ít nhất


def test_placeholder_keys_are_missing(monkeypatch):
    from vlm.env import get_key
    for v in ("None", "null", " ", "<điền key>", "your_key", "sk-your-openai-key", "sk-proj-xxxxxxxx"):
        monkeypatch.setenv("K_TEST", v)
        assert get_key("K_TEST") is None
        with pytest.raises(SystemExit, match="trống / None"):
            require_key("K_TEST")
    monkeypatch.setenv("K_TEST", "sk-proj-real-123456")
    assert get_key("K_TEST") == "sk-proj-real-123456"


def test_openrouter_zero_balance_fatal(monkeypatch):
    class S(Session):
        def get(self, url, headers=None, timeout=None):
            if url.endswith("/credits"):
                return Resp(200, {"data": {"total_credits": 5.0, "total_usage": 5.0}})
            return super().get(url, headers, timeout)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-1234567890")
    sess = S([], [{"id": "m/x", "supported_parameters": ["structured_outputs"], "pricing": {}}])
    with pytest.raises(Fatal, match="hết số dư"):
        OpenRouter({"model": "m/x", "k": 30, "max_tokens": 10, "openrouter": {}}, log=lambda *a: None, session=sess)


def test_env_last_line_wins_and_export_shadow_reported(tmp_path, monkeypatch):
    from vlm.env import key_source
    p = tmp_path / ".env"
    p.write_text("OPENROUTER_API_KEY=sk-old-aaaaaaaa\nOPENROUTER_API_KEY=sk-new-bbbbbbbb\nOTHER_API_KEY=g-file-123456\n")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_API_KEY", "sk-your-other-key")                             # export sẵn trong shell (chữ mẫu)
    load_env(str(p))
    assert require_key("OPENROUTER_API_KEY") == "sk-new-bbbbbbbb" and key_source("OPENROUTER_API_KEY") == ".env"
    assert "KHÁC giá trị trong .env" in key_source("OTHER_API_KEY")


def test_vllm_replan_greedy_plus_samples_joined(monkeypatch):
    """RePlan: mỗi ảnh 2 request cùng prompt (greedy + n sample, seed theo ảnh), ảnh trước text, không schema; text nối bằng REPLAN_SEP."""
    import sys
    import types
    from types import SimpleNamespace as NS
    from vlm.backends.vllm_local import VLLMLocal
    from vlm.parse import REPLAN_SEP

    class SP(dict):
        def __init__(self, **kw):
            super().__init__(kw)
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=SP))
    seen = {}

    def chat(convs, sampling_params, **kw):
        seen.update(convs=convs, params=sampling_params)
        out = lambda t: NS(text=t, token_ids=[0] * len(t), finish_reason="stop")
        return [NS(prompt_token_ids=[0] * 7, outputs=[out("G")]), NS(prompt_token_ids=[0] * 7, outputs=[out("s1"), out("s22")])]
    b = VLLMLocal.__new__(VLLMLocal)
    b.cfg = {"max_tokens": 2048, "sampling": {"temperature": 0.0}, "replan": {"samples": 2, "temperature": 1.0, "top_p": 0.99},
             "prompt": "replan"}
    b._so, b.llm = {}, NS(chat=chat)
    job = {"image_id": "x", "attempt": 0, "temperature": None,
           "messages": [{"role": "user", "text": "T", "image": "b64", "mime": "image/png", "image_first": True}]}
    (res,) = b.batch([job])
    assert res["text"] == REPLAN_SEP.join(["G", "s1", "s22"]) and res["in"] == 7 and res["out"] == 1 + 2 + 3
    g, s = seen["params"]
    assert g["temperature"] == 0.0 and "n" not in g and s["n"] == 2 and s["temperature"] == 1.0 and s["top_p"] == 0.99
    assert s["seed"] == b.sample_params(job)["seed"] != b.sample_params({**job, "attempt": 1})["seed"]
    assert [c["type"] for c in seen["convs"][0][0]["content"]] == ["image_url", "text"] and seen["convs"][0] is seen["convs"][1]
