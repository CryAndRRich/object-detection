"""ZETA5 / ZETA6 backend với HTTP giả (không mạng): OpenRouter payload (reasoning / provider / ảnh data URL / schema tác giả + require_parameters,
bỏ tham số model không nhận, model không có structured_outputs = Fatal),
đọc usage.cost + token suy nghĩ, thử lại 429 theo Retry-After, 402 = Fatal; Gemini payload (inlineData, thinkingLevel, key ở header),
bỏ phần thought, token suy nghĩ cộng vào token ra, giãn RPM; `.env` (env var ưu tiên, không in giá trị); bộ nhớ vLLM."""

import json

import pytest

from vlm.backends import Fatal
from vlm.backends.gemini import Gemini
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


def test_gemini_payload_thoughts_and_rpm(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-abcdefgh")
    body = {"candidates": [{"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": '[{"y": 1}]'}]},
                            "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 900, "candidatesTokenCount": 200, "thoughtsTokenCount": 400}}
    sess = Session([Resp(200, body), Resp(200, body)])
    now = [0.0]
    sleeps = []
    cfg = {"model": "gemini-robotics-er-2-preview", "k": 30, "max_tokens": 50, "sampling": {"temperature": 1.0},
           "gemini": {"thinking_level": "medium", "rpm": 30, "concurrency": 1}}
    be = Gemini(cfg, log=lambda *a: None, sleep=sleeps.append, session=sess, clock=lambda: now[0])
    job = {**JOB, "messages": [{"role": "system", "text": "SYS", "image": None}, {**JOB["messages"][0], "mime": "image/png"}]}
    r1, r2 = be.batch([job, job])
    assert r1["text"] == '[{"y": 1}]' and r1["in"] == 900 and r1["out"] == 600 and r1["reasoning"] == 400 and r1["cost"] is None
    assert sleeps == [2.0]                                                                   # 30 RPM -> cách nhau 2 s
    url, hdr, b = sess.sent[0]
    assert url.endswith("gemini-robotics-er-2-preview:generateContent") and "key=" not in url and hdr["x-goog-api-key"]
    assert b["contents"][0]["parts"][1] == {"inlineData": {"mimeType": "image/png", "data": "QUJD"}} and len(b["contents"]) == 1
    assert b["systemInstruction"] == {"parts": [{"text": "SYS"}]}
    assert b["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "medium"} and b["generationConfig"]["temperature"] == 1.0
    assert b["generationConfig"]["responseMimeType"] == "application/json"
    assert b["generationConfig"]["responseJsonSchema"]["properties"]["candidates"]["maxItems"] == 30          # schema tác giả qua API
    sess.posts = [Resp(400, {"error": {"message": "Invalid JSON payload: responseJsonSchema ..."}})]
    with pytest.raises(Fatal, match="400 với responseJsonSchema"):
        be.batch([job])


def test_env_loader(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("# c\nOPENROUTER_API_KEY='sk-file-abcdef'\nexport GEMINI_API_KEY=g-file-123456\nEMPTY=\n")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-zzzzzz")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("EMPTY", raising=False)
    assert load_env(str(p)) == ["OPENROUTER_API_KEY", "GEMINI_API_KEY", "EMPTY"]
    assert require_key("OPENROUTER_API_KEY") == "sk-env-zzzzzz" and require_key("GEMINI_API_KEY") == "g-file-123456"
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


class OASession(Session):
    """OpenAI giả: GET /models/<m> -> `model_status`; POST lần lượt `posts` (lần đầu = request thăm dò)."""

    def __init__(self, posts, model_status=200):
        super().__init__(posts)
        self.model_status = model_status

    def get(self, url, headers=None, timeout=None):
        return Resp(self.model_status, {"id": url.rsplit("/", 1)[-1]})


OA_OK = {"id": "chatcmpl-1", "choices": [{"message": {"content": '{"candidates": []}'}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "prompt_tokens_details": {"cached_tokens": 600},
                   "completion_tokens_details": {"reasoning_tokens": 0}}}
QUOTA = Resp(429, {"error": {"code": "insufficient_quota", "message": "You exceeded your current quota"}})


def _oa(monkeypatch, posts, model_status=200, key="sk-proj-test-abcdef123456"):
    from vlm.backends.openai_direct import OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", key)
    cfg = {"model": "openai/gpt-6-luna", "k": 30, "max_tokens": 100, "reasoning": "none", "sampling": {},
           "price": {"in": 0.1, "out": 0.5, "cached": 0.025}, "openai": {"concurrency": 2}, "openrouter": {"detail": "original"}}
    sess, logs, sleeps = OASession(posts, model_status), [], []
    return OpenAI(cfg, log=logs.append, sleep=sleeps.append, session=sess), sess, logs, sleeps


def test_openai_payload_probe_drop_and_cost(monkeypatch):
    from vlm.runner import record_cost
    unsup = Resp(400, {"error": {"code": "unsupported_parameter", "param": "temperature", "message": "no temperature"}})
    be, sess, logs, _ = _oa(monkeypatch, [Resp(200, OA_OK), unsup, Resp(200, OA_OK)])
    assert len(sess.sent) == 1 and sess.sent[0][2]["max_completion_tokens"] == 16                  # thăm dò số dư lúc khởi động
    r = be.batch([{**JOB, "temperature": 0.7}])[0]
    body = sess.sent[-1][2]
    assert body["model"] == "gpt-6-luna" and body["reasoning_effort"] == "none" and "temperature" not in body
    assert "temperature" in sess.sent[1][2] and any("không nhận `temperature`" in m for m in logs)
    assert body["response_format"]["json_schema"]["name"] == "insertion_layout" and "max_tokens" not in body
    assert body["messages"][0]["content"][1]["image_url"]["detail"] == "original"                    # cùng detail OpenRouter
    assert r["cost"] is None and r["cached"] == 600 and r["provider"] == "OpenAI (trực tiếp)"
    assert record_cost(r, be.cfg) == pytest.approx((400 * 0.1 + 600 * 0.025 + 200 * 0.5) / 1e6)
    assert not any("abcdef123456" in m for m in logs)
    sess.posts.append(Resp(400, {"error": {"code": "unsupported_value", "param": "reasoning_effort"}}))
    with pytest.raises(Fatal, match="không nhận `reasoning_effort`"):                               # không bỏ: đổi câu trả lời
        be.batch([JOB])


@pytest.mark.parametrize("posts,status,key,match", [
    ([QUOTA], 200, "sk-proj-test-abcdef123456", "hết tiền"),
    ([], 401, "sk-proj-test-abcdef123456", "từ chối key"),
    ([], 404, "sk-proj-test-abcdef123456", "không dùng được model"),
    ([], 200, "sk-or-v1-abcdefghijkl", "là key OpenRouter")])
def test_openai_startup_fatal(monkeypatch, posts, status, key, match):
    with pytest.raises(Fatal, match=match):
        _oa(monkeypatch, posts, status, key)


def test_chain_skips_missing_key_and_switches_on_fatal(monkeypatch):
    """Key OpenAI `None` -> bỏ qua; OpenAI hết tiền giữa chừng -> job gửi lại ở OpenRouter; hết danh sách -> Fatal."""
    from vlm.backends import Chain, make_backend

    class B:
        concurrency = 2

        def __init__(self, kind, fail_after=None):
            self.kind, self.n, self.fail_after = kind, 0, fail_after

        def call(self, job):
            self.n += 1
            if self.fail_after is not None and self.n > self.fail_after:
                raise Fatal("429 insufficient_quota")
            return {"text": self.kind}

    built = []

    def make(kind, cfg, log, sleep):
        if kind == "none":
            from vlm.env import require_key
            require_key("ZETA_TEST_KEY")
        b = B(kind, fail_after=1 if kind == "openai" else None)
        built.append(b)
        return b
    monkeypatch.setenv("ZETA_TEST_KEY", "None")
    logs = []
    ch = Chain(["none", "openai", "openrouter"], {}, log=logs.append, make=make)
    assert ch.kind == "openai" and any("bỏ none" in m and "None" in m for m in logs)
    out = [ch.call(JOB) for _ in range(3)]
    assert [o["text"] for o in out] == ["openai", "openrouter", "openrouter"] and ch.kind == "openrouter"
    built[-1].fail_after = 0
    with pytest.raises(Fatal, match="hết backend"):
        ch.call(JOB)
    monkeypatch.setenv("OPENAI_API_KEY", "null")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(Fatal, match="hết backend.*OPENROUTER_API_KEY"):
        make_backend({"backend": "openai", "fallback": ["openrouter"], "model": "openai/x"}, log=lambda *a: None)


def test_placeholder_keys_are_missing(monkeypatch):
    from vlm.env import get_key
    for v in ("None", "null", " ", "<điền key>", "your_key", "sk-your-openai-key", "sk-proj-xxxxxxxx"):
        monkeypatch.setenv("K_TEST", v)
        assert get_key("K_TEST") is None
        with pytest.raises(SystemExit, match="trống / None"):
            require_key("K_TEST")
    monkeypatch.setenv("K_TEST", "sk-proj-real-123456")
    assert get_key("K_TEST") == "sk-proj-real-123456"


def test_hash_same_for_openai_and_openrouter_route():
    from vlm.prompt import prompt_hash
    cfg = {"backend": "openrouter", "model": "openai/gpt-6-luna", "k": 30, "icl": 0, "reasoning": "none",
           "openrouter": {"provider": {"data_collection": "deny"}}}
    assert prompt_hash({**cfg, "backend": "openai", "fallback": ["openrouter"], "openai": {"concurrency": 4}}) == prompt_hash(cfg)
    assert prompt_hash({**cfg, "backend": "gemini"}) != prompt_hash(cfg)


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


def test_gemini_daily_quota_is_fatal(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-abcdefgh")
    q = Resp(429, {"error": {"status": "RESOURCE_EXHAUSTED", "details": [{"violations": [
        {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
    be = Gemini({"model": "m", "k": 30, "max_tokens": 50, "gemini": {}}, log=lambda *a: None, sleep=lambda s: None, session=Session([q]))
    with pytest.raises(Fatal, match="hết quota ngày"):
        be.batch([JOB])
    monkeypatch.setenv("GEMINI_API_KEY", "None")
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):                                   # không có key -> dừng
        Gemini({"model": "m", "gemini": {}}, log=lambda *a: None)


def test_env_last_line_wins_and_export_shadow_reported(tmp_path, monkeypatch):
    from vlm.env import key_source
    p = tmp_path / ".env"
    p.write_text("OPENAI_API_KEY=sk-old-aaaaaaaa\nOPENAI_API_KEY=sk-new-bbbbbbbb\nGEMINI_API_KEY=g-file-123456\n")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-your-gemini-key")                             # export sẵn trong shell (chữ mẫu)
    load_env(str(p))
    assert require_key("OPENAI_API_KEY") == "sk-new-bbbbbbbb" and key_source("OPENAI_API_KEY") == ".env"
    assert "KHÁC giá trị trong .env" in key_source("GEMINI_API_KEY")


def test_gemini_minute_quota_waits_retry_delay(monkeypatch):
    """429 theo phút: chờ đúng `retryDelay` Google gợi ý (+1 s) rồi thử lại, không dừng."""
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test-abcdefgh")
    q = Resp(429, {"error": {"status": "RESOURCE_EXHAUSTED", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}, {"retryDelay": "37s"}]}})
    ok = Resp(200, {"candidates": [{"content": {"parts": [{"text": "{}"}]}}], "usageMetadata": {}})
    sleeps = []
    be = Gemini({"model": "m", "k": 30, "max_tokens": 50, "gemini": {}}, log=lambda *a: None, sleep=sleeps.append, session=Session([q, ok]))
    assert be.batch([JOB])[0]["text"] == "{}" and sleeps == [38.0]


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
