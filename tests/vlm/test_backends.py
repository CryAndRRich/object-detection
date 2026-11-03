"""ZETA-VLM backend với HTTP giả (không mạng): OpenRouter payload (reasoning / provider / ảnh data URL / schema tác giả + require_parameters,
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


def _or(monkeypatch, posts, supported=ALL, schema="api", **o):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-1234567890")
    cfg = {"model": "openai/gpt-6-luna", "format": "json_pixel", "k": 30, "schema": schema, "max_tokens": 100, "reasoning": "none", "sampling": {"temperature": 0.2},
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


def test_openrouter_unsupported_params_dropped_schema_prompt_and_fatal(monkeypatch):
    be, sess, logs, _ = _or(monkeypatch, [Resp(200, OK)], supported=("structured_outputs", "reasoning"))   # Luna: không temperature
    be.batch([{**JOB, "temperature": 0.7}])
    assert "temperature" not in sess.sent[0][2] and any("không nhận `temperature`" in m for m in logs)
    be, sess, _, _ = _or(monkeypatch, [Resp(200, OK)], supported=("temperature",), schema="prompt")
    be.batch([JOB])
    assert "response_format" not in sess.sent[0][2] and sess.sent[0][2]["provider"] == {"data_collection": "deny"}
    with pytest.raises(Fatal, match="schema: prompt"):
        _or(monkeypatch, [], supported=("temperature",))
    be, _, _, _ = _or(monkeypatch, [Resp(402, {"error": "credits"})])
    with pytest.raises(Fatal, match="402"):
        be.batch([JOB])
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
    cfg = {"model": "gemini-robotics-er-2-preview", "format": "json_pixel", "k": 30, "max_tokens": 50, "sampling": {"temperature": 1.0},
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
    with pytest.raises(Fatal, match="schema: prompt"):
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


def test_vllm_memory_utilization():
    assert memory_utilization(10, 12 * 1024, 24 * 1024) == pytest.approx(10 / 24)
    with pytest.raises(RuntimeError, match="GPU trống 9.5 GiB < cần 10.0 GiB"):
        memory_utilization(10, 9.5 * 1024, 24 * 1024)
