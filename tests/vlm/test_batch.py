"""`run.py --batch` với HTTP giả (không mạng): OpenAI Batch API — file JSONL đúng dạng, chờ tới trạng thái cuối, khớp kết quả
theo khoá (thứ tự dòng ra khác dòng vào), tiền theo `price_batch`, request hết hạn gửi lại, CHẠY LẠI KHÔNG GỬI TRÙNG (dùng lô trong
state.json), chặn `--max-cost` trước khi gửi."""

import json

import pytest

from vlm.backends import Fatal, make_backend
from vlm.backends.batch import BatchBackend, OpenAIBatchAPI
from vlm.runner import batch_size


class R:
    def __init__(self, status, body=None, text=None, headers=None):
        self.status_code, self._b, self.headers = status, body, headers or {}
        self.text = text if text is not None else json.dumps(body)

    def json(self):
        return self._b if self._b is not None else json.loads(self.text)


class Fake:
    """Session giả: `route(method, url, kw) -> R`; ghi mọi lời gọi."""

    def __init__(self, route):
        self.route, self.calls = route, []

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self.route(method, url, kw)

    def get(self, url, headers=None, timeout=None, **kw):
        return self.request("GET", url, headers=headers, **kw)

    def post(self, url, headers=None, json=None, timeout=None, **kw):
        return self.request("POST", url, headers=headers, json=json, **kw)


JOBS = [{"image_id": f"im{i}", "temperature": None, "attempt": 0,
         "messages": [{"role": "user", "text": "hello", "image": "QUJD"}]} for i in range(3)]
CFG_OA = {"name": "gpt-6-luna", "model": "openai/gpt-6-luna", "k": 30, "max_tokens": 100, "reasoning": "none", "sampling": {},
          "retries": 0, "price": {"in": 0.1, "out": 0.5, "cached": 0.01}, "price_batch": {"in": 0.05, "out": 0.25, "cached": 0.005},
          "openai": {}, "batch": {"poll_sec": 1, "limit_wait_sec": 5}}


def chat(text="{}", n_in=1000, n_out=200, cached=400):
    return {"id": "c", "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out, "prompt_tokens_details": {"cached_tokens": cached}}}


def openai_server(state):
    """OpenAI giả: lô thứ i trả `state["outputs"][i]` (dòng JSONL) sau 1 lần poll `in_progress`."""
    def route(method, url, kw):
        if url.endswith("/models/gpt-6-luna"):
            return R(200, {"id": "gpt-6-luna"})
        if url.endswith("/chat/completions"):
            return R(200, chat())                                              # thăm dò
        if url.endswith("/files") and method == "POST":
            state["files"].append(kw["files"]["file"][1].decode())
            return R(200, {"id": f"file-{len(state['files'])}"})
        if url.endswith("/batches") and method == "POST":
            bid = f"batch_{len(state['batches'])}"
            state["batches"].append(kw["json"])
            state["polls"][bid] = 0
            return R(200, {"id": bid})
        if "/batches/" in url:
            bid = url.rsplit("/", 1)[-1]
            state["polls"][bid] += 1
            st = "in_progress" if state["polls"][bid] == 1 else "completed"
            return R(200, {"id": bid, "status": st, "request_counts": {"total": 3, "completed": 2, "failed": 1},
                           "output_file_id": f"out-{bid}" if st == "completed" else None})
        if "/files/out-batch_" in url:
            i = int(url.split("out-batch_")[1].split("/")[0])
            return R(200, text="".join(json.dumps(x) + "\n" for x in state["outputs"][i]))
        raise AssertionError(url)
    return route


def ok_line(k, **kw):
    return {"custom_id": k, "response": {"status_code": 200, "body": chat(**kw)}, "error": None}


def _oa_backend(monkeypatch, tmp_path, state):
    from vlm.backends.openai_direct import OpenAI
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-test-abcdef123456")
    sleeps, logs = [], []
    sync = OpenAI(CFG_OA, log=logs.append, sleep=sleeps.append, session=Fake(openai_server(state)))
    be = BatchBackend(sync, OpenAIBatchAPI(sync, sleeps.append), CFG_OA, log=logs.append, sleep=sleeps.append)
    be.set_state_dir(str(tmp_path / "batch"))
    return be, sleeps, logs


def test_openai_batch_flow_cost_order_and_expired_resubmit(monkeypatch, tmp_path):
    expired = {"custom_id": "im1#0", "response": None, "error": {"code": "batch_expired", "message": "window expired"}}
    st = {"files": [], "batches": [], "polls": {},
          "outputs": [[ok_line("im2#0", text='{"a": 2}'), expired, ok_line("im0#0", text='{"a": 0}')],   # thứ tự khác dòng vào
                      [ok_line("im1#0", text='{"a": 1}')]]}
    be, sleeps, logs = _oa_backend(monkeypatch, tmp_path, st)
    out = be.batch(JOBS)
    assert [o["text"] for o in out] == ['{"a": 0}', '{"a": 1}', '{"a": 2}']                 # khớp theo custom_id
    lines = [json.loads(x) for x in st["files"][0].splitlines()]
    assert [x["custom_id"] for x in lines] == ["im0#0", "im1#0", "im2#0"] and lines[0]["url"] == "/v1/chat/completions"
    body = lines[0]["body"]
    assert body["model"] == "gpt-6-luna" and body["response_format"]["json_schema"]["name"] == "insertion_layout"
    assert body["messages"][0]["content"][1]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"
    assert st["batches"][0]["completion_window"] == "24h" and len(st["batches"]) == 2                 # hết hạn -> gửi lại 1 request
    assert [json.loads(x)["custom_id"] for x in st["files"][1].splitlines()] == ["im1#0"]
    assert out[0]["cost"] == pytest.approx((600 * 0.05 + 400 * 0.005 + 200 * 0.25) / 1e6)           # giá BATCH, có cache
    assert out[0]["latency"] is None and out[0]["provider"] == "OpenAI Batch"
    state = json.load(open(tmp_path / "batch" / "state.json"))
    assert [p["id"] for p in state["parts"]] == ["batch_0", "batch_1"] and (tmp_path / "batch" / "batch_0.jsonl").exists()


def test_rerun_reuses_submitted_batch_no_double_submit(monkeypatch, tmp_path):
    """Bị ngắt khi đang chờ: chạy lại cùng ảnh -> poll lô cũ trong state.json, KHÔNG tạo lô mới (không trả tiền 2 lần)."""
    st = {"files": [], "batches": [], "polls": {}, "outputs": [[ok_line(f"im{i}#0") for i in range(3)]]}
    be, _, _ = _oa_backend(monkeypatch, tmp_path, st)

    class Stop(Exception):
        pass

    def boom(sec):
        raise Stop()
    be.sleep = boom                                        # ngắt ở lần chờ poll đầu tiên
    with pytest.raises(Stop):
        be.batch(JOBS)
    assert len(st["batches"]) == 1
    be2, _, logs = _oa_backend(monkeypatch, tmp_path, st)
    out = be2.batch(JOBS)
    assert len(st["batches"]) == 1 and all(o["text"] == "{}" for o in out) and any("dùng lại lô" in m for m in logs)
    assert len(be2.batch(JOBS)) == 3 and len(st["batches"]) == 1                                     # file kết quả đã tải: đọc lại
    retry = [{**JOBS[0], "attempt": 1, "temperature": 0.7}]                                          # lần thử lại = request mới
    st["outputs"].append([ok_line("im0#1")])
    be2.batch(retry)
    assert len(st["batches"]) == 2


def test_batch_needs_price_and_supported_backend(monkeypatch):
    with pytest.raises(Fatal, match="chỉ cho backend openai"):
        make_backend({"backend": "openrouter", "model": "x"}, log=lambda *a: None, batch=True)

    class S:
        cfg = {}
    with pytest.raises(Fatal, match="thiếu `price_batch"):
        BatchBackend(S(), OpenAIBatchAPI.__new__(OpenAIBatchAPI), {"name": "x", "model": "m"}, log=lambda *a: None)


def test_batch_size_caps_by_budget_before_submit():
    cfg = {"batch": {"chunk": 2000, "first": 20}}
    assert batch_size(cfg, {}, None, 0.0, 5000)[0] == 20                                             # chưa có giá: lô đầu nhỏ
    done = {f"i{i}": {"cost": 0.01} for i in range(20)}
    assert batch_size(cfg, done, None, 0.0, 5000)[0] == 2000
    assert batch_size(cfg, done, 5.0, 1.0, 5000)[0] == 400                                           # ($5 - $1) / $0,01
    cfgp = {**cfg, "price": {"in": 1.0, "out": 5.0}, "price_batch": {"in": 0.5, "out": 2.5}}
    assert batch_size(cfgp, done, 5.0, 1.0, 5000)[0] == 800                                          # câu gọi thường quy giá batch (½)
    assert batch_size(cfgp, {k: {**v, "batch": True} for k, v in done.items()}, 5.0, 1.0, 5000)[0] == 400
    n, why = batch_size(cfg, done, 5.0, 5.0, 5000)
    assert n == 0 and "max-cost" in why


def test_batch_drops_fallback(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "None")
    logs = []
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):                   # không Chain sang OpenRouter: dừng
        make_backend({"backend": "openai", "fallback": ["openrouter"], "model": "openai/x", "price_batch": {"in": 1, "out": 1}},
                     log=logs.append, batch=True)
    assert any("bỏ dự phòng" in m for m in logs)
