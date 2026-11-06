"""Backend VLM: `make_backend(cfg, log)` theo `cfg["backend"]` (openai | openrouter | vllm; import lười — venv chính không cần
vllm). `cfg["fallback"]` = backend thử tiếp theo thứ tự (`Chain`): key trống / None / sai / hết tiền (`Fatal` hay `SystemExit` lúc
dựng, `Fatal` giữa chừng) -> chuyển sang cái sau, job đang dở gửi lại ở cái mới; hết danh sách -> `Fatal`.
Giao diện chung: `backend.batch(jobs) -> [Result]`, job = {"image_id", "messages", "temperature" (None = theo config), "attempt"};
Result = dict
{"text", "in", "out" (gồm token suy nghĩ), "reasoning", "cost" (USD, None = tính từ giá config), "sec" (cả lần gọi, gồm chờ thử lại
HTTP), "latency" (chỉ request thành công cuối: gửi -> nhận đủ; None nếu không đo được từng ảnh), "error"}. Lỗi không cứu được
(sai key, hết tiền, model không tồn tại) -> `Fatal`: dừng cả job, câu trả lời đã có vẫn còn trong responses.jsonl."""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

__all__ = ["Fatal", "Chain", "make_backend", "result", "backoff", "token_cost", "RETRY_STATUS", "FATAL_STATUS"]

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403, 404}


class Fatal(RuntimeError):
    pass


def result(text="", n_in=0, n_out=0, reasoning=0, cost=None, sec=0.0, error=None, latency=None, **extra):
    return {"text": text or "", "in": int(n_in or 0), "out": int(n_out or 0), "reasoning": int(reasoning or 0), "cost": cost,
            "sec": float(sec), "latency": None if latency is None else float(latency), "error": error, **extra}


def token_cost(res, price):
    """USD theo token × giá USD / 1M (`in`, `out`, `cached` = giá token vào trúng cache; thiếu thì tính như `in`)."""
    p = price or {}
    cached = min(int(res.get("cached") or 0), int(res["in"]))
    return ((res["in"] - cached) * float(p.get("in", 0)) + cached * float(p.get("cached", p.get("in", 0)))
            + res["out"] * float(p.get("out", 0))) / 1e6


def backoff(attempt, retry_after=None, cap=60.0):
    """Giây chờ trước lần thử lại thứ `attempt` (0, 1, …): `Retry-After` nếu có, không thì mũ 2 + jitter, tối đa `cap`."""
    if retry_after is not None:
        try:
            return min(cap, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(cap, 2.0 ** attempt + random.random())


def _make_one(kind, cfg, log, sleep):
    if kind == "openai_batch":
        from vlm.backends.batch import BatchBackend, OpenAIBatchAPI
        sync = _make_one("openai", cfg, log, sleep)
        return BatchBackend(sync, OpenAIBatchAPI(sync, sleep), cfg, log=log, sleep=sleep)
    if kind == "openai":
        from vlm.backends.openai_direct import OpenAI
        return OpenAI(cfg, log=log, sleep=sleep)
    if kind == "openrouter":
        from vlm.backends.openrouter import OpenRouter
        return OpenRouter(cfg, log=log, sleep=sleep)
    if kind == "vllm":
        from vlm.backends.vllm_local import VLLMLocal
        return VLLMLocal(cfg, log=log)
    raise ValueError(f"backend {kind!r} không thuộc openai | openrouter | vllm")


class Chain:
    """Thử lần lượt `kinds`; `kind` = backend đang dùng (ghi vào log)."""

    def __init__(self, kinds, cfg, log=print, sleep=time.sleep, make=_make_one):
        self.kinds, self.cfg, self.log, self.sleep, self.make = list(kinds), cfg, log, sleep, make
        self._lock, self._i, self.cur, self.kind, self.state_dir = threading.Lock(), -1, None, None, None
        self._advance("bắt đầu")

    def _advance(self, why):
        while True:
            self._i += 1
            if self._i >= len(self.kinds):
                raise Fatal(f"hết backend để thử ({' -> '.join(self.kinds)}); lỗi cuối: {why}")
            kind = self.kinds[self._i]
            try:
                self.cur, self.kind = self.make(kind, self.cfg, self.log, self.sleep), kind
                if self.state_dir and hasattr(self.cur, "set_state_dir"):
                    self.cur.set_state_dir(self.state_dir)
                self.log(f"[route] dùng {kind} ({self._i + 1} / {len(self.kinds)}: {' -> '.join(self.kinds)})")
                return
            except (Fatal, SystemExit) as e:
                why = f"{kind}: {e}"
                self.log(f"[route] bỏ {kind}: {e}")

    @property
    def concurrency(self):
        return self.cur.concurrency

    @property
    def is_batch(self):
        return getattr(self.cur, "is_batch", False)

    def set_state_dir(self, d):
        self.state_dir = d
        if hasattr(self.cur, "set_state_dir"):
            self.cur.set_state_dir(d)

    def call(self, job):
        while True:
            cur = self.cur
            try:
                return cur.call(job)
            except Fatal as e:
                with self._lock:
                    if self.cur is cur:                      # luồng đầu gặp lỗi chuyển; luồng khác chỉ gửi lại ở backend mới
                        self.log(f"[route] {self.kind} lỗi không cứu được -> chuyển: {e}")
                        self._advance(f"{self.kind}: {e}")

    def batch(self, jobs):
        cur = self.cur
        if getattr(cur, "is_batch", False):          # Batch API: cả lô một lần; lỗi không cứu được -> backend sau, gửi lại cả lô
            try:
                return cur.batch(jobs)
            except Fatal as e:
                with self._lock:
                    if self.cur is cur:
                        self.log(f"[route] {self.kind} lỗi không cứu được -> chuyển: {e}")
                        self._advance(f"{self.kind}: {e}")
                return self.batch(jobs)
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self.call, jobs))


def make_backend(cfg, log=print, sleep=time.sleep, batch=False):
    """`batch` (run.py --batch): Batch API của backend đầu (chỉ openai), BỎ `fallback` — đường dự phòng là gọi thường giá gấp đôi,
    không tự chuyển sang; key lỗi thì dừng (chạy lại không --batch nếu muốn đi OpenRouter)."""
    kinds = [cfg["backend"], *(cfg.get("fallback") or [])]
    if batch:
        if kinds[0] != "openai":
            raise Fatal(f"--batch chỉ cho backend openai (config: {kinds[0]}; OpenRouter / vLLM không có Batch API)")
        if len(kinds) > 1:
            log(f"[route] --batch: bỏ dự phòng {kinds[1:]} (gọi thường = giá gấp đôi) — key {kinds[0]} lỗi thì dừng")
        kinds = [kinds[0] + "_batch"]
    return Chain(kinds, cfg, log, sleep) if len(kinds) > 1 else _make_one(kinds[0], cfg, log, sleep)
