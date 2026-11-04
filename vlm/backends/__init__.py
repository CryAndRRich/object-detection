"""Backend VLM: `make_backend(cfg, log)` theo `cfg["backend"]` (openai | openrouter | gemini | vllm; import lười — venv chính không cần
vllm). `cfg["fallback"]` = backend thử tiếp theo thứ tự (`Chain`): key trống / None / sai / hết tiền (`Fatal` hay `SystemExit` lúc
dựng, `Fatal` giữa chừng) -> chuyển sang cái sau, job đang dở gửi lại ở cái mới; hết danh sách -> `Fatal`.
Giao diện chung: `backend.batch(jobs) -> [Result]`, job = {"image_id", "messages", "temperature" (None = theo config)}; Result = dict
{"text", "in", "out" (gồm token suy nghĩ), "reasoning", "cost" (USD, None = tính từ giá config), "sec" (cả lần gọi, gồm chờ thử lại
HTTP), "latency" (chỉ request thành công cuối: gửi -> nhận đủ; None nếu không đo được từng ảnh), "error"}. Lỗi không cứu được
(sai key, hết tiền, model không tồn tại) -> `Fatal`: dừng cả job, câu trả lời đã có vẫn còn trong responses.jsonl."""

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

__all__ = ["Fatal", "Chain", "make_backend", "result", "backoff", "RETRY_STATUS", "FATAL_STATUS"]

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403, 404}


class Fatal(RuntimeError):
    pass


def result(text="", n_in=0, n_out=0, reasoning=0, cost=None, sec=0.0, error=None, latency=None, **extra):
    return {"text": text or "", "in": int(n_in or 0), "out": int(n_out or 0), "reasoning": int(reasoning or 0), "cost": cost,
            "sec": float(sec), "latency": None if latency is None else float(latency), "error": error, **extra}


def backoff(attempt, retry_after=None, cap=60.0):
    """Giây chờ trước lần thử lại thứ `attempt` (0, 1, …): `Retry-After` nếu có, không thì mũ 2 + jitter, tối đa `cap`."""
    if retry_after is not None:
        try:
            return min(cap, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(cap, 2.0 ** attempt + random.random())


def _make_one(kind, cfg, log, sleep):
    if kind == "openai":
        from vlm.backends.openai_direct import OpenAI
        return OpenAI(cfg, log=log, sleep=sleep)
    if kind == "openrouter":
        from vlm.backends.openrouter import OpenRouter
        return OpenRouter(cfg, log=log, sleep=sleep)
    if kind == "gemini":
        from vlm.backends.gemini import Gemini
        return Gemini(cfg, log=log, sleep=sleep)
    if kind == "vllm":
        from vlm.backends.vllm_local import VLLMLocal
        return VLLMLocal(cfg, log=log)
    raise ValueError(f"backend {kind!r} không thuộc openai | openrouter | gemini | vllm")


class Chain:
    """Thử lần lượt `kinds`; `kind` = backend đang dùng (ghi vào log)."""

    def __init__(self, kinds, cfg, log=print, sleep=time.sleep, make=_make_one):
        self.kinds, self.cfg, self.log, self.sleep, self.make = list(kinds), cfg, log, sleep, make
        self._lock, self._i, self.cur, self.kind = threading.Lock(), -1, None, None
        self._advance("bắt đầu")

    def _advance(self, why):
        while True:
            self._i += 1
            if self._i >= len(self.kinds):
                raise Fatal(f"hết backend để thử ({' -> '.join(self.kinds)}); lỗi cuối: {why}")
            kind = self.kinds[self._i]
            try:
                self.cur, self.kind = self.make(kind, self.cfg, self.log, self.sleep), kind
                self.log(f"[route] dùng {kind} ({self._i + 1} / {len(self.kinds)}: {' -> '.join(self.kinds)})")
                return
            except (Fatal, SystemExit) as e:
                why = f"{kind}: {e}"
                self.log(f"[route] bỏ {kind}: {e}")

    @property
    def concurrency(self):
        return self.cur.concurrency

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
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self.call, jobs))


def make_backend(cfg, log=print, sleep=time.sleep):
    kinds = [cfg["backend"], *(cfg.get("fallback") or [])]
    return Chain(kinds, cfg, log, sleep) if len(kinds) > 1 else _make_one(kinds[0], cfg, log, sleep)
