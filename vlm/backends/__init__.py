"""Backend VLM: `make_backend(cfg, log)` theo `cfg["backend"]` (openrouter | gemini | vllm; import lười — venv chính không cần vllm).
Giao diện chung: `backend.batch(jobs) -> [Result]`, job = {"image_id", "messages", "temperature" (None = theo config)}; Result = dict
{"text", "in", "out" (gồm token suy nghĩ), "reasoning", "cost" (USD, None = tính từ giá config), "sec", "error"}. Lỗi không cứu được
(sai key, hết tiền, model không tồn tại) -> `Fatal`: dừng cả job, câu trả lời đã có vẫn còn trong responses.jsonl."""

import random
import time

__all__ = ["Fatal", "make_backend", "result", "backoff", "RETRY_STATUS", "FATAL_STATUS"]

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403, 404}


class Fatal(RuntimeError):
    pass


def result(text="", n_in=0, n_out=0, reasoning=0, cost=None, sec=0.0, error=None, **extra):
    return {"text": text or "", "in": int(n_in or 0), "out": int(n_out or 0), "reasoning": int(reasoning or 0), "cost": cost,
            "sec": float(sec), "error": error, **extra}


def backoff(attempt, retry_after=None, cap=60.0):
    """Giây chờ trước lần thử lại thứ `attempt` (0, 1, …): `Retry-After` nếu có, không thì mũ 2 + jitter, tối đa `cap`."""
    if retry_after is not None:
        try:
            return min(cap, max(0.0, float(retry_after)))
        except ValueError:
            pass
    return min(cap, 2.0 ** attempt + random.random())


def make_backend(cfg, log=print, sleep=time.sleep):
    kind = cfg["backend"]
    if kind == "openrouter":
        from vlm.backends.openrouter import OpenRouter
        return OpenRouter(cfg, log=log, sleep=sleep)
    if kind == "gemini":
        from vlm.backends.gemini import Gemini
        return Gemini(cfg, log=log, sleep=sleep)
    if kind == "vllm":
        from vlm.backends.vllm_local import VLLMLocal
        return VLLMLocal(cfg, log=log)
    raise ValueError(f"backend {kind!r} không thuộc openrouter | gemini | vllm")
