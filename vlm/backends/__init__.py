"""Backend VLM: `make_backend(cfg, log)` theo `cfg["backend"]` (openrouter | vllm; import lười — venv chính không cần vllm).
Giao diện chung: `backend.batch(jobs) -> [Result]`, job = {"image_id", "messages", "temperature" (None = theo config), "attempt"};
Result = dict
{"text", "in", "out" (gồm token suy nghĩ), "reasoning", "cost" (USD, None = tính từ giá config), "sec" (cả lần gọi, gồm chờ thử lại
HTTP), "latency" (chỉ request thành công cuối: gửi -> nhận đủ; None nếu không đo được từng ảnh), "error"}. Lỗi không cứu được
(sai key, hết tiền, model không tồn tại) -> `Fatal`: dừng cả job, câu trả lời đã có vẫn còn trong responses.jsonl."""

import hashlib
import random
import time

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403, 404}


class Fatal(RuntimeError):
    pass


def result(text="", n_in=0, n_out=0, reasoning=0, cost=None, sec=0.0, error=None, latency=None, **extra):
    return {"text": text or "", "in": int(n_in or 0), "out": int(n_out or 0), "reasoning": int(reasoning or 0), "cost": cost,
            "sec": float(sec), "latency": None if latency is None else float(latency), "error": error, **extra}


def stable_int(s, nbytes=4, order="big"):
    """Số nguyên tất định từ chuỗi (blake2b) — seed / chọn ảnh / chọn lỗ không phụ thuộc tiến trình hay PYTHONHASHSEED."""
    return int.from_bytes(hashlib.blake2b(str(s).encode(), digest_size=nbytes).digest(), order)


def chat_messages(job, detail=None):
    """Hội thoại kiểu OpenAI chat (OpenRouter + vLLM `LLM.chat` dùng chung): system / assistant = chuỗi; user = [text, ảnh data URL]
    (ảnh trước nếu lượt có `image_first` — RePlan); `detail` = độ phân giải ảnh OpenAI (None = không gửi)."""
    msgs = []
    for t in job["messages"]:
        if t["role"] != "user":
            msgs.append({"role": t["role"], "content": t["text"]})
            continue
        content = [{"type": "text", "text": t["text"]}]
        if t["image"]:
            img = {"url": f"data:{t.get('mime') or 'image/jpeg'};base64,{t['image']}"}
            if detail:
                img["detail"] = detail
            content.append({"type": "image_url", "image_url": img})
            if t.get("image_first"):
                content.reverse()
        msgs.append({"role": "user", "content": content})
    return msgs


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


def make_backend(cfg, log=print, sleep=time.sleep):
    kind = cfg["backend"]
    if kind == "openrouter":
        from vlm.backends.openrouter import OpenRouter
        return OpenRouter(cfg, log=log, sleep=sleep)
    if kind == "vllm":
        from vlm.backends.vllm_local import VLLMLocal
        return VLLMLocal(cfg, log=log)
    raise ValueError(f"backend {kind!r} không thuộc openrouter | vllm")
