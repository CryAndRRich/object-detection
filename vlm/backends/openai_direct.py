"""OpenAI trực tiếp (`POST https://api.openai.com/v1/chat/completions`, key `OPENAI_API_KEY`) — cùng hội thoại / schema tác giả như
OpenRouter (`chat_messages`, `box_schema`), chỉ khác đường đi. Theo docs OpenAI:
- `max_completion_tokens` (model suy nghĩ không nhận `max_tokens`), `reasoning_effort`, `response_format` json_schema;
- response KHÔNG có tiền ⇒ `cost` None ⇒ runner tính token × `price` config (`price.cached` cho token vào đã cache nếu có);
- KHÔNG có API đọc số dư cho key thường ⇒ lúc khởi động gửi MỘT request thăm dò 16 token (~$0,00001): hết tiền = 429
  `insufficient_quota` -> `Fatal` (có `fallback` thì chuyển OpenRouter); giữa chừng hết tiền cũng vậy;
- 400 `unsupported_parameter` / `unsupported_value`: bỏ tham số sampling đó rồi gửi lại (như OpenRouter bỏ tham số model không
  nhận); riêng `reasoning_effort` / `response_format` không bỏ (đổi câu trả lời) -> `Fatal`."""

import time
from concurrent.futures import ThreadPoolExecutor

import requests

from vlm.backends import RETRY_STATUS, Fatal, backoff, result
from vlm.backends.openrouter import chat_messages
from vlm.env import key_source, mask, require_key
from vlm.prompt import box_schema

__all__ = ["OpenAI", "API"]

API = "https://api.openai.com/v1"
NO_DROP = ("reasoning_effort", "response_format", "messages", "model")


def _err(r):
    try:
        e = r.json().get("error") or {}
        return e if isinstance(e, dict) else {"message": str(e)}
    except ValueError:
        return {"message": r.text[:300]}


class OpenAI:
    def __init__(self, cfg, log=print, sleep=time.sleep, session=None):
        self.cfg, self.log, self.sleep = cfg, log, sleep
        self.o = cfg.get("openai") or {}
        self.key = require_key("OPENAI_API_KEY")
        if self.key.startswith("sk-or-"):
            raise Fatal("OPENAI_API_KEY là key OpenRouter (sk-or-…) — key OpenAI tạo ở https://platform.openai.com/api-keys")
        self.model = self.o.get("model") or cfg["model"].removeprefix("openai/")
        self.detail = self.o.get("detail", (cfg.get("openrouter") or {}).get("detail"))     # cùng `detail` với OpenRouter (hash)
        self.http = session or requests.Session()
        self.headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        self.concurrency = int(self.o.get("concurrency", 8))
        self._dropped = set()
        p = cfg.get("price") or {}
        log(f"[openai] key {mask(self.key)} (nguồn: {key_source('OPENAI_API_KEY')}) | "
            f"{self.model} | concurrency {self.concurrency} | tiền = token × giá config "
            f"vào ${p.get('in')} / ra ${p.get('out')} / cache ${p.get('cached', p.get('in'))} mỗi 1M (OpenAI không trả tiền mỗi request)")
        self._probe()

    def _probe(self):
        """Key + quyền model + còn tiền: GET /models/<model> (miễn phí) rồi một request 16 token."""
        r = self.http.get(f"{API}/models/{self.model}", headers=self.headers, timeout=30)
        if r.status_code in (401, 403):
            raise Fatal(f"OpenAI từ chối key ({r.status_code}): {_err(r).get('message', '')[:200]}")
        if r.status_code == 404:
            raise Fatal(f"key OpenAI không dùng được model {self.model!r} (404)")
        if not self.o.get("probe", True):
            return
        body = {"model": self.model, "messages": [{"role": "user", "content": "Reply OK."}], "max_completion_tokens": 16}
        if self.cfg.get("retries", 2):                 # biết trước model có nhận temperature (lần thử lại) không — Batch không hỏi lại được
            body["temperature"] = float(self.cfg.get("retry_temperature", 0.7))
        r = self.http.post(f"{API}/chat/completions", headers=self.headers, json=body, timeout=60)
        e = _err(r) if r.status_code != 200 else {}
        if self._learn_drop(r.status_code, e, body):
            body.pop(e["param"])
            r = self.http.post(f"{API}/chat/completions", headers=self.headers, json=body, timeout=60)
            e = _err(r) if r.status_code != 200 else {}
        if r.status_code == 429 and e.get("code") == "insufficient_quota" or r.status_code in (401, 402, 403):
            raise Fatal(f"OpenAI {r.status_code} {e.get('code')}: hết tiền / quota — nạp ở "
                        f"https://platform.openai.com/settings/organization/billing ({e.get('message', '')[:150]})")
        self.log(f"[openai] thăm dò: HTTP {r.status_code} -> key còn dùng được (số dư: OpenAI không có API đọc cho key thường, xem "
                 f"trang billing; theo dõi tiền bằng --max-cost)")

    def payload(self, job):
        body = {"model": self.model, "messages": chat_messages(job, self.detail), "max_completion_tokens": int(self.cfg["max_tokens"])}
        opt = {**(self.cfg.get("sampling") or {})}
        if job.get("temperature") is not None:
            opt["temperature"] = job["temperature"]
        if self.cfg.get("reasoning"):
            opt["reasoning_effort"] = self.cfg["reasoning"]
        body.update({k: v for k, v in opt.items() if k not in self._dropped})
        body["response_format"] = {"type": "json_schema", "json_schema": box_schema(self.cfg["k"])}
        return body

    def call(self, job):
        t0 = time.time()
        tries = int(self.o.get("http_retries", 6))
        err = None
        for i in range(tries + 1):
            body = self.payload(job)
            t_req = time.time()
            try:
                r = self.http.post(f"{API}/chat/completions", headers=self.headers, json=body, timeout=float(self.o.get("timeout", 300)))
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
                if i < tries:
                    self.sleep(backoff(i))
                continue
            if r.status_code == 200:
                return self.parse(r.json(), time.time() - t0, time.time() - t_req)
            e = _err(r)
            if self._learn_drop(r.status_code, e, body):
                continue
            if r.status_code == 429 and e.get("code") == "insufficient_quota" or r.status_code in (401, 402, 403, 404):
                raise Fatal(f"OpenAI {r.status_code} {e.get('code')} cho {self.model}: {e.get('message', '')[:300]}")
            if r.status_code in RETRY_STATUS:
                err = f"HTTP {r.status_code}: {e.get('message', '')[:200]}"
                if i < tries:
                    self.sleep(backoff(i, r.headers.get("Retry-After")))
                continue
            return result(sec=time.time() - t0, error=f"HTTP {r.status_code}: {e.get('message', '')[:300]}")
        return result(sec=time.time() - t0, error=f"hết {tries} lần thử lại: {err}")

    def parse(self, d, sec=0.0, latency=None, provider="OpenAI (trực tiếp)"):
        """Body chat.completion -> Result (dùng chung cho gọi thường + Batch API)."""
        ch = (d.get("choices") or [{}])[0]
        u = d.get("usage") or {}
        msg = ch.get("message") or {}
        text = msg.get("content") or ""
        return result(text, u.get("prompt_tokens"), u.get("completion_tokens"),
                      (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), None, sec,
                      None if text else f"không có text (refusal {str(msg.get('refusal'))[:100]}, {ch.get('finish_reason')})",
                      latency, cached=(u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
                      gen_id=d.get("id"), finish=ch.get("finish_reason"), provider=provider)

    def _learn_drop(self, status, e, body):
        """400 unsupported_parameter / value cho tham số sampling -> bỏ hẳn tham số đó (True = gửi lại); reasoning / schema -> Fatal."""
        if status != 400 or e.get("code") not in ("unsupported_parameter", "unsupported_value") or e.get("param") not in body:
            return False
        if e["param"] in NO_DROP:
            raise Fatal(f"OpenAI {self.model} không nhận `{e['param']}`: {e.get('message', '')[:200]}")
        if e["param"] not in self._dropped:
            self._dropped.add(e["param"])
            self.log(f"[openai] {self.model} không nhận `{e['param']}` -> bỏ")
        return True

    def batch(self, jobs):
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self.call, jobs))
