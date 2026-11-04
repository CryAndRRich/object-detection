"""OpenRouter (`POST /api/v1/chat/completions`, OpenAI-compatible) bằng `requests` + ThreadPool. Theo docs OpenRouter (đọc 2026-11-03):
- `usage.cost` (USD) có sẵn trong MỌI response (cờ `usage.include` cũ không còn tác dụng); token suy nghĩ ở
  `usage.completion_tokens_details.reasoning_tokens` (đã nằm trong `completion_tokens`).
- `reasoning: {"effort": none | minimal | low | medium | high | xhigh | max}`; `provider: {"data_collection": "deny", "zdr": …}`.
- 429 / 5xx: chờ (`Retry-After`) rồi thử lại; 401 / 402 / 403 / 404 = `Fatal`.
- `response_format` json_schema = schema `insertion_layout` của tác giả (strict False) + `provider.require_parameters`; model không có
  `structured_outputs` trong `supported_parameters` (`/api/v1/models`) -> `Fatal`.
- Lượt system / assistant = chuỗi; lượt user = text rồi ảnh data URL (như tác giả)."""

import time
from concurrent.futures import ThreadPoolExecutor

import requests

from vlm.backends import FATAL_STATUS, RETRY_STATUS, Fatal, backoff, result
from vlm.env import mask, require_key
from vlm.prompt import box_schema

__all__ = ["OpenRouter", "API"]

API = "https://openrouter.ai/api/v1"


class OpenRouter:
    def __init__(self, cfg, log=print, sleep=time.sleep, session=None):
        self.cfg, self.log, self.sleep = cfg, log, sleep
        self.o = cfg.get("openrouter") or {}
        self.key = require_key("OPENROUTER_API_KEY")
        if not self.key.startswith("sk-or-"):          # key OpenRouter: sk-or-v1-…; sk-proj- / sk- = key OpenAI -> 401 khó hiểu
            raise Fatal(f"OPENROUTER_API_KEY không phải key OpenRouter (phải bắt đầu bằng sk-or-, đang là {self.key[:8]}…) — tạo key ở "
                        f"https://openrouter.ai/settings/keys; key OpenAI (sk-proj-…) không dùng được cho OpenRouter")
        self.http = session or requests.Session()
        self.headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                        "X-OpenRouter-Title": "multi_condition ZETA5 / ZETA6"}
        self.concurrency = int(self.o.get("concurrency", 8))
        self.model_info = self._model_info()
        self.supported, self._dropped = set(self.model_info.get("supported_parameters") or []), set()
        if "structured_outputs" not in self.supported:                                         # schema tác giả qua API
            raise Fatal(f"{cfg['model']} không có structured_outputs trên OpenRouter (`/api/v1/models`) -> không gửi được schema "
                        "tác giả, chọn model khác")
        log(f"[openrouter] key {mask(self.key)} | {cfg['model']} | concurrency {self.concurrency} | provider {self.o.get('provider')}")
        self._print_credits()

    def _model_info(self):
        r = self.http.get(f"{API}/models", timeout=60)
        r.raise_for_status()
        for m in r.json()["data"]:
            if m["id"] == self.cfg["model"]:
                p = m.get("pricing", {})
                self.log(f"[openrouter] giá hiện tại {self.cfg['model']}: vào ${float(p.get('prompt', 0)) * 1e6:.3f} / "
                         f"ra ${float(p.get('completion', 0)) * 1e6:.3f} / 1M token")
                return m
        raise Fatal(f"OpenRouter không có model {self.cfg['model']!r} (sửa `model` trong config)")

    def _print_credits(self):
        """In hạn mức key (`/key`) + số dư tài khoản (`/credits`). OpenRouter TẠM GIỮ credit cho mỗi request đang bay theo trần
        (token vào + `max_tokens` × giá ra) ⇒ số dư < concurrency × trần thì bị 402 `in_flight_budget_exhausted`."""
        try:
            r = self.http.get(f"{API}/key", headers=self.headers, timeout=30)
            if r.status_code in FATAL_STATUS:
                raise Fatal(f"OpenRouter từ chối key ({r.status_code}): {r.text[:200]}")
            d = r.json().get("data", {})
            self.log(f"[openrouter] key: đã dùng ${d.get('usage') or 0:.4f}, hạn mức KEY còn {d.get('limit_remaining')} (không phải số dư)")
            r = self.http.get(f"{API}/credits", headers=self.headers, timeout=30)
            if r.status_code == 200:
                c = r.json().get("data", {})
                bal = float(c.get("total_credits") or 0) - float(c.get("total_usage") or 0)
                p = self.model_info.get("pricing", {})
                hold = (12000 * float(p.get("prompt", 0)) + int(self.cfg["max_tokens"]) * float(p.get("completion", 0))) * self.concurrency
                self.log(f"[openrouter] số dư tài khoản ${bal:.4f} | tạm giữ ước ~${hold:.3f} cho {self.concurrency} request song song")
                if bal < hold:
                    self.log("[openrouter] ⚠️ số dư < tạm giữ -> sẽ bị 402 in-flight (code chờ + thử lại); nạp credit hoặc giảm "
                             "`openrouter.concurrency`")
        except requests.RequestException as e:
            self.log(f"[openrouter] không đọc được /key, /credits ({e}) — chạy tiếp")

    def payload(self, job):
        msgs = []
        for t in job["messages"]:
            content = [{"type": "text", "text": t["text"]}]
            if t["image"]:
                img = {"url": f"data:{t.get('mime') or 'image/jpeg'};base64,{t['image']}"}
                if self.o.get("detail"):
                    img["detail"] = self.o["detail"]
                content.append({"type": "image_url", "image_url": img})
            msgs.append({"role": t["role"], "content": content if t["role"] == "user" else t["text"]})
        body = {"model": self.cfg["model"], "messages": msgs, "max_tokens": int(self.cfg["max_tokens"])}
        opt = {**(self.cfg.get("sampling") or {})}
        if job.get("temperature") is not None:
            opt["temperature"] = job["temperature"]
        if self.cfg.get("reasoning"):
            opt["reasoning"] = {"effort": self.cfg["reasoning"]}
        for k, v in opt.items():               # bỏ tham số model không nhận (require_parameters sẽ không route được)
            if k in self.supported:
                body[k] = v
            elif k not in self._dropped:
                self._dropped.add(k)
                self.log(f"[openrouter] {self.cfg['model']} không nhận `{k}` -> bỏ")
        body["response_format"] = {"type": "json_schema", "json_schema": box_schema(self.cfg["k"])}
        body["provider"] = {"require_parameters": True, **(self.o.get("provider") or {})}  # chỉ route tới provider nhận response_format
        return body

    def call(self, job):
        body = self.payload(job)
        t0 = time.time()
        tries = int(self.o.get("http_retries", 6))
        err = None
        for i in range(tries + 1):
            t_req = time.time()
            try:
                r = self.http.post(f"{API}/chat/completions", headers=self.headers, json=body, timeout=float(self.o.get("timeout", 300)))
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
                if i < tries:
                    self.sleep(backoff(i))
                continue
            if r.status_code == 402 and "in_flight" in r.text:      # tạm giữ credit cho request đang bay: chờ rồi thử lại
                err = f"HTTP 402 in-flight: {r.text[:200]}"
                if i < tries:
                    self.sleep(backoff(i + 2))
                continue
            if r.status_code in FATAL_STATUS:
                raise Fatal(f"OpenRouter {r.status_code} cho {self.cfg['model']}: {r.text[:300]}")
            if r.status_code in RETRY_STATUS:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if i < tries:
                    self.sleep(backoff(i, r.headers.get("Retry-After")))
                continue
            if r.status_code != 200:
                return result(sec=time.time() - t0, error=f"HTTP {r.status_code}: {r.text[:300]}")
            d = r.json()
            if "error" in d and not d.get("choices"):
                err = f"error: {str(d['error'])[:300]}"
                if i < tries:
                    self.sleep(backoff(i))
                continue
            ch = d["choices"][0]
            u = d.get("usage") or {}
            text = ch.get("message", {}).get("content") or ""
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            return result(text, u.get("prompt_tokens"), u.get("completion_tokens"),
                          (u.get("completion_tokens_details") or {}).get("reasoning_tokens"), u.get("cost"), time.time() - t0,
                          None, time.time() - t_req, gen_id=d.get("id"), finish=ch.get("finish_reason"), provider=d.get("provider"))
        return result(sec=time.time() - t0, error=f"hết {tries} lần thử lại: {err}")

    def batch(self, jobs):
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self.call, jobs))
