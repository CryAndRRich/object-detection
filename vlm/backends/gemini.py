"""Gemini API (Google AI Studio) qua REST `v1beta/models/<model>:generateContent` bằng `requests` (không cần SDK; Google: "generateContent
remains fully supported"). Key ở header `x-goog-api-key` (không nằm trong URL / log). Theo docs ER-2 (đọc 2026-11-03):
- `thinkingConfig.thinkingLevel`: minimal | low | medium | high (mặc định high; docs khuyên medium);
- `usageMetadata.{promptTokenCount, candidatesTokenCount, thoughtsTokenCount}` — token suy nghĩ tính tiền như token ra;
- 429 RESOURCE_EXHAUSTED / 408 / 5xx: backoff + jitter; 400 / 403 không thử lại;
- lượt system -> `systemInstruction`; `responseMimeType` JSON + `responseJsonSchema` = schema tác giả (400 vì schema -> `Fatal`);
- free tier: không mất tiền nhưng Google dùng dữ liệu để cải thiện sản phẩm; số RPM / RPD free không công bố (xem trong AI Studio)
  ⇒ `gemini.rpm` trong config giới hạn tốc độ gửi."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from vlm.backends import RETRY_STATUS, Fatal, backoff, result
from vlm.env import key_source, mask, require_key
from vlm.prompt import box_schema

__all__ = ["Gemini", "API", "retry_delay"]

API = "https://generativelanguage.googleapis.com/v1beta/models"


def retry_delay(r):
    """Giây chờ Google gợi ý trong body 429 (`google.rpc.RetryInfo.retryDelay`, vd "37s"; Gemini không gửi header Retry-After)."""
    try:
        for d in (r.json().get("error") or {}).get("details") or []:
            if str(d.get("retryDelay", "")).endswith("s"):
                return float(d["retryDelay"][:-1]) + 1.0
    except (ValueError, AttributeError, TypeError):
        pass
    return None


class Gemini:
    def __init__(self, cfg, log=print, sleep=time.sleep, session=None, clock=time.monotonic):
        self.cfg, self.log, self.sleep, self.clock = cfg, log, sleep, clock
        self.g = cfg.get("gemini") or {}
        self.key = require_key("GEMINI_API_KEY")
        self.http = session or requests.Session()
        self.concurrency = int(self.g.get("concurrency", 4))
        self.rpm = float(self.g.get("rpm") or 0)
        self._lock, self._next = threading.Lock(), 0.0
        log(f"[gemini] key {mask(self.key)} (nguồn: {key_source('GEMINI_API_KEY')}) | "
            f"{cfg['model']} | thinking {self.g.get('thinking_level')} | rpm {self.rpm or '∞'} | "
            f"concurrency {self.concurrency} | free_tier {bool(self.g.get('free_tier'))}")

    def _throttle(self):
        """Giãn đều các lần gửi: cách nhau >= 60 / rpm giây (mọi luồng dùng chung)."""
        if not self.rpm:
            return
        with self._lock:
            now = self.clock()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + 60.0 / self.rpm
        if wait:
            self.sleep(wait)

    def payload(self, job):
        contents, system = [], []
        for t in job["messages"]:
            if t["role"] == "system":
                system.append({"text": t["text"]})
                continue
            parts = [{"text": t["text"]}]
            if t["image"]:
                parts.append({"inlineData": {"mimeType": t.get("mime") or "image/jpeg", "data": t["image"]}})
            contents.append({"role": "model" if t["role"] == "assistant" else "user", "parts": parts})
        gen = {"maxOutputTokens": int(self.cfg["max_tokens"]), **(self.cfg.get("sampling") or {})}
        if job.get("temperature") is not None:
            gen["temperature"] = job["temperature"]
        if self.g.get("thinking_level"):
            gen["thinkingConfig"] = {"thinkingLevel": self.g["thinking_level"]}
        gen["responseMimeType"] = "application/json"
        gen["responseJsonSchema"] = box_schema(self.cfg["k"])["schema"]
        body = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": system}
        return body

    def call(self, job):
        body = self.payload(job)
        url = f"{API}/{self.cfg['model']}:generateContent"
        hdr = {"x-goog-api-key": self.key, "Content-Type": "application/json"}
        t0 = time.time()
        tries = int(self.g.get("http_retries", 6))
        err = None
        for i in range(tries + 1):
            self._throttle()
            t_req = time.time()
            try:
                r = self.http.post(url, headers=hdr, json=body, timeout=float(self.g.get("timeout", 300)))
            except requests.RequestException as e:
                err = f"{type(e).__name__}: {e}"
                if i < tries:
                    self.sleep(backoff(i))
                continue
            if r.status_code in (401, 403, 404):
                raise Fatal(f"Gemini {r.status_code} cho {self.cfg['model']}: {r.text[:300]}")
            if r.status_code == 429 and "PerDay" in r.text:          # hết quota NGÀY (free tier: RPD) — thử lại vô ích
                raise Fatal(f"Gemini hết quota ngày cho {self.cfg['model']} (429 {r.text[:200]}) — chạy lại hôm sau (resume) hoặc bật "
                            "billing")
            if r.status_code in RETRY_STATUS:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if i < tries:
                    self.sleep(backoff(i, r.headers.get("Retry-After") or retry_delay(r)))
                continue
            if r.status_code == 400 and "chema" in r.text:
                raise Fatal(f"Gemini 400 với responseJsonSchema (schema tác giả) cho {self.cfg['model']}: {r.text[:300]}")
            if r.status_code != 200:
                return result(sec=time.time() - t0, error=f"HTTP {r.status_code}: {r.text[:300]}")
            return self.parse(r.json(), time.time() - t0, time.time() - t_req)
        return result(sec=time.time() - t0, error=f"hết {tries} lần thử lại: {err}")

    def parse(self, d, sec=0.0, latency=None):
        """GenerateContentResponse -> Result (gọi thường + Batch API). `cached` = token vào trúng cache ngầm (implicit caching,
        `usageMetadata.cachedContentTokenCount`) — tính theo `price.cached`."""
        u = d.get("usageMetadata") or {}
        cand = (d.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []) if not p.get("thought"))
        think = u.get("thoughtsTokenCount") or 0
        return result(text, u.get("promptTokenCount"), (u.get("candidatesTokenCount") or 0) + think, think, None, sec,
                      None if text else f"không có text (finishReason {cand.get('finishReason')})", latency,
                      cached=u.get("cachedContentTokenCount") or 0, finish=cand.get("finishReason"))

    def batch(self, jobs):
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self.call, jobs))
