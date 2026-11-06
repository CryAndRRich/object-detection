"""Batch API (`run.py --batch`): gửi cả lô request một lần, provider trả trong <= 24 giờ, giá token = 50 % gọi thường. Câu trả lời như
gọi thường (cùng payload / model / schema); KHÔNG dùng để đo latency (`tools/latency.py` luôn gọi thường). Docs đọc 2026-11-04:
- OpenAI (developers.openai.com/api/docs/guides/batch): file JSONL `{"custom_id", "method", "url": "/v1/chat/completions", "body"}`,
  <= 50.000 request và <= 200 MB / file, `completion_window` chỉ `24h`; quá hạn -> `expired`: request xong vẫn trả + tính tiền, chưa xong
  bị huỷ; thứ tự dòng ra có thể khác dòng vào (khớp theo `custom_id`); lỗi từng request ở `error_file_id`.
(Gemini Batch đã bỏ cùng ER-2, 2026-11-06 — git giữ lịch sử.)
An toàn tiền: (1) mỗi lô ghi vào `<out_dir>/batch/state.json` NGAY khi tạo ⇒ chạy lại cùng lệnh thì CHỜ / LẤY lô cũ, không gửi lại;
(2) file kết quả tải về giữ ở `<out_dir>/batch/`; (3) giá = `price_batch` của config (bảng giá Batch chính thức) — thiếu thì dừng, không
đoán; (4) runner chặn `--max-cost` TRƯỚC khi gửi (ước theo TB tiền / ảnh đã có; chưa có thì lô đầu chỉ `batch.first` ảnh)."""

import json
import os
import re
import time

import requests

from vlm.backends import Fatal, backoff, result, token_cost

__all__ = ["BatchBackend", "OpenAIBatchAPI", "LimitError"]

TERMINAL = ("succeeded", "failed", "expired", "cancelled")


class LimitError(RuntimeError):
    """Provider từ chối vì hạn mức hàng đợi batch (token đang xếp hàng) — chờ rồi gửi lại, không tốn tiền."""


class _HTTP:
    """requests + thử lại lỗi mạng / 429 / 5xx (poll, tải file) — không thử lại 4xx khác."""

    def __init__(self, http, sleep, tries=8):
        self.http, self.sleep, self.tries = http, sleep, tries

    def __call__(self, method, url, retry429=True, **kw):
        """`retry429` False (tạo lô): 429 = hàng đợi batch đầy -> trả về ngay để gọi nơi chờ lâu (`LimitError`)."""
        err, timeout = None, kw.pop("timeout", 600)
        for i in range(self.tries):
            try:
                r = self.http.request(method, url, timeout=timeout, **kw)
            except requests.RequestException as e:
                err = e
                self.sleep(backoff(i, cap=120))
                continue
            if (r.status_code == 429 and retry429) or r.status_code >= 500:
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                self.sleep(backoff(i, r.headers.get("Retry-After"), cap=120))
                continue
            return r
        raise requests.ConnectionError(f"{method} {url}: hết {self.tries} lần thử ({err})")


class OpenAIBatchAPI:
    name = "openai"
    max_bytes, max_requests = 190e6, 50000

    def __init__(self, sync, sleep=time.sleep):
        self.sync = sync
        self.api = "https://api.openai.com/v1"
        self.h = {"Authorization": sync.headers["Authorization"]}
        self.req = _HTTP(sync.http, sleep)

    def line(self, key, job):
        return {"custom_id": key, "method": "POST", "url": "/v1/chat/completions", "body": self.sync.payload(job)}

    def create(self, data, label):
        r = self.req("POST", f"{self.api}/files", headers=self.h, files={"file": (f"{label}.jsonl", data)}, data={"purpose": "batch"})
        if r.status_code != 200:
            raise Fatal(f"OpenAI tải file batch lỗi {r.status_code}: {r.text[:300]}")
        fid = r.json()["id"]
        r = self.req("POST", f"{self.api}/batches", retry429=False, headers={**self.h, "Content-Type": "application/json"},
                     json={"input_file_id": fid, "endpoint": "/v1/chat/completions", "completion_window": "24h",
                           "metadata": {"label": label[:64]}})
        if r.status_code == 429:
            raise LimitError(f"OpenAI tạo batch 429: {r.text[:200]}")
        if r.status_code != 200:
            raise Fatal(f"OpenAI tạo batch lỗi {r.status_code}: {r.text[:300]}")
        return r.json()["id"]

    def status(self, bid):
        r = self.req("GET", f"{self.api}/batches/{bid}", headers=self.h)
        if r.status_code != 200:
            raise Fatal(f"OpenAI đọc batch {bid} lỗi {r.status_code}: {r.text[:300]}")
        d = r.json()
        st = d.get("status")
        state = {"completed": "succeeded", "failed": "failed", "expired": "expired", "cancelled": "cancelled"}.get(st, "running")
        errs = (d.get("errors") or {}).get("data") or []
        reason = "; ".join(f"{e.get('code')}: {e.get('message')}" for e in errs)[:400]
        c = d.get("request_counts") or {}
        return {"state": state, "raw": st, "done": c.get("completed", 0), "failed": c.get("failed", 0), "total": c.get("total"),
                "files": [f for f in (d.get("output_file_id"), d.get("error_file_id")) if f], "reason": reason,
                "limit": any(e.get("code") == "token_limit_exceeded" for e in errs)}

    def download(self, fid):
        r = self.req("GET", f"{self.api}/files/{fid}/content", headers=self.h)
        if r.status_code != 200:
            raise Fatal(f"OpenAI tải kết quả {fid} lỗi {r.status_code}: {r.text[:300]}")
        return r.text

    def parse_line(self, d):
        resp = d.get("response") or {}
        if resp.get("status_code") == 200 and isinstance(resp.get("body"), dict):
            return d["custom_id"], self.sync.parse(resp["body"], provider="OpenAI Batch")
        e = (resp.get("body") or {}).get("error") if isinstance(resp.get("body"), dict) else None
        e = e or d.get("error") or {}
        return d.get("custom_id"), result(error=f"batch: {e.get('code')}: {str(e.get('message'))[:200]}",
                                          expired=e.get("code") == "batch_expired", param=e.get("param"), code=e.get("code"))


class BatchBackend:
    """`batch(jobs)` = gửi (hoặc lấy lại) các lô chứa đúng những request này, chờ xong, trả Result theo thứ tự `jobs`. Khoá request
    = `<image_id>#<lần thử>` ⇒ lần thử lại (0 box, nhiệt `retry_temperature`) là request mới, còn chạy lại sau khi bị ngắt thì khớp
    đúng lô cũ."""

    is_batch = True

    def __init__(self, sync, api, cfg, log=print, sleep=time.sleep, clock=time.time):
        self.sync, self.api, self.cfg, self.log, self.sleep, self.clock = sync, api, cfg, log, sleep, clock
        self.b = cfg.get("batch") or {}
        self.price = cfg.get("price_batch")
        if not self.price or "in" not in self.price or "out" not in self.price:
            raise Fatal(f"config {cfg.get('name')} thiếu `price_batch: {{in, out[, cached]}}` (bảng giá Batch chính thức) — không đoán giá")
        self.concurrency = 1
        self.state_dir = None
        self.poll = float(self.b.get("poll_sec", 60))
        self.log_every = float(self.b.get("log_sec", 600))
        self.max_requests = int(min(self.b.get("max_requests", 1000), api.max_requests))
        self.max_bytes = float(min(float(self.b.get("max_mb", 1e9)) * 1e6, api.max_bytes))
        self.limit_wait = float(self.b.get("limit_wait_sec", 900))
        self.resubmit = int(self.b.get("resubmit", 3))
        log(f"[batch] {api.name} | {cfg['model']} | giá batch vào ${self.price['in']} / ra ${self.price['out']} / cache "
            f"${self.price.get('cached', self.price['in'])} mỗi 1M | <= {self.max_requests} request, <= {self.max_bytes / 1e6:.0f} MB / lô")

    def set_state_dir(self, d):
        self.state_dir = d

    # ---- trạng thái lô (chống gửi trùng) ----
    def _path(self, *p):
        if not self.state_dir:
            raise RuntimeError("BatchBackend cần set_state_dir(<out_dir>/batch) trước khi gửi")
        os.makedirs(self.state_dir, exist_ok=True)
        return os.path.join(self.state_dir, *p)

    def _load(self):
        p = self._path("state.json")
        if not os.path.exists(p):
            return []
        with open(p) as f:
            return json.load(f)["parts"]

    def _save(self, parts):
        p = self._path("state.json")
        with open(p + ".tmp", "w") as f:
            json.dump({"model": self.cfg["model"], "provider": self.api.name, "parts": parts}, f, indent=1)
        os.replace(p + ".tmp", p)

    @staticmethod
    def key(job):
        return f"{job['image_id']}#{int(job.get('attempt') or 0)}"

    # ---- gửi / chờ / lấy ----
    def _split(self, jobs):
        """-> [(keys, bytes)] mỗi phần <= max_requests và <= max_bytes."""
        parts, keys, buf, size = [], [], [], 0
        for j in jobs:
            b = (json.dumps(self.api.line(self.key(j), j), ensure_ascii=False) + "\n").encode()
            if keys and (len(keys) >= self.max_requests or size + len(b) > self.max_bytes):
                parts.append((keys, b"".join(buf)))
                keys, buf, size = [], [], 0
            keys.append(self.key(j))
            buf.append(b)
            size += len(b)
        if keys:
            parts.append((keys, b"".join(buf)))
        return parts

    def _submit(self, jobs, parts):
        new = []
        for keys, data in self._split(jobs):
            label = f"zeta-{self.cfg.get('name')}-{time.strftime('%m%d-%H%M%S')}-{len(parts)}"
            p = {"id": None, "status": "submitting", "keys": keys, "label": label, "bytes": len(data), "created": self.clock()}
            parts.append(p)
            self._save(parts)                                       # ghi TRƯỚC khi gọi tạo (tạo không idempotent)
            try:
                p["id"] = self.api.create(data, label)
                p["status"] = "running"
                self.log(f"[batch] đã gửi {p['id']}: {len(keys)} request, {len(data) / 1e6:.1f} MB")
            except LimitError as e:
                p["status"], p["reason"], p["limit"] = "failed", str(e), True
                self.log(f"[batch] hàng đợi đầy, chưa gửi {len(keys)} request: {e}")
            self._save(parts)
            new.append(p)
        return new

    def _wait(self, p, parts):
        """Chờ lô `p` tới trạng thái cuối, tải kết quả về `<state_dir>/<id>.jsonl` (một lần). -> {key: Result}."""
        if p["status"] == "failed":
            return {k: result(error=f"batch {p.get('id')}: {p.get('reason')}", resubmit=bool(p.get("limit"))) for k in p["keys"]}
        out = self._path(re.sub(r"[^\w.-]", "_", p["id"]) + ".jsonl")
        t_last, raw_last = 0.0, None
        while not os.path.exists(out):
            try:
                st = self.api.status(p["id"])
            except requests.RequestException as e:            # mạng chập chờn trong lúc chờ nhiều giờ: chờ tiếp, không bỏ lô
                self.log(f"[batch] {p['id']}: không đọc được trạng thái ({e}) — thử lại sau {self.poll:.0f} s")
                self.sleep(self.poll)
                continue
            now = self.clock()
            if st["raw"] != raw_last or now - t_last >= self.log_every or st["state"] in TERMINAL:
                el = now - p["created"]
                self.log(f"[batch] {p['id']} {st['raw']} | {st['done']}/{st['total'] or len(p['keys'])} xong, {st['failed']} lỗi | "
                         f"đã chờ {int(el // 3600)}h{int(el % 3600 // 60):02d}m (hạn OpenAI 24h) | "
                         f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
                t_last, raw_last = now, st["raw"]
            if st["state"] in TERMINAL:
                if st["state"] == "failed" and not st["files"]:
                    p["status"], p["reason"], p["limit"] = "failed", st["reason"], st["limit"]
                    self._save(parts)
                    if not st["limit"]:
                        raise Fatal(f"batch {p['id']} lỗi (không có kết quả): {st['reason']}")
                    return self._wait(p, parts)
                text = "".join(self.api.download(f) for f in st["files"])
                with open(out + ".tmp", "w") as f:
                    f.write(text)
                os.replace(out + ".tmp", out)
                p["status"], p["finished"] = st["state"], now
                self._save(parts)
                break
            self.sleep(self.poll)
        n = max(len(p["keys"]), 1)
        sec = (p.get("finished", self.clock()) - p["created"]) / n          # thông lượng (giờ ước), không phải latency
        res = {}
        with open(out) as f:
            for line in f:
                if line.strip():
                    k, r = self.api.parse_line(json.loads(line))
                    r["sec"], r["latency"], r["batch"] = sec, None, True
                    if not r["error"]:
                        r["cost"] = token_cost(r, self.price)
                    res[k] = r
        for k in p["keys"]:
            if k not in res:                       # hết hạn / huỷ: không có dòng -> chưa tính tiền, gửi lại
                res[k] = result(error=f"batch {p['id']} {p['status']}: không có kết quả", resubmit=True)
            elif res[k].get("expired"):
                res[k]["resubmit"] = True
        return res

    def batch(self, jobs):
        want = {self.key(j): j for j in jobs}
        parts = self._load()
        for p in parts:
            if p["status"] == "submitting":         # bị ngắt giữa lúc tạo: không biết đã tạo chưa
                p["status"], p["reason"] = "failed", "bị ngắt lúc tạo — có thể đã gửi trùng, kiểm trang batches của provider"
                self.log(f"[batch] ⚠️ {p['label']}: {p['reason']}")
        got = {}
        for p in parts:                              # 1) lô cũ còn chứa request đang cần: chờ / lấy, không gửi lại
            mine = [k for k in p["keys"] if k in want and k not in got]
            if not mine or p["status"] == "failed":
                continue
            self.log(f"[batch] dùng lại lô đã gửi {p['id']} ({len(mine)} request cần)")
            r = self._wait(p, parts)
            got.update({k: r[k] for k in mine if not r[k]["error"]})
        rest = [want[k] for k in want if k not in got]
        for i in range(self.resubmit + 1):           # 2) phần còn lại: gửi mới (hàng đợi đầy / hết hạn thì chờ rồi gửi lại)
            if not rest:
                break
            if i:
                self.log(f"[batch] gửi lại {len(rest)} request sau {self.limit_wait / 60:.0f} phút "
                         "(hàng đợi đầy / hết hạn, chưa tính tiền)")
                self.sleep(self.limit_wait)
            outs = {}
            for p in self._submit(rest, parts):
                outs.update(self._wait(p, parts))
            nxt = []
            for j in rest:
                r = outs.get(self.key(j)) or result(error="batch: không có kết quả", resubmit=True)
                if r.get("code") in ("unsupported_parameter", "unsupported_value") and hasattr(self.sync, "_learn_drop"):
                    if self.sync._learn_drop(400, {"code": r["code"], "param": r["param"]}, self.sync.payload(j)):
                        r["resubmit"] = True                # tham số sampling model không nhận: bỏ rồi gửi lại
                if r.get("resubmit") and i < self.resubmit:
                    nxt.append(j)
                else:
                    got[self.key(j)] = r
            rest = nxt
        return [got.get(self.key(j)) or result(error="batch: không có kết quả sau các lần gửi lại") for j in jobs]
