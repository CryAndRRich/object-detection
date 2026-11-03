"""Chạy một VLM trên một tập test (CE-130 `test_all` | CE-CoCount `review5`): chọn mẫu (pilot K ảnh | all), resume, gọi backend theo
khối, parse, ghi từng câu trả lời ngay, tiến độ + ETA + tiền, ước tiền / giờ cho cả tập, rồi chấm bằng đúng code chấm ZETA
(`baseline/score.py::run_score`): K = 30 -> `<name>.json` (+ `_boxes.json`), K = 1 (box đầu) -> `<name>_k1.json`.

Pilot = K mẫu đầu theo hash blake2b(`seed:image_id`) ⇒ ổn định, và là tập con của all ⇒ chạy all dùng lại câu trả lời pilot.
`responses.jsonl` (một dòng / ảnh, có `prompt_hash`): resume bỏ ảnh đã xong cùng hash; đổi prompt / model / sampling ⇒ hỏi lại.
Lỗi API (sau khi backend đã thử lại) KHÔNG ghi ⇒ chạy lại sẽ hỏi lại ảnh đó; model trả lời mà 0 box hợp lệ sau `retries` lần ⇒ ghi
(`n_valid` = 0, chấm bằng box cả ảnh)."""

import hashlib
import json
import os
import time

import numpy as np

from ce_localization.data.turns import iter_jsonl
from vlm.backends import Fatal
from vlm.parse import fill, parse_boxes
from vlm.prompt import author_turns, build_messages, encode_image

__all__ = ["dataset_dir", "select", "load_responses", "record_cost", "summarize", "run_dataset", "FULL_SIZES"]

FULL_SIZES = {"ce130": 6122, "cocount": 1263}           # test_all / review5 — chỉ để in khi chạy một tập


def dataset_dir(ns):
    return f"ce130_{ns.split}" if ns.dataset == "ce130" else f"cocount_review{ns.cocount_min_obj}"


def _rank(seed, image_id):
    return hashlib.blake2b(f"{seed}:{image_id}".encode(), digest_size=8).digest()


def select(samples, seed, pilot=None):
    """all: giữ đúng thứ tự `load_samples` (mốc prior trùng eval CE-Loc); pilot K: K mẫu đầu theo hash, rồi trả về theo thứ tự gốc."""
    if not pilot:
        return list(samples)
    keep = {s["image_id"] for s in sorted(samples, key=lambda s: _rank(seed, s["image_id"]))[:pilot]}
    return [s for s in samples if s["image_id"] in keep]


def load_responses(path, phash):
    if not os.path.exists(path):
        return {}
    return {r["image_id"]: r for r in iter_jsonl(path) if r.get("prompt_hash") == phash}


def record_cost(res, cfg):
    """USD của một lần gọi: `cost` backend trả (OpenRouter) hay token × giá config (`price.in` / `price.out` USD / 1M)."""
    if res.get("cost") is not None:
        return float(res["cost"])
    p = cfg.get("price") or {}
    return (res["in"] * float(p.get("in", 0)) + res["out"] * float(p.get("out", 0))) / 1e6


def _fmt_sec(s):
    s = int(round(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _k(n):
    return f"{n / 1000:.1f}k"


def _icl_turns(cfg, ns, log=print):
    """Lượt few-shot theo `icl`: "author" = 2 ví dụ của tác giả (`vlm/icl/author.json`); 0 = zero-shot."""
    icl = cfg.get("icl") or 0
    if icl == 0:
        return []
    if icl != "author":
        raise ValueError(f"icl {icl!r} không thuộc author | 0")
    img = cfg.get("image") or {}
    if ns.dataset == "ce130":
        log(f"[{ns.dataset}] ⚠️ icl author: ví dụ 1 là ảnh GT nhánh test 2_b1 -> 4 mẫu test_all 2_b1_t1..t4 cùng cảnh với ví dụ "
            f"(như tác giả chạy; số của 4 mẫu này lạc quan)")
    return author_turns(lambda path: encode_image(path, img.get("max_side"), img.get("jpeg_quality", 95)))


def summarize(recs, cfg, n_full, par):
    """Số tổng của các ảnh đã có câu trả lời + ước cho cả tập (`n_full` ảnh) ở mức song song `par`."""
    K = cfg["k"]
    cost = np.array([r["cost"] for r in recs], dtype=float) if recs else np.zeros(0)
    sec = np.array([r["sec"] for r in recs], dtype=float) if recs else np.zeros(0)
    nv = [r["n_valid"] for r in recs]
    out = {"n": len(recs), "n_fail": int(sum(v == 0 for v in nv)), "n_short": int(sum(0 < v < K for v in nv)),
           "mean_valid": float(np.mean(nv)) if nv else 0.0, "cost": float(cost.sum()),
           "tokens_in_mean": float(np.mean([r["in"] for r in recs])) if recs else 0.0,
           "tokens_out_mean": float(np.mean([r["out"] for r in recs])) if recs else 0.0,
           "tokens_reasoning_mean": float(np.mean([r["reasoning"] for r in recs])) if recs else 0.0,
           "attempts_mean": float(np.mean([r["attempts"] for r in recs])) if recs else 0.0}
    if len(cost):
        out.update(cost_per_image_mean=float(cost.mean()), cost_per_image_median=float(np.median(cost)),
                   cost_per_image_p90=float(np.percentile(cost, 90)), sec_per_call_mean=float(sec.mean()),
                   est_full_cost=float(cost.mean() * n_full), est_full_cost_p90=float(np.percentile(cost, 90) * n_full),
                   est_full_hours=float(n_full * sec.mean() / max(par, 1) / 3600))
    out["n_full"] = n_full
    return out


def run_dataset(cfg, ns, backend, out_dir, name, phash, *, pilot=None, max_cost=None, score=True, log=print, spent0=0.0):
    """-> (summary, đã tiêu USD trong lần chạy này). `backend` None = chỉ chấm từ responses.jsonl (`--score-only`)."""
    from baseline.score import load_data, run_score
    from baseline.scoring import read_boxes, write_json, write_jsonl

    tag = f"[{ns.dataset} {name}]"
    samples, index, geo = load_data(ns)
    sel = select(samples, cfg.get("seed", 0), pilot)
    os.makedirs(out_dir, exist_ok=True)
    resp_path = os.path.join(out_dir, "responses.jsonl")
    done = load_responses(resp_path, phash)
    pending = [s for s in sel if s["image_id"] not in done]
    K, retries = cfg["k"], int(cfg.get("retries", 2))
    img_cfg = cfg.get("image") or {}
    # mức song song để ước giờ; vLLM: 1 (`sec` đã chia theo khối)
    par = int(((cfg.get("openrouter") or {}).get("concurrency")) or ((cfg.get("gemini") or {}).get("concurrency")) or 1)
    rpm = float((cfg.get("gemini") or {}).get("rpm") or 0)
    log(f"{tag} {len(sel)} / {len(samples)} mẫu | đã có {len(sel) - len(pending)} (prompt_hash {phash}) | cần hỏi {len(pending)} | "
        f"-> {out_dir}")
    examples = _icl_turns(cfg, ns, log) if pending and backend is not None else []

    spent, t0, last_log, n_new, n_api_err, stopped = 0.0, time.time(), 0.0, 0, 0, None
    chunk = int(cfg.get("chunk") or max(8, 4 * par))
    if pending and backend is None:
        stopped = f"còn {len(pending)} ảnh chưa có câu trả lời (--score-only không gọi model)"
    with open(resp_path, "a") as fout:
        for c0 in range(0, len(pending) if backend is not None else 0, chunk):
            jobs = []
            for s in pending[c0:c0 + chunk]:
                im = encode_image(s["image"], img_cfg.get("max_side"), img_cfg.get("jpeg_quality", 95))
                jobs.append({"image_id": s["image_id"], "messages": build_messages(s, im, cfg, examples), "temperature": None,
                             "sample": s, "im": im, "attempts": 0, "cost": 0.0, "in": 0, "out": 0, "reasoning": 0, "sec": 0.0})
            while jobs:
                try:
                    results = backend.batch([{k: j[k] for k in ("image_id", "messages", "temperature")} for j in jobs])
                except Fatal as e:
                    stopped = f"lỗi không cứu được: {e}"
                    break
                again = []
                for j, res in zip(jobs, results):
                    c = record_cost(res, cfg)
                    spent += c
                    j["attempts"] += 1
                    for k in ("in", "out", "reasoning", "sec"):
                        j[k] += res[k]
                    j["cost"] += c
                    if res["error"] and not res["text"]:
                        n_api_err += 1
                        log(f"{tag} lỗi API {j['image_id']}: {res['error']}")
                        continue
                    boxes = parse_boxes(res["text"], cfg, j["im"]["canvas_wh"], j["im"]["orig_wh"])
                    if not boxes and j["attempts"] <= retries:
                        j["temperature"] = cfg.get("retry_temperature", 0.7)
                        again.append(j)
                        continue
                    n_api_err = 0
                    rec = {"image_id": j["image_id"], "prompt_hash": phash, "model": cfg["model"], "text": res["text"], "boxes": boxes,
                           "n_valid": len(boxes), "attempts": j["attempts"], "in": j["in"], "out": j["out"], "reasoning": j["reasoning"],
                           "cost": j["cost"], "sec": j["sec"], "sent_wh": list(j["im"]["sent_wh"]), "canvas_wh": list(j["im"]["canvas_wh"]),
                           "orig_wh": list(j["im"]["orig_wh"]),
                           "finish": res.get("finish"), "date": time.strftime("%Y-%m-%d %H:%M:%S")}
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    done[j["image_id"]] = rec
                    n_new += 1
                jobs = again
                if stopped:
                    break
            if stopped:
                break
            el = time.time() - t0
            if n_new and (time.time() - last_log >= float(cfg.get("log_sec", 60)) or c0 + chunk >= len(pending)):
                last_log = time.time()
                recs = [done[s["image_id"]] for s in sel if s["image_id"] in done]
                sm = summarize(recs, cfg, len(sel), par)
                left = len(pending) - n_new
                log(f"{tag} {len(recs)}/{len(sel)} | {_fmt_sec(el)} | ETA {_fmt_sec(left * el / n_new)} | ${spent:.4f} lần này "
                    f"(TB ${sm['cost_per_image_mean']:.5f}/ảnh) | vào {_k(sm['tokens_in_mean'])} / ra {_k(sm['tokens_out_mean'])} "
                    f"(nghĩ {_k(sm['tokens_reasoning_mean'])}) tok/ảnh | 0 box {sm['n_fail']} thiếu {sm['n_short']} | lỗi API liên tiếp "
                    f"{n_api_err}")
            if max_cost is not None and spent0 + spent >= max_cost:
                stopped = f"chạm --max-cost ${max_cost:.2f} (đã tiêu ${spent0 + spent:.4f})"
                break
            if n_api_err >= int(cfg.get("max_api_errors", 20)):
                stopped = f"{n_api_err} lỗi API liên tiếp (hết quota / mạng / provider?) — xem log, chạy lại để tiếp"
                break

    el = time.time() - t0
    recs = [done[s["image_id"]] for s in sel if s["image_id"] in done]
    sm = summarize(recs, cfg, len(sel), par)
    sm_full = summarize(recs, cfg, len(samples), par)
    if rpm and recs:                                                   # Gemini: tốc độ bị chặn bởi RPM
        sm_full["est_full_hours"] = max(sm_full["est_full_hours"], len(samples) * float(np.mean([r["attempts"] for r in recs])) / rpm / 60)
    info = {"model": cfg["model"], "config": cfg.get("_path"), "prompt_hash": phash, "dataset": ns.dataset, "mode": name,
            "selected": sm, "full_estimate": {k: sm_full[k] for k in ("n_full", "est_full_cost", "est_full_cost_p90", "est_full_hours")
                                              if k in sm_full},
            "this_run": {"new": n_new, "sec": el, "spent": spent, "stopped": stopped},
            "free_tier": bool((cfg.get("gemini") or {}).get("free_tier")), "date": time.strftime("%Y-%m-%d %H:%M")}
    write_json(os.path.join(out_dir, f"{name}_run.json"), info)
    if recs and "est_full_cost" in sm_full:
        free = " (free tier: thực trả $0)" if info["free_tier"] else ""
        log(f"{tag} xong {len(recs)}/{len(sel)} ảnh, lần này {n_new} ảnh mới trong {_fmt_sec(el)}, ${spent:.4f}{free} | TB "
            f"${sm['cost_per_image_mean']:.5f}/ảnh (p90 ${sm['cost_per_image_p90']:.5f}) | 0 box {sm['n_fail']}, thiếu {sm['n_short']}")
        log(f"{tag} ƯỚC cả tập {len(samples)} ảnh: ${sm_full['est_full_cost']:.2f} (p90 ${sm_full['est_full_cost_p90']:.2f}){free}, "
            f"~{sm_full['est_full_hours']:.1f} giờ ở song song {par}")
    if stopped:
        log(f"{tag} DỪNG: {stopped}")
    if len(recs) < len(sel):
        log(f"{tag} chưa chấm: còn {len(sel) - len(recs)} ảnh chưa có câu trả lời — chạy lại cùng lệnh để tiếp")
        return info, spent
    if not score:
        log(f"{tag} --no-score: chấm sau bằng --score-only (venv chính)")
        return info, spent

    rows = [{"image_id": r["image_id"], "boxes": fill(r["boxes"], K, r["orig_wh"]), "n_valid": r["n_valid"]} for r in recs]
    raw = os.path.join(out_dir, f"{name}_raw.jsonl")
    write_jsonl(raw, rows)
    meta = {"method": "vlm", "model": cfg["model"], "backend": cfg["backend"], "config": cfg.get("_path"), "prompt_hash": phash,
            "raw": raw, "n_fail": sm["n_fail"], "n_short": sm["n_short"], "mean_valid": sm["mean_valid"], "cost": sm["cost"],
            "free_tier": info["free_tier"], "date": time.strftime("%Y-%m-%d %H:%M")}
    for first, suffix in ((None, ""), (1, "_k1")):
        ns.out = os.path.join(out_dir, f"{name}{suffix}.json")
        log(f"{tag} chấm K = {first or K} -> {ns.out}")
        run_score(ns, sel, index, geo, read_boxes(raw, first), {**meta, "first": first}, log=log)
    return info, spent
