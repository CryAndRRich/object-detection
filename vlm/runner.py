"""Chạy một VLM trên một tập test (CE-130 `test_all` | CE-CoCount `review5`): chọn mẫu (pilot K ảnh | all), resume, gọi backend theo
khối, parse, ghi từng câu trả lời ngay, tiến độ + ETA + tiền, ước tiền / giờ cho cả tập, rồi chấm bằng đúng code chấm ZETA
(`baseline/scoring.py`). Model trả 30 box kèm `score`; **kết quả chính ZETA5 / 6 = C-NLL của box `score` cao nhất** (người dùng chốt
2026-11-04); tham khảo: box C-NLL thấp nhất trong 30 (như ZETA1–3). Mỗi tập 3 file: `responses.jsonl` (ghi dần lúc chạy, để resume),
`<name>.jsonl` (1 dòng / ảnh, mọi thông tin), `<name>.json` (số tổng cả hai luật + tiền / token / thời gian / ước cả tập). Chấm parse
lại từ `text` (câu trả lời cũ chấm theo cùng luật). Không box nào có score (regex / JSON không `candidates`) ⇒ box đầu model trả,
đếm `n_noscore`.

Pilot = K mẫu đầu theo hash blake2b(`seed:image_id`) ⇒ ổn định, và là tập con của all ⇒ chạy all dùng lại câu trả lời pilot.
`responses.jsonl` (một dòng / ảnh, có `prompt_hash`): resume bỏ ảnh đã xong cùng hash; đổi prompt / model / sampling ⇒ hỏi lại.
Lỗi API (sau khi backend đã thử lại) KHÔNG ghi ⇒ chạy lại sẽ hỏi lại ảnh đó; model trả lời mà 0 box hợp lệ sau `retries` lần ⇒ ghi
(`n_valid` = 0, chấm bằng box cả ảnh)."""

import json
import os
import time

import numpy as np

from ce_localization.data.turns import iter_jsonl
from ce_localization.utils.log import fmt_time
from vlm.backends import Fatal, stable_int, token_cost
from vlm.parse import fill, parse_candidates, parse_scored
from vlm.prompt import author_turns, build_messages, encode_image

def dataset_dir(ns):
    return f"ce130_{ns.split}" if ns.dataset == "ce130" else f"cocount_review{ns.cocount_min_obj}"


def _rank(seed, image_id):
    return stable_int(f"{seed}:{image_id}", 8)


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
    """USD của một lần gọi: `cost` backend trả (OpenRouter) hay token × giá config (`price.in` / `price.out` USD / 1M; token vào đã
    cache — `cached`, OpenAI — theo `price.cached`, không có thì như token vào)."""
    if res.get("cost") is not None:
        return float(res["cost"])
    return token_cost(res, cfg.get("price"))


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
    """-> (summary, đã tiêu USD trong lần chạy này). `backend` None = chỉ chấm từ responses.jsonl (`--score-only`). Latency 1 ảnh
    KHÔNG đo ở đây (`sec` = thông lượng) — dùng `tools/latency.py`."""
    from baseline.score import load_data

    tag = f"[{ns.dataset} {name}]"
    samples, index, geo = load_data(ns)
    sel = select(samples, cfg.get("seed", 0), pilot)
    os.makedirs(out_dir, exist_ok=True)
    resp_path = os.path.join(out_dir, "responses.jsonl")
    done = load_responses(resp_path, phash)
    pending = [s for s in sel if s["image_id"] not in done]
    retries = int(cfg.get("retries", 2))
    img_cfg = cfg.get("image") or {}
    # mức song song để ước giờ; vLLM: 1 (`sec` đã chia theo khối)
    par = int((cfg.get("openrouter") or {}).get("concurrency") or 1)
    log(f"{tag} {len(sel)} / {len(samples)} mẫu | đã có {len(sel) - len(pending)} (prompt_hash {phash}) | cần hỏi {len(pending)} | "
        f"-> {out_dir}")
    examples = _icl_turns(cfg, ns, log) if pending and backend is not None else []

    spent, t0, last_log, n_new, n_api_err, stopped = 0.0, time.time(), 0.0, 0, 0, None
    chunk = int(cfg.get("chunk") or max(8, 4 * par))
    if pending and backend is None:
        stopped = f"còn {len(pending)} ảnh chưa có câu trả lời (--score-only không gọi model)"
    c0 = 0
    with open(resp_path, "a") as fout:
        while backend is not None and c0 < len(pending):
            c1 = min(c0 + chunk, len(pending))
            jobs = []
            for s in pending[c0:c1]:
                im = encode_image(s["image"], img_cfg.get("max_side"), img_cfg.get("jpeg_quality", 95))
                jobs.append({"image_id": s["image_id"], "messages": build_messages(s, im, cfg, examples), "temperature": None,
                             "sample": s, "im": im, "attempts": 0, "cost": 0.0, "in": 0, "out": 0, "reasoning": 0, "cached": 0, "sec": 0.0,
                             "latency": None})
            while jobs:
                try:
                    results = backend.batch([{"image_id": j["image_id"], "messages": j["messages"], "temperature": j["temperature"],
                                              "attempt": j["attempts"]} for j in jobs])
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
                    j["cached"] += int(res.get("cached") or 0)                 # token vào trúng cache
                    j["cost"] += c
                    j["latency"] = res.get("latency")                 # của lần gọi có câu trả lời được giữ
                    if res["error"] and not res["text"]:
                        n_api_err += 1
                        log(f"{tag} lỗi API {j['image_id']}: {res['error']}")
                        continue
                    boxes, scores = parse_scored(res["text"], cfg, j["im"]["canvas_wh"], j["im"]["orig_wh"])
                    if not boxes and j["attempts"] <= retries:
                        j["temperature"] = cfg.get("retry_temperature", 0.7)
                        again.append(j)
                        continue
                    n_api_err = 0
                    rec = {"image_id": j["image_id"], "prompt_hash": phash, "model": cfg["model"], "text": res["text"], "boxes": boxes,
                           "scores": scores, "n_valid": len(boxes), "attempts": j["attempts"], "in": j["in"], "out": j["out"], "reasoning": j["reasoning"], "cached": j["cached"],
                           "cost": j["cost"], "sec": j["sec"],
                           "latency": j["latency"], "gen_id": res.get("gen_id"), "provider": res.get("provider"),
                           "sent_wh": list(j["im"]["sent_wh"]), "canvas_wh": list(j["im"]["canvas_wh"]),
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
            if n_new and (time.time() - last_log >= float(cfg.get("log_sec", 60)) or c1 >= len(pending)):
                last_log = time.time()
                recs = [done[s["image_id"]] for s in sel if s["image_id"] in done]
                sm = summarize(recs, cfg, len(sel), par)
                left = len(pending) - n_new
                log(f"{tag} {len(recs)}/{len(sel)} | {fmt_time(el)} | ETA {fmt_time(left * el / n_new)} | ${spent:.4f} lần này "
                    f"(TB ${sm['cost_per_image_mean']:.5f}/ảnh) | vào {_k(sm['tokens_in_mean'])} / ra {_k(sm['tokens_out_mean'])} "
                    f"(nghĩ {_k(sm['tokens_reasoning_mean'])}) tok/ảnh | "
                    f"0 box {sm['n_fail']} thiếu {sm['n_short']} | lỗi API liên tiếp {n_api_err}")
            if max_cost is not None and spent0 + spent >= max_cost:
                stopped = f"chạm --max-cost ${max_cost:.2f} (đã tiêu ${spent0 + spent:.4f})"
                break
            if n_api_err >= int(cfg.get("max_api_errors", 20)):
                stopped = f"{n_api_err} lỗi API liên tiếp (hết quota / mạng / provider?) — xem log, chạy lại để tiếp"
                break
            c0 = c1

    el = time.time() - t0
    recs = [done[s["image_id"]] for s in sel if s["image_id"] in done]
    sm = summarize(recs, cfg, len(sel), par)
    sm_full = summarize(recs, cfg, len(samples), par)
    info = {"model": cfg["model"], "backend": cfg["backend"], "config": cfg.get("_path"), "prompt_hash": phash, "dataset": ns.dataset,
            "mode": name, "selected": sm,
            "full_estimate": {k: sm_full[k] for k in ("n_full", "est_full_cost", "est_full_cost_p90", "est_full_hours") if k in sm_full},
            "this_run": {"new": n_new, "sec": el, "spent": spent, "stopped": stopped},
            "date": time.strftime("%Y-%m-%d %H:%M")}
    if recs and "est_full_cost" in sm_full:
        log(f"{tag} xong {len(recs)}/{len(sel)} ảnh, lần này {n_new} ảnh mới trong {fmt_time(el)}, ${spent:.4f} | TB "
            f"${sm['cost_per_image_mean']:.5f}/ảnh (p90 ${sm['cost_per_image_p90']:.5f}) | 0 box {sm['n_fail']}, thiếu {sm['n_short']}")
        log(f"{tag} ƯỚC cả tập {len(samples)} ảnh: ${sm_full['est_full_cost']:.2f} (p90 ${sm_full['est_full_cost_p90']:.2f}), "
            f"~{sm_full['est_full_hours']:.1f} giờ ở song song {par}")
    if stopped:
        log(f"{tag} DỪNG: {stopped}")
    summary_path = os.path.join(out_dir, f"{name}.json")
    if len(recs) < len(sel) or not score:
        write_summary(summary_path, info)
        log(f"{tag} chưa chấm: còn {len(sel) - len(recs)} ảnh chưa có câu trả lời — chạy lại cùng lệnh để tiếp" if len(recs) < len(sel)
            else f"{tag} --no-score: chấm sau bằng --score-only (venv chính)")
        return info, spent
    score_all(recs, sel, index, geo, cfg, ns, info, out_dir, name, tag, log)
    return info, spent


def write_summary(path, info, scores=None):
    """`<name>.json` = tiền / token / thời gian / ước cả tập (+ kết quả chấm khi đủ câu trả lời)."""
    from ce_localization.engine.add_eval import write_json
    write_json(path, {**info, **(scores or {})})


SHOW = ("cnll", "cnll_gt", "on_object", "in_image", "degenerate")


def score_all(recs, sel, index, geo, cfg, ns, info, out_dir, name, tag, log=print):
    """Chấm bằng code chấm ZETA rồi ghi 2 file: `<name>.jsonl` (1 dòng / ảnh: câu trả lời + mọi box kèm score, reason, C-NLL…) và
    `<name>.json` (số tổng: `top_score` = CHÍNH, box score cao nhất; `min_cnll_K` = tham khảo, box C-NLL thấp nhất trong K).
    Parse lại từ `text` ⇒ câu trả lời cũ chấm theo cùng luật."""
    from baseline.scoring import records_from_boxes, score_records, write_jsonl

    K = cfg["k"]
    by_id = {r["image_id"]: r for r in recs}
    cands = {i: parse_candidates(r["text"], cfg, r["canvas_wh"], r["orig_wh"]) for i, r in by_id.items()}
    boxes = {i: fill([c["box"] for c in cands[i]], K, r["orig_wh"]) for i, r in by_id.items()}
    n_noscore = sum(bool(cs) and cs[0]["score"] is None for cs in cands.values())
    if n_noscore:
        log(f"{tag} ⚠️ {n_noscore} ảnh không có score -> lấy box đầu model trả")
    obj_size = ns.obj_size if ns.dataset == "cocount" else None
    out, per_box = {}, {}
    for sel_name, first in (("top_score", 1), (f"min_cnll_{K}", None)):
        recs_c, k = records_from_boxes(sel, {i: b[:first] if first else b for i, b in boxes.items()}, geo)
        results, prior, dump = score_records(recs_c, ns.dataset, k, index=index, obj_size=obj_size, seed=ns.seed)
        out[sel_name] = {"K": k, "results": results, "prior": prior}
        if first is None:
            per_box = {key: {row["image_id"]: row for row in rows} for key, rows in dump.items()}
        for key, res in results.items():
            pri = prior.get(key, {})
            log(f"{tag} {sel_name} {key} ({res['n']} mẫu; C-NLL trên {res['n_cnll']} mẫu >= 5 vật): "
                + " | ".join(f"{m} {res[m]:.4f} (prior {pri.get(m, float('nan')):.4f})" for m in SHOW if m in res))

    main_key = next(iter(per_box))
    rows = []
    for s in sel:
        i = s["image_id"]
        r, cs, d = by_id[i], cands[i], per_box[main_key][i]
        box_rows = cs if cs else [{"box": boxes[i][0], "score": None, "fallback": "full_image"}]
        for j, c in enumerate(box_rows):
            c["box_512"] = list(d["boxes"][j])                          # canvas letterbox 512 của code chấm
            for key, rows_k in per_box.items():
                cn = rows_k[i].get("cnll")
                if cn is not None:
                    c["cnll" + key[len(main_key):]] = cn[j]
        cnll = [c["cnll"] for c in box_rows if "cnll" in c]
        rows.append({"image_id": i, "class": s.get("class"), "t": d.get("t"), "orig_wh": r["orig_wh"], "n_obj": d.get("n_obj"),
                     "n_valid": len(cs), "cnll_top": cnll[0] if cnll else None, "cnll_min": min(cnll) if cnll else None,
                     "cnll_gt": d.get("cnll_gt"), "boxes": box_rows,
                     **{k: r.get(k) for k in ("attempts", "in", "out", "reasoning", "cost", "sec")}, "request_sec": r.get("latency"),
                     **{k: r.get(k) for k in ("finish", "provider", "gen_id", "date", "text")}})
    jl = os.path.join(out_dir, f"{name}.jsonl")
    write_jsonl(jl, rows)
    write_summary(os.path.join(out_dir, f"{name}.json"), info,
                  {"main": "top_score", "n_noscore": n_noscore, "dataset_meta": _dataset_meta(ns), **out})
    log(f"{tag} -> {jl} (1 dòng / ảnh) + {name}.json (số tổng)")


def _dataset_meta(ns):
    return {"split": ns.split if ns.dataset == "ce130" else None, "seed": ns.seed,
            **({"cocount_review": ns.cocount_review, "cocount_min_obj": ns.cocount_min_obj} if ns.dataset == "cocount" else {})}
