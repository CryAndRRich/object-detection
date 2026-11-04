#!/usr/bin/env python3
"""ZETA5 / ZETA6: một VLM trả 30 box chỗ thêm vật / ảnh trên CE-130 `test_all` + CE-CoCount `review5`, chấm bằng code chấm CE-Loc
(`baseline/score.py`). Một config = một model (`config/{openrouter,server,gemini}/`). Key: `vlm/.env` (mẫu `.env.example`).

  cd /mnt/disk1/aiotlab/haitn/object-detection/vlm
  python run.py --config config/openrouter/gpt-6-luna.yaml --pilot 5 --dry-run       # xem prompt, KHÔNG gọi API
  python run.py --config config/openrouter/gpt-6-luna.yaml --pilot 50                # thử 50 ảnh / tập, in tiền + ước cả tập
  python run.py --config config/openrouter/gpt-6-luna.yaml --all --max-cost 10       # toàn bộ test (dùng lại câu trả lời pilot)
  # vLLM (venv riêng): python ../tools/run_on_free_gpu.py -- run.py --config config/server/internvl3.5-8b.yaml --pilot 50 --no-score
  #   rồi venv chính:  python run.py --config config/server/internvl3.5-8b.yaml --pilot 50 --score-only

Output `<out-root>/<name>/{ce130_test_all,cocount_review5}/`: `<pilotK|test>.jsonl` (1 dòng / ảnh: box + score + reason + C-NLL
từng box + token / tiền + text), `<..>.json` (số tổng: `top_score` = CHÍNH, `min_cnll_30` tham khảo; tiền, token, ước cả tập),
`responses.jsonl` (câu trả lời thô ghi dần, để resume).
"""

import argparse
import os
import socket
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.utils.config import load_config  # noqa: E402
from vlm.backends import Fatal, make_backend  # noqa: E402
from vlm.env import load_env  # noqa: E402
from vlm.prompt import build_messages, encode_image, prompt_hash  # noqa: E402
from vlm.runner import FULL_SIZES, dataset_dir, run_dataset, select  # noqa: E402


def log(*a):
    print(*a, flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", type=int, help="K ảnh / tập (cố định theo seed, là tập con của --all)")
    mode.add_argument("--all", action="store_true", help="toàn bộ tập test")
    ap.add_argument("--dataset", default="all", choices=["ce130", "cocount", "all"])
    ap.add_argument("--max-cost", type=float, default=None, help="USD tối đa cho lần chạy này (cả hai tập); chạm thì dừng êm")
    ap.add_argument("--dry-run", action="store_true", help="in prompt 2 ảnh đầu, không gọi model, không cần key")
    ap.add_argument("--no-score", action="store_true", help="chỉ sinh câu trả lời (venv vLLM)")
    ap.add_argument("--score-only", action="store_true", help="chỉ chấm từ responses.jsonl, không gọi model")
    ap.add_argument("--out-root", default="../../output/add/vlm")
    ap.add_argument("--split", default="test_all", choices=["test", "test_all", "val"])
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--samples-root", default="../data/samples")
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--cocount-review", default="../data/cocount_review.jsonl")
    ap.add_argument("--cocount-min-obj", type=int, default=5)
    ap.add_argument("--obj-size", nargs="+", default=["mean", "exemplar"], choices=["mean", "exemplar"])
    ap.add_argument("--seed", type=int, default=0, help="seed mốc prior khi chấm (0 như eval CE-Loc)")
    return ap.parse_args(argv)


def data_ns(a, dataset):
    return argparse.Namespace(dataset=dataset, split=a.split, turn_index=a.turn_index, samples_root=a.samples_root,
                              cocount_root=a.cocount_root, cocount_review=a.cocount_review, cocount_min_obj=a.cocount_min_obj,
                              obj_size=a.obj_size, seed=a.seed, limit=None, out=None)


def dry_run(cfg, a, datasets):
    from baseline.score import load_data
    from vlm.runner import _icl_turns
    img = cfg.get("image") or {}
    for d in datasets:
        ns = data_ns(a, d)
        samples, _, _ = load_data(ns)
        sel = select(samples, cfg.get("seed", 0), a.pilot)
        examples = _icl_turns(cfg, ns, log)
        log(f"\n===== {d}: {len(sel)} / {len(samples)} mẫu -> {os.path.join(a.out_root, cfg['name'], dataset_dir(ns))} | few-shot "
            f"{len(examples)} lượt")
        for i, s in enumerate(sel[:2]):
            im = encode_image(s["image"], img.get("max_side"), img.get("jpeg_quality", 95))
            msgs = build_messages(s, im, cfg, examples)
            if i == 0:
                log(f"--- SYSTEM\n{msgs[0]['text']}")
                for t in examples:
                    log(f"--- {t['role'].upper()} (few-shot{', có ảnh' if t['image'] else ''})\n{t['text']}")
            log(f"--- USER {s['image_id']} | lớp {s['class']!r} | ảnh gốc = canvas {im['orig_wh']} -> gửi {im['sent_wh']} {im['mime']} "
                f"({len(im['b64']) * 3 // 4 // 1024} KB) | {len(msgs)} lượt\n{msgs[-1]['text']}")


def main(argv=None):
    a = parse_args(argv)
    if a.no_score and a.score_only:
        raise SystemExit("--no-score và --score-only loại trừ nhau")
    cfg = load_config(a.config)
    cfg["_path"] = a.config
    phash = prompt_hash(cfg)
    datasets = ["ce130", "cocount"] if a.dataset == "all" else [a.dataset]
    name = f"pilot{a.pilot}" if a.pilot else "test"
    log(f"[vlm] {time.strftime('%Y-%m-%d %H:%M:%S')} | {socket.gethostname()} | {cfg['name']} = {cfg['model']} ({cfg['backend']}) | "
        f"{name} | tập {datasets} | prompt_hash {phash} | K {cfg['k']} | icl {cfg.get('icl', 0)} | max-cost {a.max_cost}")
    if a.dry_run:
        dry_run(cfg, a, datasets)
        return
    found = load_env()
    log(f"[vlm] .env: {len(found)} biến ({', '.join(found) or 'không có file'})")
    try:
        backend = None if a.score_only else make_backend(cfg, log)
    except Fatal as e:
        raise SystemExit(f"[vlm] {e}")
    spent, est = 0.0, []
    for d in datasets:
        ns = data_ns(a, d)
        out_dir = os.path.join(a.out_root, cfg["name"], dataset_dir(ns))
        info, s = run_dataset(cfg, ns, backend, out_dir, name, phash, pilot=a.pilot, max_cost=a.max_cost, score=not a.no_score,
                              log=log, spent0=spent)
        spent += s
        est.append(info["full_estimate"])
        if info["this_run"]["stopped"] and a.max_cost is not None and spent >= a.max_cost:
            break
    if len(est) == 2 and all("est_full_cost" in e for e in est):
        log(f"[vlm] ƯỚC --all cả hai tập ({sum(e['n_full'] for e in est)} ảnh): ${sum(e['est_full_cost'] for e in est):.2f} "
            f"(p90 ${sum(e['est_full_cost_p90'] for e in est):.2f}), ~{sum(e['est_full_hours'] for e in est):.1f} giờ")
    elif len(datasets) == 1:
        log(f"[vlm] (cả hai tập: {sum(FULL_SIZES.values())} ảnh)")
    log(f"[vlm] xong, lần này tiêu ${spent:.4f}")


if __name__ == "__main__":
    main()
