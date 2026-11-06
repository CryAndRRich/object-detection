#!/usr/bin/env python3
"""ZETA5 / ZETA6: một VLM trả 30 box chỗ thêm vật / ảnh trên CE-130 `test_all` + CE-CoCount `review5`, chấm bằng code chấm CE-Loc
(`baseline/score.py`). Một config = một model (`config/{openrouter,kaggle,server,server2}/`). Key: `vlm/.env` (mẫu `.env.example`).

  cd /mnt/disk1/aiotlab/haitn/object-detection/vlm
  python run.py --config config/openrouter/gpt-6-luna.yaml --pilot 5 --dry-run       # xem prompt, KHÔNG gọi API
  python run.py --config config/openrouter/gpt-6-luna.yaml --pilot 50                # thử 50 ảnh / tập, in tiền + ước cả tập
  python run.py --config config/openrouter/gpt-6-luna.yaml --all --max-cost 10       # toàn bộ test (dùng lại câu trả lời pilot)
  # vLLM (venv riêng): python ../tools/run_on_free_gpu.py -- run.py --config config/server2/internvl3.5-8b.yaml --pilot 50 --no-score
  #   rồi venv chính:  python run.py --config config/server2/internvl3.5-8b.yaml --pilot 50 --score-only

Output `<out-root>/<name>/{ce130_test_all,cocount_review5}/`: `<pilotK|test>.jsonl` (1 dòng / ảnh: box + score + reason + C-NLL
từng box + token / tiền + text), `<..>.json` (số tổng: `top_score` = CHÍNH, `min_cnll_30` tham khảo; tiền, token, ước cả tập),
`responses.jsonl` (câu trả lời thô ghi dần, để resume).
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.utils.config import load_config  # noqa: E402
from vlm.backends import Fatal, make_backend  # noqa: E402
from vlm.env import load_env  # noqa: E402
from vlm.prompt import build_messages, encode_image, prompt_hash  # noqa: E402
from vlm.runner import dataset_dir, merge_shards, run_dataset, select  # noqa: E402


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
    ap.add_argument("--batch", action="store_true", help="Batch API (chỉ backend openai): giá 50 %%, trả trong <= 24 h; không đo latency")
    ap.add_argument("--dry-run", action="store_true", help="in prompt 2 ảnh đầu, không gọi model, không cần key")
    ap.add_argument("--no-score", action="store_true", help="chỉ sinh câu trả lời (venv vLLM)")
    ap.add_argument("--score-only", action="store_true", help="chỉ chấm từ responses.jsonl, không gọi model")
    ap.add_argument("--coord", default=None, choices=["pixel", "norm1000"],
                    help="đọc toạ độ theo thang này thay `coord` của config (chỉ lúc parse / chấm; file ra thêm hậu tố _<coord>)")
    ap.add_argument("--shard", default=None, help="i/n — nội bộ: bản thứ i trong n bản song song (`vllm.data_parallel`, tự đặt)")
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
            if i == 0 and msgs[0]["role"] == "system":
                log(f"--- SYSTEM\n{msgs[0]['text']}")
                for t in examples:
                    log(f"--- {t['role'].upper()} (few-shot{', có ảnh' if t['image'] else ''})\n{t['text']}")
            log(f"--- USER {s['image_id']} | lớp {s['class']!r} | ảnh gốc = canvas {im['orig_wh']} -> gửi {im['sent_wh']} {im['mime']} "
                f"({len(im['b64']) * 3 // 4 // 1024} KB) | {len(msgs)} lượt{' | ảnh trước text' if msgs[-1].get('image_first') else ''}"
                + ("" if i and msgs[0]["role"] != "system" else f"\n{msgs[-1]['text']}"))


def data_parallel(cfg):
    return int((cfg.get("vllm") or {}).get("data_parallel") or 1) if cfg["backend"] == "vllm" else 1


def launch_shards(cfg, a, argv, datasets, dp, stop_wait=90.0, stagger=60.0):
    """vLLM `data_parallel` n: n tiến trình con `--shard i/n`, mỗi con MỘT GPU (CUDA_VISIBLE_DEVICES thứ i), chia ảnh theo hash
    (`runner.in_shard`), ghi `responses_shard<i>.jsonl`; xong (hay bị dừng: SIGTERM chuyển cho con, quá `stop_wait` giây thì kill) ⇒
    gộp vào `responses.jsonl`. Weight HF tải trước một lần, các con khởi động cách nhau `stagger` giây (không tranh tải / ghi cache
    compile của vLLM cùng lúc). -> mã thoát lớn nhất của các con."""
    gpus = [g.strip() for g in (os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(map(str, range(dp)))).split(",") if g.strip()]
    if len(gpus) < dp:
        raise SystemExit(f"[vlm] data_parallel {dp} cần {dp} GPU, CUDA_VISIBLE_DEVICES chỉ có {gpus}")
    log(f"[vlm] data_parallel {dp}: {dp} bản model, GPU {gpus[:dp]}, chia ảnh theo hash")
    if cfg["backend"] == "vllm":
        from huggingface_hub import snapshot_download
        t0 = time.time()
        snapshot_download(cfg["model"])
        log(f"[vlm] tải weight {cfg['model']} {time.time() - t0:.0f} s")
    procs, stop = [], {"at": None}

    def on_signal(signum, frame):
        if stop["at"] is None:
            stop["at"] = time.time()
            log(f"[vlm] nhận tín hiệu {signum} -> dừng {dp} bản (câu trả lời đã ghi giữ nguyên)")
            for p in procs:
                p.terminate()
    old = {sg: signal.signal(sg, on_signal) for sg in (signal.SIGTERM, signal.SIGINT)}
    try:
        for i in range(dp):
            if i and stagger:
                time.sleep(stagger)
            if stop["at"] is not None:
                break
            procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), *argv, "--shard", f"{i}/{dp}"],
                                          env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i])))
        while any(p.poll() is None for p in procs):
            if stop["at"] is not None and time.time() - stop["at"] > stop_wait:
                for p in procs:
                    p.kill()
            time.sleep(0.5)
    finally:
        for sg, h in old.items():
            signal.signal(sg, h)
        for d in datasets:
            out_dir = os.path.join(a.out_root, cfg["name"], dataset_dir(data_ns(a, d)))
            if os.path.isdir(out_dir):
                log(f"[vlm] {d}: gộp {merge_shards(out_dir)} câu trả lời của {dp} bản -> {out_dir}/responses.jsonl")
    rcs = [p.returncode for p in procs]
    log(f"[vlm] {dp} bản xong, mã thoát {rcs}")
    return max(rcs, key=abs) if rcs else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = parse_args(argv)
    if a.no_score and a.score_only:
        raise SystemExit("--no-score và --score-only loại trừ nhau")
    cfg = load_config(a.config)
    cfg["_path"] = a.config
    phash = prompt_hash(cfg)
    datasets = ["ce130", "cocount"] if a.dataset == "all" else [a.dataset]
    name = f"pilot{a.pilot}" if a.pilot else "test"
    if a.coord:
        cfg["coord"] = a.coord
        name += f"_{a.coord}"
    log(f"[vlm] {time.strftime('%Y-%m-%d %H:%M:%S')} | {socket.gethostname()} | {cfg['name']} = {cfg['model']} ({cfg['backend']}) | "
        f"{name} | tập {datasets} | prompt_hash {phash} | K {cfg['k']} | icl {cfg.get('icl', 0)} | max-cost {a.max_cost} | batch {a.batch}")
    if a.dry_run:
        dry_run(cfg, a, datasets)
        return
    dp = data_parallel(cfg)
    shard = tuple(int(v) for v in a.shard.split("/")) if a.shard else None
    if dp > 1 and shard is None and not a.score_only:
        rc = launch_shards(cfg, a, argv, datasets, dp)
        if rc:
            raise SystemExit(rc)
        return
    if shard:
        name += f"_shard{shard[0]}"
        log(f"[vlm] bản {shard[0]} / {shard[1]} | CUDA_VISIBLE_DEVICES {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    found = load_env()
    log(f"[vlm] .env: {len(found)} biến ({', '.join(found) or 'không có file'})")
    try:
        backend = None if a.score_only else make_backend(cfg, log, batch=a.batch)
    except Fatal as e:
        raise SystemExit(f"[vlm] {e}")
    spent, est, fatal = 0.0, [], False
    for d in datasets:
        ns = data_ns(a, d)
        out_dir = os.path.join(a.out_root, cfg["name"], dataset_dir(ns))
        info, s = run_dataset(cfg, ns, backend, out_dir, name, phash, pilot=a.pilot, max_cost=a.max_cost, score=not a.no_score,
                              log=log, spent0=spent, shard=shard)
        spent += s
        est.append(info["full_estimate"])
        stopped = info["this_run"]["stopped"] or ""
        fatal = stopped.startswith("lỗi không cứu được")
        if fatal or (stopped and a.max_cost is not None and spent >= a.max_cost):
            log(f"[vlm] dừng, không chạy tập sau ({stopped}) — câu trả lời đã có giữ trong responses.jsonl; chạy lại cùng lệnh để tiếp")
            break
    if len(est) == 2 and all("est_full_cost" in e for e in est):
        log(f"[vlm] ƯỚC --all cả hai tập ({sum(e['n_full'] for e in est)} ảnh): ${sum(e['est_full_cost'] for e in est):.2f} "
            f"(p90 ${sum(e['est_full_cost_p90'] for e in est):.2f}), ~{sum(e['est_full_hours'] for e in est):.1f} giờ")
    log(f"[vlm] xong, lần này tiêu ${spent:.4f}")
    if fatal:
        raise SystemExit(3)                 # dừng vì lỗi không cứu được (hết quota / key): mã khác 0 cho vòng lệnh ngoài `|| break`


if __name__ == "__main__":
    main()
