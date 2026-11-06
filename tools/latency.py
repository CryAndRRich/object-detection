#!/usr/bin/env python3
"""Latency 1 ảnh — MỘT định nghĩa chung cho mọi method bài ADD: CE-Loc (DELTA), ZETA1–3 (baseline), ZETA5–6 (VLM). Chốt 2026-11-04.

Latency = thời gian thực (wall-clock) từ lúc đưa MỘT ảnh (+ tên lớp, box vật có sẵn) vào method tới lúc có K = 30 box xyxy pixel
ẢNH GỐC, trung bình trên một tập ảnh cố định:
  - TỪNG ảnh một (batch 1), sau `--warmup` ảnh khởi động / tập KHÁC tập đo (không tính; VLM: tránh cache prefix / ảnh của vLLM);
  - GỒM: đọc + giải mã ảnh, tiền xử lý của method (letterbox / resize / processor VLM / base64), model (forward | khử nhiễu 100 bước |
    sinh token), đưa box về pixel ảnh gốc (VLM: parse JSON). GPU đồng bộ (`torch.cuda.synchronize`) trước khi bấm giờ;
  - KHÔNG gồm: thời gian đĩa (mọi file của tập đo được đọc trước một lần ⇒ nằm trong page cache), nạp model, nhúng CLIP tên lớp
    (CE-Loc / HiddenObjects tính sẵn một lần cho mọi lớp, như bảng tra), chấm C-NLL;
  - tập đo: `--n` ảnh / tập (mặc định 100 CE-130 `test_all` + 100 CE-CoCount `review5`, chốt 2026-11-06) = 100 ảnh đầu theo hash
    seed 0 (= `pilot 100` của VLM, chứa pilot 20); warm-up = các ảnh kế tiếp theo hash;
  - VLM qua API (OpenRouter): cùng cách đo, gồm mạng + hàng đợi provider ⇒ `kind: api`, không so trực tiếp với GPU. Mỗi ảnh
    MỘT lần gọi (không thử lại khi 0 box; `n_valid` ghi kèm). VLM vLLM: config như lúc chạy full (`tensor_parallel_size` ghi kèm);
    RePlan (ZETA4): 1 ảnh = câu greedy + 29 câu sample (đủ 30 box).
Báo CHỈ số TB giây / ảnh (mỗi tập + chung) kèm tên GPU. Ra `<out-root>/<name>.json` (mặc định `output/add/latency/`), có từng ảnh.

  cd /mnt/disk1/aiotlab/haitn/object-detection
  python tools/run_on_free_gpu.py -- tools/latency.py celoc --ckpt weights/add/delta8/snapshots/ep010.pth
  python tools/run_on_free_gpu.py -- tools/latency.py zeta --config baseline/config/zeta3.yaml --ckpt weights/add/zeta3/best.pth
  <venv vllm>/bin/python tools/run_on_free_gpu.py -- tools/latency.py vlm --config vlm/config/server2/qwen3-vl-4b.yaml   # server 2
  python tools/latency.py vlm --config vlm/config/openrouter/glm-5.3-flash.yaml               # API (key trong vlm/.env)
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # object-detection/
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

PICK_SEED = 0                    # = seed chọn pilot của VLM ⇒ tập đo = pilot `--n`
DATASETS = ("ce130", "cocount")
DEFINITION = ("1 ảnh / lần (batch 1), sau warm-up trên ảnh khác; từ đọc + giải mã ảnh (file đã trong page cache) tới 30 box pixel "
              "ảnh gốc: gồm tiền xử lý, model, đọc box; không gồm nạp model, nhúng CLIP tên lớp (tính sẵn), chấm C-NLL; "
              "GPU đồng bộ trước khi bấm giờ; API gồm mạng")


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------- khung đo chung

def pick(samples, n, warmup, seed=PICK_SEED):
    """-> (tập đo = `vlm.runner.select(samples, seed, n)` theo thứ tự gốc, `warmup` ảnh warm-up = các ảnh kế tiếp theo hash)."""
    from vlm.runner import _rank, select
    if len(samples) < n + warmup:
        raise ValueError(f"tập chỉ có {len(samples)} ảnh < n {n} + warmup {warmup}")
    order = sorted(samples, key=lambda s: _rank(seed, s["image_id"]))
    return select(samples, seed, n), order[n:n + warmup]


def measure(run, measured, warm, prefetch=None, sync=None, tag="", clock=time.perf_counter):
    """`run(sample)` -> dict có "boxes" (K box) [+ trường phụ]. -> list hàng {image_id, sec, n_boxes, ...} của tập đo. `prefetch(s)`:
    đọc trước mọi file của ảnh (page cache, không tính giờ); `sync()`: chờ GPU xong. Hàng có "error" (API lỗi) vẫn giữ, không vào TB."""
    sync = sync or (lambda: None)
    for s in list(warm) + list(measured):
        (prefetch or read_bytes)(s)
    for s in warm:
        run(s)
    sync()
    rows, t0 = [], time.time()
    for i, s in enumerate(measured, 1):
        sync()
        t = clock()
        try:
            out = run(s)
            sync()
            sec, err = clock() - t, None
        except RuntimeError as e:                 # API lỗi sau khi backend đã thử lại: ghi, không vào TB
            out, sec, err = {"boxes": []}, clock() - t, str(e)
        rows.append({"image_id": s["image_id"], "sec": sec, "n_boxes": len(out["boxes"]), **{k: v for k, v in out.items() if k != "boxes"},
                     **({"error": err} if err else {})})
        el = time.time() - t0
        log(f"  [{tag} {i:3d}/{len(measured)}] {sec:8.3f} s{'  LỖI ' + err if err else ''} | {el:.0f} s | ETA "
            f"{el / i * (len(measured) - i):.0f} s")
    return rows


def read_bytes(s):
    with open(s["image"], "rb") as f:
        f.read()


def summarize(rows_by_ds):
    """{tập: hàng} -> {"latency_s": TB mọi ảnh không lỗi, "per_dataset": {tập: {n, n_error, latency_s}}}."""
    ok = lambda rs: [r["sec"] for r in rs if "error" not in r]  # noqa: E731
    per = {d: {"n": len(rs), "n_error": len(rs) - len(ok(rs)), "latency_s": float(np.mean(ok(rs))) if ok(rs) else None}
           for d, rs in rows_by_ds.items()}
    allv = [v for rs in rows_by_ds.values() for v in ok(rs)]
    return {"latency_s": float(np.mean(allv)) if allv else None, "per_dataset": per}


def gpu_info():
    """Tên + bộ nhớ dùng + utilization lúc bắt đầu của các GPU trong CUDA_VISIBLE_DEVICES (server dùng chung: job khác làm chậm)."""
    ids = [i.strip() for i in (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",") if i.strip()]
    cmd = ["nvidia-smi", *(["-i", ",".join(ids)] if ids else []), "--query-gpu=name,memory.used,memory.total,utilization.gpu",
           "--format=csv,noheader"]
    try:
        return [line.strip() for line in subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        return []


def data_ns(a, dataset):
    return argparse.Namespace(dataset=dataset, split="test_all", turn_index=a.turn_index, samples_root=a.samples_root,
                              cocount_root=a.cocount_root, cocount_review=a.cocount_review, cocount_min_obj=a.cocount_min_obj, limit=None)


def torch_sync(dev):
    import torch
    return (lambda: torch.cuda.synchronize(dev)) if dev.type == "cuda" else None


def torch_device(a):
    import torch
    return torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))


# ----------------------------------------------------------------------------- từng loại method

class CELoc:
    """CE-Loc DELTA4+: đúng đường `eval.py` (dataset -> collate -> `predict_batch`), 1 ảnh / batch, K = eval.n_samples."""

    def __init__(self, a, log=log):
        from ce_localization.models.box_policy import load_checkpoint
        self.a, self.dev = a, torch_device(a)
        model, self.cfg, self.iter = load_checkpoint(a.ckpt)
        self.model = model.to(self.dev).eval()
        self.amp = bool(self.cfg["training"].get("amp")) and self.dev.type == "cuda"
        self.K, self.S = self.cfg["eval"]["n_samples"], self.cfg["data"]["image_size"]
        self.sync = torch_sync(self.dev)
        self.info = {"ckpt": a.ckpt, "iter": self.iter, "sampler": a.sampler, "amp": self.amp, "K": self.K, "device": str(self.dev)}

    def setup(self, dataset, samples):
        import torch
        import ce_localization.train as train_mod
        from ce_localization.data.cocount import CoCountAddDataset, load_review
        from ce_localization.data.turns import CE130AddDataset, TurnIndex, collate_add, letterbox_size, to_device_add
        from ce_localization.engine.add_eval import predict_batch
        a = self.a
        if dataset == "ce130":
            ds = CE130AddDataset(TurnIndex(a.turn_index), a.ce130_root, a.samples_root, "test_all", self.S)
        else:
            ds = CoCountAddDataset(a.cocount_root, self.S, review=load_review(a.cocount_review), min_obj=a.cocount_min_obj)
        pos = {k: i for i, k in enumerate(ds.keys)}
        tt = train_mod.build_text_table(sorted({s["class"] for s in samples}), self.cfg, self.dev)
        self.model.check_text_fingerprint(tt.token_sha)
        var = [("", {"sampler": a.sampler})]

        def run(s):
            batch = to_device_add(collate_add([ds[pos[s["image_id"]]]]), self.dev)
            gens = {"": torch.Generator(device=self.dev.type).manual_seed(a.seed)}
            boxes = predict_batch(self.model, batch, tt, self.K, gens, var, self.amp)[""][0]
            return {"boxes": boxes / letterbox_size(s["wh"], self.S)[0]}             # pixel canvas -> pixel ảnh gốc

        return run, lambda s: ds[pos[s["image_id"]]]                                # prefetch: đọc mọi file của mẫu


class Zeta:
    """ZETA1–3: đúng `baseline.methods.predict` với danh sách 1 ảnh (HiddenObjects: CLIP text tính sẵn, DataLoader không worker)."""

    def __init__(self, a, log=log):
        from baseline.methods import load_model
        from ce_localization.utils.config import load_config
        self.a, self.dev = a, torch_device(a)
        cfg = load_config(a.config)
        self.cfg = {**cfg, "predict": {**cfg["predict"], "num_workers": 0}}
        self.model = load_model(self.cfg, a.ckpt, self.dev)
        self.sync = torch_sync(self.dev)
        self.info = {"config": a.config, "ckpt": a.ckpt, "baseline": cfg["method"], "K": cfg["predict"]["k"], "device": str(self.dev)}

    def setup(self, dataset, samples):
        from baseline.methods import predict
        kw = {"log_every": 0}
        if self.cfg["method"] == "hiddenobjects":
            from baseline.methods.hiddenobjects import class_texts
            kw["texts"] = class_texts([s["class"] for s in samples], self.model.clip_text, self.dev)
        return (lambda s: {"boxes": predict(self.cfg, self.model, [s], dev=self.dev, seed=self.a.seed, log=None, **kw)[0]["boxes"]},
                None)


class VLM:
    """ZETA5–6: đúng prompt / ảnh / parse của `vlm/run.py`, mỗi ảnh một request riêng (vLLM: `LLM.chat` 1 hội thoại)."""

    def __init__(self, a, log=log):
        from ce_localization.utils.config import load_config
        from vlm.backends import make_backend
        from vlm.env import load_env
        from vlm.prompt import prompt_hash
        self.a = a
        self.cfg = load_config(a.config)
        self.cfg["_path"] = a.config
        load_env()
        self.backend = make_backend(self.cfg, log)
        self.sync = None
        self.info = {"config": a.config, "model": self.cfg["model"], "backend": self.cfg["backend"], "K": self.cfg["k"],
                     "prompt_hash": prompt_hash(self.cfg), "engine": self.cfg.get("engine"), "max_tokens": self.cfg.get("max_tokens"),
                     "data_parallel": (self.cfg.get("vllm") or {}).get("data_parallel")}            # latency = MỘT bản (một GPU)

    def setup(self, dataset, samples):
        from vlm.parse import fill, parse_scored
        from vlm.prompt import build_messages, encode_image
        from vlm.runner import _icl_turns, record_cost
        cfg, img = self.cfg, self.cfg.get("image") or {}
        examples = _icl_turns(cfg, argparse.Namespace(dataset=dataset), log)        # lượt few-shot: một lần (như lúc chạy full)

        def run(s):
            im = encode_image(s["image"], img.get("max_side"), img.get("jpeg_quality", 95))
            res = self.backend.batch([{"image_id": s["image_id"], "messages": build_messages(s, im, cfg, examples), "temperature": None}])[0]
            if res["error"] and not res["text"]:
                raise RuntimeError(str(res["error"]))
            boxes, _ = parse_scored(res["text"], cfg, im["canvas_wh"], im["orig_wh"])
            return {"boxes": fill(boxes, cfg["k"], im["orig_wh"]), "n_valid": len(boxes), "out_tokens": res["out"],
                    "finish": res.get("finish"), "cost": record_cost(res, cfg)}

        return run, None


METHODS = {"celoc": CELoc, "zeta": Zeta, "vlm": VLM}


def default_name(a):
    if a.method == "celoc":
        d, stem = os.path.split(os.path.splitext(os.path.abspath(a.ckpt))[0])
        run = os.path.basename(os.path.dirname(d)) if os.path.basename(d) == "snapshots" else os.path.basename(d)
        return f"{run}_{stem}"
    if a.method == "zeta":
        return f"{os.path.splitext(os.path.basename(a.config))[0]}_{os.path.splitext(os.path.basename(a.ckpt))[0]}"
    from ce_localization.utils.config import load_config
    return load_config(a.config)["name"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="method", required=True)
    for m in METHODS:
        p = sub.add_parser(m)
        if m in ("celoc", "zeta"):
            p.add_argument("--ckpt", required=True)
            p.add_argument("--device", default=None)
        if m in ("zeta", "vlm"):
            p.add_argument("--config", required=True)
        if m == "celoc":
            p.add_argument("--sampler", default="mock", choices=["mock", "ddim"])
            p.add_argument("--ce130-root", default=os.path.join(ROOT, "data", "all_phase2_V2"))
        p.add_argument("--name", default=None, help="tên file ra (mặc định: celoc <run>_<ckpt>, zeta <config>_<ckpt>, vlm cfg name)")
        p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
        p.add_argument("--n", type=int, default=100, help="số ảnh đo / tập (= pilot 100 của VLM, chốt 2026-11-06)")
        p.add_argument("--warmup", type=int, default=3, help="số ảnh warm-up / tập (khác tập đo, không tính)")
        p.add_argument("--seed", type=int, default=0, help="seed nhiễu của model (CE-Loc / CSANet), như eval")
        p.add_argument("--out-root", default=os.path.join(ROOT, "..", "output", "add", "latency"))
        p.add_argument("--turn-index", default=os.path.join(ROOT, "data", "turn_index.json"))
        p.add_argument("--samples-root", default=os.path.join(ROOT, "data", "samples"))
        p.add_argument("--cocount-root", default=os.path.join(ROOT, "data", "cocount"))
        p.add_argument("--cocount-review", default=os.path.join(ROOT, "data", "cocount_review.jsonl"))
        p.add_argument("--cocount-min-obj", type=int, default=5)
    return ap.parse_args(argv)


def main(argv=None):
    from baseline.score import load_data
    a = parse_args(argv)
    name = a.name or default_name(a) + ("" if len(a.datasets) == len(DATASETS) else "_" + "_".join(a.datasets))
    out = os.path.join(a.out_root, f"{name}.json")
    t0 = time.time()
    gpus = gpu_info()
    log(f"[latency] {time.strftime('%Y-%m-%d %H:%M:%S')} | {socket.gethostname()} | {a.method} {name} | {a.n} ảnh đo + {a.warmup} "
        f"warm-up / tập {a.datasets} | GPU {gpus or '-'}")
    m = METHODS[a.method](a)
    log(f"[latency] nạp xong {time.time() - t0:.0f} s | {m.info}")
    rows = {}
    for d in a.datasets:
        samples, _, _ = load_data(data_ns(a, d))
        measured, warm = pick(samples, a.n, a.warmup)
        run, prefetch = m.setup(d, measured + warm)
        log(f"[latency] {d}: đo {len(measured)} / {len(samples)} ảnh, warm-up {len(warm)}")
        rows[d] = [{"dataset": d, **r} for r in measure(run, measured, warm, prefetch, m.sync, tag=d)]
    sm = summarize(rows)
    kind = "api" if a.method == "vlm" and m.cfg["backend"] in ("openai", "openrouter") else "local"
    res = {"name": name, "method": a.method, "kind": kind, "definition": DEFINITION, **sm, "gpu": gpus if kind == "local" else None,
           "host": socket.gethostname(), "date": time.strftime("%Y-%m-%d %H:%M"), "n_per_dataset": a.n, "warmup": a.warmup,
           "pick_seed": PICK_SEED, **m.info, "images": [r for d in a.datasets for r in rows[d]]}
    os.makedirs(a.out_root, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    f4 = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
    for d, v in sm["per_dataset"].items():
        log(f"[latency] {d:8s} {v['n']:3d} ảnh | TB {f4(v['latency_s'])} s / ảnh" + (f" | lỗi {v['n_error']}" if v["n_error"] else ""))
    log(f"[latency] CHUNG TB {f4(sm['latency_s'])} s / ảnh ({kind}) -> {out} | {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
