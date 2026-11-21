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
  python tools/latency.py zeta --config baseline/config/zeta7_1.yaml --criterion euclid_px-free-k1   # OracleLoc: CPU 1 tiến trình
  <venv vllm>/bin/python tools/run_on_free_gpu.py -- tools/latency.py vlm --config vlm/config/server2/qwen3-vl-4b.yaml   # server 2
  python tools/latency.py vlm --config vlm/config/openrouter/glm-5.3-flash.yaml               # API (key trong vlm/.env)

Bộ nhớ GPU lúc suy luận (`--mem`, chốt 2026-11-21; ra `<out-root>/mem/<name>.json`, KHÔNG đè JSON latency — luồng lấy mẫu làm giờ lệch
nhẹ ⇒ số giây của lần `--mem` không báo). Cùng tập ảnh, batch 1:
  - `peak_proc_gib` (SỐ CHÍNH, chung cho torch + vLLM): đỉnh bộ nhớ GPU của CẢ cây tiến trình (vLLM: EngineCore là tiến trình con) theo
    `nvidia-smi --query-compute-apps`, lấy mẫu nền mỗi `--mem-poll` s + một lần sau mỗi ảnh; gồm CUDA context + cache allocator;
    không gồm job người khác trên cùng GPU;
  - `load_gib`: như trên, ngay sau nạp model, trước warm-up (≈ trọng số + context; vLLM đã giữ sẵn KV + CUDA graph);
  - `peak_alloc_gib` / `peak_reserved_gib` (chỉ method torch): `torch.cuda.max_memory_{allocated,reserved}` trên các ảnh đo
    (reset trước mỗi ảnh ⇒ biết ảnh nào gây đỉnh: `peak_*_image`).
  VLM vLLM giữ trước KV ⇒ số đo = mức config: đo với KV / số chuỗi vừa đủ batch 1, đổi khoá ngoài `prompt_hash` bằng
  `--vllm-set vllm.kv_cache_gib=2 engine.max_num_seqs=1` (KV thiếu cho MỘT chuỗi `max_model_len` thì vLLM tự báo lỗi lúc khởi động).
  <venv vllm>/bin/python tools/run_on_free_gpu.py -- tools/latency.py vlm --config vlm/config/server2/qwen3-vl-4b-sft.yaml --mem \
      --vllm-set vllm.kv_cache_gib=2 engine.max_num_seqs=1
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))           # object-detection/
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from ce_localization.utils.log import log  # noqa: E402

PICK_SEED = 0                    # = seed chọn pilot của VLM ⇒ tập đo = pilot `--n`
DATASETS = ("ce130", "cocount")
DEFINITION = ("1 ảnh / lần (batch 1), sau warm-up trên ảnh khác; từ đọc + giải mã ảnh (file đã trong page cache) tới 30 box pixel "
              "ảnh gốc: gồm tiền xử lý, model, đọc box; không gồm nạp model, nhúng CLIP tên lớp (tính sẵn), chấm C-NLL; "
              "GPU đồng bộ trước khi bấm giờ; API gồm mạng")


# ----------------------------------------------------------------------------- khung đo chung

def pick(samples, n, warmup, seed=PICK_SEED):
    """-> (tập đo = `vlm.runner.select(samples, seed, n)` theo thứ tự gốc, `warmup` ảnh warm-up = các ảnh kế tiếp theo hash)."""
    from vlm.runner import _rank, select
    if len(samples) < n + warmup:
        raise ValueError(f"tập chỉ có {len(samples)} ảnh < n {n} + warmup {warmup}")
    order = sorted(samples, key=lambda s: _rank(seed, s["image_id"]))
    return select(samples, seed, n), order[n:n + warmup]


def measure(run, measured, warm, prefetch=None, sync=None, tag="", clock=time.perf_counter, mem=None):
    """`run(sample)` -> dict có "boxes" (K box) [+ trường phụ]. -> list hàng {image_id, sec, n_boxes, ...} của tập đo. `prefetch(s)`:
    đọc trước mọi file của ảnh (page cache, không tính giờ); `sync()`: chờ GPU xong. Hàng có "error" (API lỗi) vẫn giữ, không vào TB.
    `mem` (GpuMem): `begin()` trước / `end()` sau mỗi ảnh đo (ngoài khoảng bấm giờ) -> trường bộ nhớ của hàng."""
    sync = sync or (lambda: None)
    for s in list(warm) + list(measured):
        (prefetch or read_bytes)(s)
    for s in warm:
        run(s)
    sync()
    rows, t0 = [], time.time()
    for i, s in enumerate(measured, 1):
        sync()
        if mem:
            mem.begin()
        t = clock()
        try:
            out = run(s)
            sync()
            sec, err = clock() - t, None
        except RuntimeError as e:                 # API lỗi sau khi backend đã thử lại: ghi, không vào TB
            out, sec, err = {"boxes": []}, clock() - t, str(e)
        rows.append({"image_id": s["image_id"], "sec": sec, "n_boxes": len(out["boxes"]), **{k: v for k, v in out.items() if k != "boxes"},
                     **({"error": err} if err else {}), **(mem.end() if mem else {})})
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


# ----------------------------------------------------------------------------- bộ nhớ GPU (`--mem`)

GIB = 2 ** 30
MEM_DEFINITION = ("batch 1, cùng tập ảnh latency; peak_proc = đỉnh bộ nhớ GPU của cả cây tiến trình (nvidia-smi compute-apps, lấy mẫu "
                  "nền + sau mỗi ảnh; gồm CUDA context + cache allocator, không gồm job khác); load = ngay sau nạp model; "
                  "peak_alloc / peak_reserved = torch.cuda.max_memory_* (chỉ method torch); vLLM: KV theo config đo")


def query_apps():
    """-> {pid: MiB} mọi tiến trình compute trên mọi GPU (nvidia-smi); lỗi -> {}."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    apps = {}
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            try:
                apps[int(parts[0])] = apps.get(int(parts[0]), 0.0) + float(parts[1])
            except ValueError:                                # "[N/A]"
                pass
    return apps


def proc_tree(root, ps_out=None):
    """-> {root} ∪ mọi tiến trình con cháu (theo `ps -A -o pid=,ppid=`)."""
    if ps_out is None:
        try:
            ps_out = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, check=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return {root}
    kids = {}
    for line in ps_out.splitlines():
        p = line.split()
        if len(p) == 2 and p[0].isdigit() and p[1].isdigit():
            kids.setdefault(int(p[1]), []).append(int(p[0]))
    tree, stack = {root}, [root]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c not in tree:
                tree.add(c)
                stack.append(c)
    return tree


class GpuMem:
    """Đỉnh bộ nhớ GPU của cây tiến trình này (+ torch max_memory_* nếu `dev` là cuda). `start()` mở luồng lấy mẫu nền mỗi `poll` s;
    `begin()` / `end()` bao một ảnh -> {proc_gib, [alloc_gib, reserved_gib]}; `load()` = mẫu ngay (sau nạp model)."""

    def __init__(self, dev=None, poll=0.2, query=None, tree=None, root=None, tree_every=10):
        self.dev = dev if dev is not None and getattr(dev, "type", None) == "cuda" else None
        self.query, self.tree = query or query_apps, tree or proc_tree
        self.poll, self.root, self.tree_every = poll, root or os.getpid(), tree_every
        self.pids, self.n, self.win, self.seen = None, 0, 0.0, False
        self.lock, self.stop_ev, self.th = threading.Lock(), threading.Event(), None

    def sample(self):
        """Một mẫu: MiB của cây tiến trình; cập nhật đỉnh cửa sổ hiện tại."""
        if self.pids is None or self.n % self.tree_every == 0:          # tiến trình con (vLLM EngineCore) có thể sinh sau
            self.pids = self.tree(self.root)
        self.n += 1
        apps = self.query()
        mib = sum(v for p, v in apps.items() if p in self.pids)
        with self.lock:
            self.seen |= any(p in self.pids for p in apps)
            self.win = max(self.win, mib)
        return mib

    def _loop(self):
        while not self.stop_ev.wait(self.poll):
            self.sample()

    def start(self):
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()
        return self

    def stop(self):
        self.stop_ev.set()
        if self.th:
            self.th.join(timeout=60)

    def load(self):
        self.pids = None                                                 # dựng lại cây (tiến trình con mới)
        return round(self.sample() * 2 ** 20 / GIB, 3)

    def begin(self):
        if self.dev is not None:
            import torch
            torch.cuda.reset_peak_memory_stats(self.dev)
        with self.lock:
            self.win = 0.0
        self.sample()

    def end(self):
        self.sample()
        with self.lock:
            out = {"proc_gib": round(self.win * 2 ** 20 / GIB, 3)}
        if self.dev is not None:
            import torch
            out["alloc_gib"] = round(torch.cuda.max_memory_allocated(self.dev) / GIB, 3)
            out["reserved_gib"] = round(torch.cuda.max_memory_reserved(self.dev) / GIB, 3)
        return out


def summarize_mem(rows_by_ds, load_gib, seen=True):
    """{tập: hàng có proc_gib [alloc_gib, reserved_gib]} -> đỉnh chung + từng tập + ảnh gây đỉnh."""
    def peaks(rs):
        out = {}
        for k in ("proc", "alloc", "reserved"):
            v = [r for r in rs if f"{k}_gib" in r]
            if v:
                top = max(v, key=lambda r: r[f"{k}_gib"])
                out[f"peak_{k}_gib"], out[f"peak_{k}_image"] = top[f"{k}_gib"], top["image_id"]
        return out
    allr = [r for rs in rows_by_ds.values() for r in rs]
    return {"definition": MEM_DEFINITION, "load_gib": load_gib, "pid_seen": seen, **peaks(allr),
            "per_dataset": {d: peaks(rs) for d, rs in rows_by_ds.items()}}


def set_dotted(cfg, items):
    """`["vllm.kv_cache_gib=2", "engine.max_num_seqs=1"]` -> ghi vào cfg (giá trị đọc kiểu YAML). -> {khoá: giá trị}."""
    import yaml
    done = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--vllm-set cần dạng khoá.con=giá_trị, nhận '{it}'")
        key, val = it.split("=", 1)
        *path, last = key.split(".")
        d = cfg
        for p in path:
            d[p] = dict(d[p]) if isinstance(d.get(p), dict) else {}     # bản sao: không sửa dict dùng chung của config gốc
            d = d[p]
        d[last] = done[key] = yaml.safe_load(val)
    return done


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

    def setup(self, dataset, samples, geo=None):
        import torch
        import ce_localization.train as train_mod
        from ce_localization.data.cocount import CoCountAddDataset, load_review
        from ce_localization.data.turns import CE130AddDataset, TurnIndex, collate_add, letterbox_size, to_device_add
        from ce_localization.engine.add_eval import predict_batch
        a = self.a
        if dataset == "ce130":
            ds = CE130AddDataset(TurnIndex(a.turn_index), a.samples_root, "test_all", self.S)
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
    """ZETA1–3: đúng `baseline.methods.predict` với danh sách 1 ảnh (HiddenObjects: CLIP text tính sẵn, DataLoader không worker).
    OracleLoc (ZETA7 / 7.1, không weight): CPU, 1 tiến trình, MỘT tiêu chí (`--criterion`, mặc định tiêu chí đầu của config); thời gian
    gồm dựng hình học (đọc chú thích + cỡ ảnh) + dò lưới."""

    def __init__(self, a, log=log):
        from baseline.methods import NO_WEIGHTS, load_model
        from ce_localization.utils.config import load_config
        cfg = load_config(a.config)
        self.oracle = cfg["method"] in NO_WEIGHTS
        if not self.oracle and not a.ckpt:
            raise SystemExit(f"{cfg['method']}: cần --ckpt")
        self.a, self.dev = a, (None if self.oracle else torch_device(a))
        if self.oracle:
            crit = a.criterion or cfg["predict"]["criteria"][0]
            self.cfg = {**cfg, "predict": {**cfg["predict"], "criteria": [crit], "workers": 1}}
            self.model, self.sync = None, None
        else:
            self.cfg = {**cfg, "predict": {**cfg["predict"], "num_workers": 0}}
            self.model = load_model(self.cfg, a.ckpt, self.dev)
            self.sync = torch_sync(self.dev)
        self.info = {"config": a.config, "ckpt": a.ckpt, "baseline": cfg["method"], "K": cfg["predict"]["k"],
                     "device": "cpu (1 tiến trình)" if self.oracle else str(self.dev),
                     **({"criterion": self.cfg["predict"]["criteria"][0], "stride": cfg["predict"]["stride"]} if self.oracle else {})}

    def setup(self, dataset, samples, geo=None):
        from baseline.methods import predict
        if self.oracle:
            crit = self.cfg["predict"]["criteria"][0]
            quiet = lambda *x, **k: None  # noqa: E731
            return (lambda s: {"boxes": predict(self.cfg, None, [s], log=quiet, geo_fn=geo, log_every=0)[crit][0]["boxes"]}, None)
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
        if a.model:
            self.cfg["model"] = a.model                        # thư mục HF đã merge LoRA (bản SFT), như `vlm/run.py --model`
        h0 = prompt_hash(self.cfg)
        overrides = set_dotted(self.cfg, a.vllm_set)
        if prompt_hash(self.cfg) != h0:
            raise SystemExit(f"--vllm-set {a.vllm_set} đổi prompt_hash (đổi câu trả lời model) — chỉ đổi khoá bộ nhớ / số chuỗi")
        load_env()
        self.backend = make_backend(self.cfg, log)
        self.sync = None
        self.info = {"config": a.config, "model": self.cfg["model"], "backend": self.cfg["backend"], "K": self.cfg["k"],
                     "prompt_hash": prompt_hash(self.cfg), "engine": self.cfg.get("engine"), "max_tokens": self.cfg.get("max_tokens"),
                     "vllm": self.cfg.get("vllm"), **({"overrides": overrides} if overrides else {})}

    def setup(self, dataset, samples, geo=None):
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
        cfg = os.path.splitext(os.path.basename(a.config))[0]
        if not a.ckpt:                                                          # OracleLoc
            from ce_localization.utils.config import load_config
            return f"{cfg}_{a.criterion or load_config(a.config)['predict']['criteria'][0]}"
        return f"{cfg}_{os.path.splitext(os.path.basename(a.ckpt))[0]}"
    from ce_localization.utils.config import load_config
    return load_config(a.config)["name"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="method", required=True)
    for m in METHODS:
        p = sub.add_parser(m)
        if m in ("celoc", "zeta"):
            p.add_argument("--ckpt", required=(m == "celoc"), default=None, help="zeta: bỏ trống với OracleLoc (không weight)")
            p.add_argument("--device", default=None)
        if m == "zeta":
            p.add_argument("--criterion", default=None, help="OracleLoc: tiêu chí đo (mặc định tiêu chí đầu của config)")
        if m in ("zeta", "vlm"):
            p.add_argument("--config", required=True)
        if m == "vlm":
            p.add_argument("--model", default=None, help="ghi đè `model` của config (thư mục merge LoRA của finetune.py), như run.py")
            p.add_argument("--vllm-set", nargs="+", default=None, metavar="KHOÁ=GIÁ_TRỊ",
                           help="đổi khoá config ngoài prompt_hash, vd vllm.kv_cache_gib=2 engine.max_num_seqs=1 (đo bộ nhớ batch 1)")
        p.add_argument("--mem", action="store_true", help="đo bộ nhớ GPU lúc suy luận -> <out-root>/mem/<name>.json (docstring)")
        p.add_argument("--mem-poll", type=float, default=0.2, help="chu kỳ lấy mẫu nền nvidia-smi (s)")
        if m == "celoc":
            p.add_argument("--sampler", default="mock", choices=["mock", "ddim"])
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
    out_dir = os.path.join(a.out_root, "mem") if a.mem else a.out_root
    out = os.path.join(out_dir, f"{name}.json")
    t0 = time.time()
    gpus = gpu_info()
    log(f"[latency] {time.strftime('%Y-%m-%d %H:%M:%S')} | {socket.gethostname()} | {a.method} {name} | {a.n} ảnh đo + {a.warmup} "
        f"warm-up / tập {a.datasets} | GPU {gpus or '-'}" + (" | ĐO BỘ NHỚ (--mem)" if a.mem else ""))
    m = METHODS[a.method](a)
    kind = "api" if a.method == "vlm" and m.cfg["backend"] == "openrouter" else "local"
    mem, load_gib = None, None
    if a.mem:
        if kind == "api" or (a.method == "zeta" and m.oracle):
            raise SystemExit("--mem chỉ cho method chạy GPU ở máy này (không API / OracleLoc CPU)")
        mem = GpuMem(dev=getattr(m, "dev", None), poll=a.mem_poll).start()
        load_gib = mem.load()
        log(f"[mem] sau nạp model: {load_gib:.2f} GiB (cây tiến trình {sorted(mem.pids)})")
    log(f"[latency] nạp xong {time.time() - t0:.0f} s | {m.info}")
    rows = {}
    try:
        for d in a.datasets:
            samples, _, geo = load_data(data_ns(a, d))
            measured, warm = pick(samples, a.n, a.warmup)
            run, prefetch = m.setup(d, measured + warm, geo)
            log(f"[latency] {d}: đo {len(measured)} / {len(samples)} ảnh, warm-up {len(warm)}")
            rows[d] = [{"dataset": d, **r} for r in measure(run, measured, warm, prefetch, m.sync, tag=d, mem=mem)]
    finally:
        if mem:
            mem.stop()
    sm = summarize(rows)
    res = {"name": name, "method": a.method, "kind": kind, "definition": DEFINITION, **sm, "gpu": gpus if kind == "local" else None,
           "host": socket.gethostname(), "date": time.strftime("%Y-%m-%d %H:%M"), "n_per_dataset": a.n, "warmup": a.warmup,
           "pick_seed": PICK_SEED, **m.info}
    if mem:
        res["mem"] = {**summarize_mem(rows, load_gib, mem.seen), "poll_s": a.mem_poll}
    res["images"] = [r for d in a.datasets for r in rows[d]]
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False, default=str)
    f4 = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
    for d, v in sm["per_dataset"].items():
        log(f"[latency] {d:8s} {v['n']:3d} ảnh | TB {f4(v['latency_s'])} s / ảnh" + (f" | lỗi {v['n_error']}" if v["n_error"] else ""))
    log(f"[latency] CHUNG TB {f4(sm['latency_s'])} s / ảnh ({kind}) -> {out} | {time.time() - t0:.0f} s")
    if mem:
        r = res["mem"]
        f2 = lambda k: f"{r[k]:.2f} GiB ({r[k.replace('_gib', '_image')]})" if k in r else "-"  # noqa: E731
        log(f"[mem] nạp {load_gib:.2f} GiB | ĐỈNH tiến trình {f2('peak_proc_gib')} | torch allocated {f2('peak_alloc_gib')} | "
            f"reserved {f2('peak_reserved_gib')}" + ("" if mem.seen else " | ⚠️ nvidia-smi KHÔNG thấy PID nào của job (container?)"))


if __name__ == "__main__":
    main()
