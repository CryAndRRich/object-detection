#!/usr/bin/env python3
"""Eval CE-Loc (DELTA4+) bài ADD: mỗi mẫu sinh `--n-samples` box độc lập từ nhiễu thuần, chấm IoU với lỗ + C-NLL + on_object
(`engine/add_eval.py`), kèm mốc `prior` (lỗ train ngẫu nhiên, không nhìn ảnh). Config lấy từ checkpoint.

  --dataset ce130 (mặc định)  --split test | test_all (6.122 mẫu, như tác giả — DELTA6+) | val ; --image inpainted [original]
  --dataset cocount           CE-CoCount (ảnh gốc, 10 chỗ trống GT, density trống): --cocount-obj-filter 3 (filt3) hoặc
                              --cocount-review F.jsonl --cocount-min-obj 5 (review5)
  --add-samplers mock ddim    vòng mock 100 bước của bài | sampler bản log-scale tác giả (--ddim-steps 100, --eta 1); hai sampler dùng
                              chung một lượt mã hoá ảnh; nhiều sampler -> khoá `<ảnh>_mock` / `<ảnh>_ddim`
  --obj-size mean exemplar    thêm khoá `<khoá>_objsize[_exemplar]`: cùng box, giữ tâm, (w, h) = TB box vật / 3 box mẫu (CoCount)
  --ablation                  thêm `_noobj` (tắt box vật) và `_norefine` (bỏ refiner: refined = x_t) — ngoài phân bố train
Cùng `--seed` + `--batch-size` ⇒ x_T trùng nhau từng mẫu giữa các checkpoint (so cặp).
Ghi `--out` (chỉ số, model + prior) và LUÔN ghi `<out>_boxes.json`: mọi khoá, mỗi mẫu `image_id`, `t`, `wh`, `boxes` [K,4] (xyxy pixel
canvas, 2 số lẻ), `n_obj`, `cnll` [K] (C-NLL từng box), `cnll_gt` (từng lỗ GT) — tính lại biến thể khác không cần chạy lại model.

  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  nohup python ../tools/run_on_free_gpu.py -- eval.py --ckpt ../weights/add/delta6/best.pth --split test_all \\
      --add-samplers mock ddim --out /mnt/disk1/aiotlab/haitn/output/add/delta6/test_all.json > $LOG 2>&1 &
"""

import argparse
import json
import os
import sys
import time

# allocator CUDA theo đoạn mở rộng được (bớt phân mảnh) — phải đặt TRƯỚC khi CUDA khởi tạo
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.data.cocount import CoCountAddDataset, load_review  # noqa: E402
from ce_localization.data.density import DensityIndex  # noqa: E402
from ce_localization.data.turns import IMAGE_KINDS, CE130AddDataset, TurnIndex, collate_add  # noqa: E402
from ce_localization.engine.add_eval import (SIZE_STATS, add_metrics, predict_add, prior_records, prior_unit_boxes,  # noqa: E402
                                             record_cnll, resize_to_objects)
from ce_localization.engine.train_utils import fmt_gpu_peak, gpu_peak  # noqa: E402
from ce_localization.models.box_policy import SAMPLERS, load_checkpoint  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402
import ce_localization.train as train_mod  # noqa: E402

SHOW = ("best_iou@{K}_any", "hit50@{K}_any", "best_iou@{K}_latest", "hit50@{K}_latest", "mean_iou_any", "box_hit50_any",
        "hole_cover", "on_object", "in_image", "degenerate", "cnll", "cnll_gt")


def variants(samplers, ablation=False, ddim_steps=100, eta=1.0):
    """-> list (hậu tố khoá, mô tả, kw của `sample_from_cond`). Một sampler: không hậu tố; nhiều: `_mock` / `_ddim`."""
    kinds = {"mock": ("mock 100 bước (vòng của bài)", {"sampler": "mock"}),
             "ddim": (f"DDIM tác giả {ddim_steps} bước, eta {eta}, kẹp x̂0", {"sampler": "ddim", "ddim_steps": ddim_steps, "eta": eta})}
    base = [("" if len(samplers) == 1 else f"_{s}", *kinds[s]) for s in samplers]
    out = list(base)
    if ablation:
        out += [(s + "_noobj", d + ", TẮT box vật", {**kw, "use_objects": False}) for s, d, kw in base]
        out += [(s + "_norefine", d + ", BỎ refiner", {**kw, "use_refiner": False}) for s, d, kw in base]
    return out


def write_json(path, obj, compact=False, note=""):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f) if compact else json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
    print(f"  -> {note + ' ' if note else ''}{path}", flush=True)


def dump_rows(rec):
    """Box + C-NLL từng box / từng lỗ GT của mỗi mẫu (dump)."""
    rows = []
    for r in rec:
        row = {"image_id": r["image_id"], "t": int(r["t"]), "wh": r["wh"].round(3).tolist(), "boxes": r["boxes"].round(2).tolist(),
               "n_obj": int(len(r["objects"]))}
        rc = record_cnll(r)
        if rc is not None:
            row["cnll"] = np.round(rc[0], 4).tolist()
            if rc[1] is not None:
                row["cnll_gt"] = np.round(rc[1], 4).tolist()
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True, help="file .json chỉ số; box + C-NLL từng box ghi vào <out>_boxes.json")
    ap.add_argument("--dataset", default="ce130", choices=["ce130", "cocount"])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "test_all"])
    ap.add_argument("--image", nargs="+", default=["inpainted"], choices=IMAGE_KINDS)
    ap.add_argument("--add-samplers", nargs="+", default=["mock"], choices=SAMPLERS)
    ap.add_argument("--ddim-steps", type=int, default=100)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--n-samples", type=int, default=None, help="số box / mẫu (mặc định eval.n_samples = 30)")
    ap.add_argument("--batch-size", type=int, default=64, help="số ẢNH / batch (× n_samples hàng khử nhiễu); OOM thì hạ 32")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ablation", action="store_true", help="thêm `_noobj` / `_norefine`")
    ap.add_argument("--obj-size", nargs="+", default=None, choices=SIZE_STATS)
    ap.add_argument("--cocount-root", default="../data/cocount")
    ap.add_argument("--cocount-obj-filter", type=float, default=None, help="filt3: 3")
    ap.add_argument("--cocount-review", default=None, help="review5: ../../output/add/box_review/cocount.jsonl (+ --cocount-min-obj 5)")
    ap.add_argument("--cocount-min-obj", type=int, default=None)
    for k in ("data-root", "samples-root", "turn-index", "density-root", "density-index"):
        ap.add_argument(f"--{k}", default=None, help=f"ghi đè data.{k.replace('-', '_')} (mặc định: config trong checkpoint)")
    a = ap.parse_args()
    if a.cocount_review and a.cocount_obj_filter is not None:
        sys.exit("--cocount-review thay cho --cocount-obj-filter: chọn một")

    t0 = time.time()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, it = load_checkpoint(a.ckpt)
    d = cfg["data"]
    for k in ("root", "samples_root", "turn_index", "density_root", "density_index"):
        v = getattr(a, "data_root" if k == "root" else k)
        if v:
            d[k] = v
    try:
        index = TurnIndex(d["turn_index"])
    except FileNotFoundError as e:
        sys.exit(str(e))
    K = a.n_samples or cfg["eval"]["n_samples"]
    model = model.to(dev).eval()
    amp = bool(cfg["training"].get("amp")) and dev.type == "cuda"
    prior_unit = prior_unit_boxes(index, "train")
    var = variants(a.add_samplers, a.ablation, a.ddim_steps, a.eta)
    cocount = a.dataset == "cocount"
    out = {"ckpt": a.ckpt, "iter": it, "dataset": a.dataset, "split": None if cocount else a.split, "n_samples": K, "seed": a.seed,
           "add_samplers": a.add_samplers, **({"ddim_steps": a.ddim_steps, "eta": a.eta} if "ddim" in a.add_samplers else {}),
           "amp": amp, "results": {}, "prior": {}}
    dump = {"ckpt": a.ckpt, "iter": it, "split": out["split"], "n_samples": K, "seed": a.seed, "results": {}}
    text_table = None
    for image in (["cocount"] if cocount else a.image):
        if cocount:
            review = load_review(a.cocount_review) if a.cocount_review else None
            ds = CoCountAddDataset(a.cocount_root, d["image_size"], max_obj_ratio=a.cocount_obj_filter, review=review,
                                   min_obj=a.cocount_min_obj)
            out.update(cocount_obj_filter=a.cocount_obj_filter, cocount_review=a.cocount_review, cocount_min_obj=a.cocount_min_obj,
                       cocount_files=len(ds), cocount_files_all=ds.n_all)
            print(f"[eval] CE-CoCount: giữ {len(ds)} / {ds.n_all} file", flush=True)
        else:
            dindex = DensityIndex(d["density_index"], d["density_root"]) if image == "original" else None
            ds = CE130AddDataset(index, d["root"], d["samples_root"], a.split, d["image_size"], image=image, density_index=dindex)
        if a.limit:
            ds.keys = ds.keys[: a.limit]
        if text_table is None:
            text_table = train_mod.build_text_table(ds.classes(), cfg, dev)
            model.check_text_fingerprint(text_table.token_sha)      # CLIP text tải lại phải đúng bản lúc train
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers, collate_fn=collate_add,
                            pin_memory=dev.type == "cuda")
        holes = image != "original"
        print(f"[eval] {a.ckpt} ({it}) | ảnh {image} | {len(ds)} mẫu | {K} box/mẫu | {[s for _, s, _ in var]} | "
              f"{fmt_time(time.time() - t0)}", flush=True)
        t = time.time()
        gpu_peak(dev)
        recs = predict_add(model, loader, text_table, K, seed=a.seed, variants=[(s, kw) for s, _, kw in var], amp=amp,
                           log_every=max(len(loader) // 10, 1))
        peak, sec = gpu_peak(dev), time.time() - t
        pri_base = prior_records(next(iter(recs.values())), prior_unit, K, seed=a.seed)
        posts = [("", None)] + [(f"_objsize{'' if st == 'mean' else '_' + st}", st) for st in a.obj_size or []]
        for suffix, st in posts:
            pri = add_metrics(pri_base if st is None else resize_to_objects(pri_base, st), with_holes=holes, latest=not cocount)
            out["prior"][image + suffix] = pri
            for s, desc, _ in var:
                rec = recs[s] if st is None else resize_to_objects(recs[s], st)
                key = image + s + suffix
                res = add_metrics(rec, with_holes=holes, latest=not cocount)
                res["eval_sec"] = sec
                if peak is not None:
                    res["gpu_peak_gb"], res["gpu_reserved_gb"] = peak
                out["results"][key] = res
                dump["results"][key] = dump_rows(rec if holes else [{k: v for k, v in r.items() if k != "holes"} for r in rec])
                print(f"\n  === {key}: {desc}{'' if st is None else f', cỡ = {st}'} ({res['n']} mẫu; C-NLL trên {res['n_cnll']} mẫu "
                      f">= 5 vật) — model | prior ({fmt_time(sec)}{fmt_gpu_peak(peak)})", flush=True)
                for k in SHOW:
                    k = k.format(K=K)
                    if k in res:
                        print(f"  {k:20s} {res[k]:10.4f} {pri[k]:10.4f}")
    write_json(a.out, out)
    write_json(os.path.splitext(a.out)[0] + "_boxes.json", dump, compact=True, note="box + C-NLL từng box")
    print(f"[eval] xong {fmt_time(time.time() - t0)}", flush=True)


if __name__ == "__main__":
    main()
