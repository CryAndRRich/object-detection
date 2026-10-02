#!/usr/bin/env python3
"""Eval CE-Loc detector (mọi config của train.py): DDIM từ nhiễu thuần -> AP + các chỉ số tách box khỏi xếp hạng.

Giao thức (docs/EXPERIMENT_ALPHA.md mục 6.2): test, N=200, top-k 100, NMS 0,5.
  oracle_recall / score_AUC / mean_bestIoU trên TOÀN BỘ box trước NMS ; AP trên top-k sau NMS.
  --oracle-score : trần khi score = IoU thật với GT (cùng box) — chênh do box hay do xếp hạng.
  --steps 1 4    : 1 bước (chính) và 4 bước (renewal + ensemble; chạy batch 1).
  --attn-diag K  : chẩn đoán attention cross-attn trên K batch (mục 6.3).
  --density M    : model 4 kênh (ALPHA3): density đưa vào — full (mặc định) / partial / empty (mục 5).
Config lấy từ checkpoint (an toàn hơn), trừ khi truyền --config.

GAMMA2 (`model.arch: propose_refine`): CE-Loc sinh `--n-samples` box MỘT lần (`--proposer-sampler`), mỗi `--refine-t`
(none / t* / noise) chấm riêng trên cùng box đó (khoá `<ảnh>_ce`, `<ảnh>_t<t*>`, `<ảnh>_noise`).

Checkpoint CE-Loc gốc của bài (`model_state_dict`, vd weights/add/paper/best_model.pth) + `--config config/gamma/gamma0.yaml`
(đường dẫn dữ liệu): eval bài add như GAMMA, kèm `excl_paper_train` (bỏ mẫu samples/train mà bài đã train).

GAMMA (`task: add`, docs/EXPERIMENT_GAMMA.md mục 5): mỗi mẫu (nhánh, lượt) sinh `--n-samples` box độc lập (U-Net 1D — GAMMA0,
GAMMA2 pha 1, checkpoint của bài: `--add-samplers`, mặc định mock 100 bước như bài, `ddpm` = 1000 bước đúng công thức; GAMMA1:
DDIM `--steps` bước, mỗi số bước một lượt), chấm IoU với lỗ + C-NLL + on_object (`engine/add_eval.py`),
kèm mốc `prior` (lỗ train ngẫu nhiên, không nhìn ảnh). GAMMA1 in thêm attention của query lên [t ; text ; vis].
  --image inpainted original : ảnh inpaint (có lỗ) và ảnh gốc không lỗ (phép thử lối tắt; chỉ chỉ số không cần GT)
  --add-density M            : model 4 kênh — sample (density của chính mẫu, mặc định cho ảnh inpaint) / full (bản
                               đầy đủ nhất của ảnh gốc, mặc định cho ảnh gốc) / empty
  LOG=/mnt/disk1/aiotlab/haitn/log/gamma/gamma0_eval_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- eval.py --ckpt ../weights/add/gamma0/best.pth --split test \
      --image inpainted original --out /mnt/disk1/aiotlab/haitn/output/gamma/gamma0_test.json > $LOG 2>&1 &

  cd object-detection/ce_localization
  LOG=/mnt/disk1/aiotlab/haitn/log/alpha/alpha0_eval_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- eval.py --ckpt ../weights/detection/alpha0/best.pth \\
      --split test --num-proposals 200 --top-k 100 --nms --oracle-score --steps 1 4 --attn-diag 20 \\
      --out /mnt/disk1/aiotlab/haitn/output/alpha/alpha0_test_N200.json > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import json
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ce_localization.data.dataset import CE130Dataset, collate  # noqa: E402
from ce_localization.data.density import EVAL_MODES, DensityIndex  # noqa: E402
from ce_localization.data.turns import ADD_DENSITY, IMAGE_KINDS, CE130AddDataset, TurnIndex, collate_add  # noqa: E402
from ce_localization.engine.add_eval import add_metrics, predict_add, prior_records, prior_unit_boxes  # noqa: E402
from ce_localization.engine.evaluate import attention_diagnostics, predict, score  # noqa: E402
from ce_localization.models.backbone import density_ratio_of  # noqa: E402
from ce_localization.models.box_policy import BoxPolicy  # noqa: E402
from ce_localization.models.detector import build_model  # noqa: E402
from ce_localization.utils.log import fmt_time  # noqa: E402
import ce_localization.train as train_mod  # noqa: E402

KEYS = ("AP50", "AP75", "AP_coco", "precision", "recall", "recall10", "recall30",
        "oracle_recall", "score_AUC", "mean_bestIoU", "score_head_cost")


def print_results(tag, res):
    print(f"\n  === {tag} ===")
    for k in KEYS:
        print(f"  {k:16s} {res[k]:.4f}")
    print(f"  {'recall/stage':16s} {' '.join(f'{v:.3f}' for v in res['oracle_recall_per_stage'])}")
    print(f"  {'box giữ/ảnh':16s} {res['kept_per_image']:.1f}  (hậu xử lý {res['postprocess']})")
    for name, v in res["size_recall"].items():
        print(f"  {'recall ' + name:16s} {v['oracle_recall']:.4f}  (n_gt {v['n_gt']})")
    for name, v in res["density_recall"].items():
        print(f"  {'ảnh ' + name:16s} oracle_recall {v['oracle_recall']:.4f} | box giữ lại phủ "
              f"{v['kept_recall']:.4f}  ({v['n_img']} ảnh, n_gt {v['n_gt']})")
    p = res["point"]
    print("  ĐIỂM (tâm box dự đoán nằm trong box GT):")
    for k in ("oracle_recall_pt", "AP_pt", "recall_pt", "precision_pt", "score_AUC_pt",
              "count_MAE@0.3", "count_MAE@0.5"):
        print(f"  {k:16s} {p[k]:.4f}")
    for name, v in p["by_density"].items():
        print(f"  {'ảnh ' + name:16s} oracle_recall_pt {v['oracle_recall_pt']:.4f} | giữ lại trúng "
              f"{v['kept_recall_pt']:.4f}  ({v['n_img']} ảnh, n_gt {v['n_gt']})")
    if "oracle_score" in res:
        o = res["oracle_score"]
        print("  TRẦN khi score = IoU thật (cùng box, cùng top-k/NMS):")
        for k in ("AP50", "AP75", "AP_coco", "recall"):
            print(f"  {k:16s} thật {res[k]:.4f} | trần {o[k]:.4f} ({res[k] / max(o[k], 1e-9) * 100:.0f} % trần)")


ADD_KEYS = ("best_iou@{K}_any", "hit50@{K}_any", "best_iou@{K}_latest", "hit50@{K}_latest", "mean_iou_any",
            "box_hit50_any", "mean_iou_latest", "box_hit50_latest", "hole_cover", "on_object", "in_image",
            "degenerate", "cnll_F1_n1_mean", "cnll_F1_n1_median", "cnll_F1_sel_mean", "cnll_F1_sel_median",
            "cnll_F2_n1_mean", "cnll_F2_n1_median", "cnll_F2_sel_mean", "cnll_F2_sel_median")


def print_add(tag, res, prior, K):
    print(f"\n  === {tag} ({res['n']} mẫu, {K} box/mẫu; C-NLL trên {res['n_cnll']} mẫu có >= 5 vật) ===")
    print(f"  {'':22s} {'model':>9s} {'prior':>9s}")
    for k in ADD_KEYS:
        k = k.format(K=K)
        if k in res:
            print(f"  {k:22s} {res[k]:9.4f} {prior[k]:9.4f}")
    for t, d in res.get("by_turn", {}).items():
        print(f"  lượt {t} ({d['n']:5d} mẫu)       best_iou_any {d['best_any']:.4f} | mean_iou_any {d['mean_any']:.4f} | "
              f"best_iou_latest {d['best_latest']:.4f}")


def paper_train_mask(index, rec):
    """Record nào là mẫu `samples/train` — tập checkpoint CE-Loc gốc của bài ĐÃ TRAIN (gồm 182 / 779 ảnh gốc test CE-130;
    samples/train và samples/test không chung ảnh gốc) -> list bool."""
    return [index.turns[r["image_id"]]["sample"].startswith("train/") for r in rec]


def main_add(a, cfg, ck, dev, t0):
    """GAMMA: eval bài add trên `--split`, mỗi loại ảnh của `--image` một lượt. `ck` là checkpoint của train.py hoặc
    checkpoint CE-Loc gốc của bài (`model_state_dict`; dữ liệu lấy từ `--config`, đầu vào kiểu bài, text bằng CLIP của
    checkpoint). Mọi checkpoint đều báo thêm `excl_paper_train` = chỉ số trên phần KHÔNG thuộc samples/train (so sòng phẳng
    với checkpoint của bài)."""
    d = cfg["data"]
    for k, v in (("samples_root", a.samples_root), ("turn_index", a.turn_index), ("density_index", a.density_index),
                 ("density_root", a.density_root)):
        if v:
            d[k] = v
    try:
        index = TurnIndex(d["turn_index"])
    except FileNotFoundError as e:
        sys.exit(str(e))
    K = a.n_samples or cfg["eval"]["n_samples"]
    paper = "model_state_dict" in ck
    clip = None
    if paper:                                         # CE-Loc gốc của bài: ResNet18, đầu vào kiểu bài, box chia canvas
        model, clip, info = BoxPolicy.load_celoc_paper(ck)
        four = info["in_channels"] == 4
        d["input_style"] = "paper"
        it, dwr = f"epoch {info['epoch']}", None
        print(f"[eval] checkpoint CE-Loc gốc của bài: {info}", flush=True)
    else:
        four = train_mod.image_in_channels(cfg) == 4
        model = build_model(cfg, pretrained_backbone=False)
        model.load_state_dict(ck["model"])
        it, dwr = ck.get("iter"), density_ratio_of(model)
    model = model.to(dev).eval()
    src = d.get("split_source", "ce130")
    prior_unit = prior_unit_boxes(index, "train", src)
    if cfg["model"].get("arch") == "box_refiner" and not paper:
        variants = [(f"_steps{s}", f"DDIM {s} bước", {"steps": s}) for s in a.steps]
    elif len(a.add_samplers) == 1:                    # GAMMA0 / bài: một sampler -> khoá không hậu tố (sampler ghi ở out)
        sm = a.add_samplers[0]
        variants = [("", "mock 100 bước (vòng của bài)" if sm == "mock" else "DDPM 1000 bước", {"sampler": sm})]
    else:
        variants = [(f"_{sm}", sm, {"sampler": sm}) for sm in a.add_samplers]
    refine_vars = []
    if cfg["model"].get("arch") == "propose_refine" and not paper:
        for v in a.refine_t:
            rt = None if v == "none" else ("noise" if v == "noise" else int(v))
            suffix = "_ce" if rt is None else ("_noise" if rt == "noise" else f"_t{rt}")
            desc = ("CE-Loc một mình" if rt is None else f"refine từ nhiễu thuần, {a.refine_steps} bước" if rt == "noise"
                    else f"CE-Loc -> refine t*={rt}, {a.refine_steps} bước")
            refine_vars.append((suffix, desc, {"refine_t": rt, "refine_steps": a.refine_steps}))
    out = {"ckpt": a.ckpt, "iter": it, "split": a.split, "n_samples": K, "density": {}, "results": {},
           "prior": {}, "density_weight_ratio": dwr, "add_samplers": a.add_samplers, "proposer_sampler": a.proposer_sampler}
    text_table = None
    for image in a.image:
        dens = (a.add_density or ("sample" if image == "inpainted" else "full")) if four else None
        dindex = DensityIndex(d["density_index"], d["density_root"]) if dens == "full" else None
        ds = CE130AddDataset(index, d["root"], d["samples_root"], a.split, d["image_size"], density=dens,
                             density_index=dindex, image=image, style=d.get("input_style", "ours"), split_source=src)
        if a.limit:
            ds.keys = ds.keys[: a.limit]
        if text_table is None:
            text_table = train_mod.build_text_table(ds.classes(), cfg, dev, **({"state_dict": clip} if paper else {}))
        loader = DataLoader(ds, batch_size=a.batch_size or cfg["eval"]["batch_size"], shuffle=False,
                            num_workers=a.num_workers, collate_fn=collate_add)
        out["density"][image] = dens
        holes = image == "inpainted"

        def report(key, desc, rec, sec, attn_key=None):
            res = add_metrics(rec, with_holes=holes)
            res["eval_sec"] = sec
            if hasattr(model, "pop_attn"):
                res["attn"] = model.pop_attn() if attn_key is None else model.pop_attn(attn_key)
            pri_rec = prior_records(rec, prior_unit, K, seed=a.seed)
            pri = add_metrics(pri_rec, with_holes=holes)
            seen = paper_train_mask(index, rec)
            if any(seen) and not all(seen):
                keep = [i for i, m in enumerate(seen) if not m]
                res["excl_paper_train"] = add_metrics([rec[i] for i in keep], with_holes=holes)
                pri["excl_paper_train"] = add_metrics([pri_rec[i] for i in keep], with_holes=holes)
            res["n_paper_train"] = int(sum(seen))
            print_add(f"ảnh {image}, {desc} ({fmt_time(sec)})", res, pri, K)
            if "excl_paper_train" in res:
                ex = res["excl_paper_train"]
                print(f"  -- bỏ {res['n_paper_train']} mẫu thuộc samples/train (bài đã train), còn {ex['n']}: " + " | ".join(
                    f"{k.format(K=K)} {ex[k.format(K=K)]:.4f} (prior {pri['excl_paper_train'][k.format(K=K)]:.4f})"
                    for k in ("best_iou@{K}_latest", "hit50@{K}_latest", "mean_iou_latest", "mean_iou_any", "on_object")
                    if k.format(K=K) in ex))
            if res.get("attn"):
                print("  attention query -> [t ; text ; vis] theo tầng: " + " | ".join(
                    "/".join(f"{x[k]:.2f}" for k in ("t", "text", "vis")) for x in res["attn"]))
            out["results"][key], out["prior"][key] = res, pri

        if refine_vars:                               # GAMMA2: CE-Loc chạy MỘT lần / batch, mọi biến thể refine dùng lại
            print(f"[eval] {a.ckpt} ({it}) | ảnh {image} | density {dens} | split {a.split} ({len(ds)} mẫu) | {K} mẫu/ảnh"
                  f" | CE-Loc {a.proposer_sampler} -> {[d_ for _, d_, _ in refine_vars]} | {fmt_time(time.time() - t0)}",
                  flush=True)
            t = time.time()
            recs = predict_add(model, loader, text_table, K, seed=a.seed, log_every=max(len(loader) // 10, 1),
                               sample_kw={"proposer_sampler": a.proposer_sampler}, variants=[v for _, _, v in refine_vars])
            for vi, ((suffix, desc, _), rec) in enumerate(zip(refine_vars, recs)):
                report(image + suffix, desc, rec, time.time() - t, attn_key=vi)
            continue
        for suffix, desc, kw in variants:             # GAMMA1: mỗi số bước DDIM; GAMMA0 / bài: DDPM (+ mock nếu xin)
            print(f"[eval] {a.ckpt} ({it}) | ảnh {image} | density {dens} | split {a.split} ({len(ds)} mẫu) | {K} mẫu/ảnh"
                  f" | {desc} | {fmt_time(time.time() - t0)}", flush=True)
            t = time.time()
            rec = predict_add(model, loader, text_table, K, seed=a.seed, log_every=max(len(loader) // 10, 1), **kw)
            report(image + suffix, desc, rec, time.time() - t)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=float)
        print(f"  -> {a.out}")
    print(f"[eval] xong {fmt_time(time.time() - t0)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None, help="mặc định dùng config LƯU TRONG checkpoint")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data-root", default=None, help="ghi đè data.root (vd. Kaggle)")
    ap.add_argument("--density", default=None, choices=EVAL_MODES,
                    help="chỉ model 4 kênh (ALPHA3); mặc định full")
    ap.add_argument("--density-root", default=None, help="ghi đè data.density_root")
    ap.add_argument("--density-index", default=None, help="ghi đè data.density_index")
    ap.add_argument("--num-proposals", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--nms", action="store_true")
    ap.add_argument("--nms-thr", type=float, default=0.5)
    ap.add_argument("--oracle-score", action="store_true")
    ap.add_argument("--steps", type=int, nargs="+", default=[1], help="số bước DDIM (detect; add: chỉ GAMMA1 box_refiner)")
    ap.add_argument("--no-renewal", action="store_true", help="tắt box renewal khi nhiều bước")
    ap.add_argument("--attn-diag", type=int, default=0, help="số batch cho chẩn đoán attention; 0 = tắt")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="detect: chỉ cho 1 bước (nhiều bước luôn batch 1), mặc định 2 (@1024 batch 8 OOM trên GPU "
                         "dùng chung); add: mặc định eval.batch_size của config")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="file .json trong /mnt/disk1/aiotlab/haitn/output/<nhóm>/")
    ap.add_argument("--image", nargs="+", default=list(IMAGE_KINDS), choices=IMAGE_KINDS, help="GAMMA: loại ảnh vào")
    ap.add_argument("--add-density", default=None, choices=ADD_DENSITY, help="GAMMA, model 4 kênh: density đưa vào")
    ap.add_argument("--n-samples", type=int, default=None, help="GAMMA: số box / mẫu (mặc định eval.n_samples)")
    ap.add_argument("--refine-t", nargs="+", default=["none", "0", "100", "200", "400", "700", "noise"],
                    help="GAMMA2: biến thể refine — none (CE-Loc một mình) | t* (cộng nhiễu box CE-Loc tới t* rồi DDIM) | noise")
    ap.add_argument("--refine-steps", type=int, default=1, help="GAMMA2: số bước DDIM của refine")
    ap.add_argument("--proposer-sampler", default="mock", choices=["ddpm", "mock"],
                    help="GAMMA2: sampler của CE-Loc — mặc định mock 100 bước như bài")
    ap.add_argument("--add-samplers", nargs="+", default=["mock"], choices=["ddpm", "mock"],
                    help="GAMMA0 / GAMMA2 pha 1 / checkpoint của bài (U-Net 1D): mock (mặc định — vòng 100 bước của bài) và / hoặc "
                         "ddpm (1000 bước đúng công thức)")
    ap.add_argument("--samples-root", default=None, help="GAMMA: ghi đè data.samples_root")
    ap.add_argument("--turn-index", default=None, help="GAMMA: ghi đè data.turn_index")
    a = ap.parse_args()

    t0 = time.time()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in ck:                     # checkpoint CE-Loc gốc của bài: dữ liệu / bộ đo từ --config
        if not a.config:
            sys.exit("checkpoint CE-Loc gốc của bài: cần --config (vd config/gamma/gamma0.yaml) để lấy đường dẫn dữ liệu")
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
        if a.data_root:
            cfg["data"]["root"] = a.data_root
        return main_add(a, cfg, ck, dev, t0)
    if a.config:
        with open(a.config) as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = ck["config"]
    if a.data_root:
        cfg["data"]["root"] = a.data_root
    if cfg.get("task", "detect") == "add":
        return main_add(a, cfg, ck, dev, t0)
    a.batch_size = a.batch_size or 2
    if a.density_root:
        cfg["data"]["density_root"] = a.density_root
    if a.density_index:
        cfg["data"]["density_index"] = a.density_index
    dindex = train_mod.density_setup(cfg)
    if dindex is None and a.density:
        sys.exit(f"--density {a.density} nhưng checkpoint là model 3 kênh (không density)")
    density = (a.density or "full") if dindex is not None else None
    n_prop = a.num_proposals or cfg["diffusion"]["num_proposals"]
    top_k = a.top_k or cfg["eval"]["top_k"]
    nms_thr = a.nms_thr if a.nms else None

    ds = CE130Dataset(cfg["data"]["root"], a.split, cfg["data"]["image_size"], density=density,
                    density_index=dindex)
    if a.limit:
        ds.items = ds.items[: a.limit]
    text_table = train_mod.build_text_table(ds.classes(), cfg, dev)
    model = build_model(cfg, pretrained_backbone=False).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    kinds = {}
    if density is not None:                          # loại density thực tế (partial -> full khi ảnh chỉ có 1 bản)
        for it in ds.items:
            k = dindex.pick(it["image_id"], density)[1]
            kinds[k] = kinds.get(k, 0) + 1
    dw = density_ratio_of(model)
    print(f"[eval] {a.ckpt} (iter {ck.get('iter')}) | memory={cfg['model']['memory']} | "
          f"density={density} {kinds or ''}"
          + ("" if dw is None else f" (‖W density‖/‖W RGB‖ conv1 {dw:.4f})") + f" | split={a.split} "
          f"({len(ds)} ảnh) | N={n_prop} | top_k={top_k} | nms={nms_thr} | steps={a.steps} | "
          f"khởi động {fmt_time(time.time() - t0)}", flush=True)

    out = {"ckpt": a.ckpt, "iter": ck.get("iter"), "split": a.split, "n_proposals": n_prop,
           "top_k": top_k, "nms_thr": nms_thr, "memory": cfg["model"]["memory"],
           "density": density, "density_kinds": kinds, "density_weight_ratio": dw, "results": {}}
    for steps in a.steps:
        bs = a.batch_size if steps == 1 else 1
        loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=a.num_workers,
                            collate_fn=collate)
        t = time.time()
        rec, stage = predict(model, loader, text_table, n_prop, steps=steps, top_k=top_k,
                             nms_thr=nms_thr, renewal=not a.no_renewal, seed=a.seed,
                             log_every=max(len(loader) // 10, 1))
        el = time.time() - t
        # có NMS thì báo CẢ HAI thứ tự hậu xử lý (người dùng chốt 2026-09-29): khoá cũ `steps{k}` =
        # top-k trước (so được với bảng cũ), `steps{k}_nmsfirst` = NMS trước như DiffusionDet
        orders = ["topk_first", "nms_first"] if nms_thr is not None else ["topk_first"]
        for order in orders:
            res = score(rec, stage, top_k, nms_thr, oracle=a.oracle_score, order=order)
            res["eval_sec"] = el
            key = f"steps{steps}" + ("_nmsfirst" if order == "nms_first" else "")
            print_results(f"{steps} bước, {order} ({fmt_time(el)})", res)
            out["results"][key] = res

    if a.attn_diag:
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False, num_workers=a.num_workers,
                            collate_fn=collate)
        t = time.time()
        diag = attention_diagnostics(model, loader, text_table, n_prop, max_batches=a.attn_diag,
                                     seed=a.seed)
        print(f"\n  === chẩn đoán attention ({fmt_time(time.time() - t)}) — khối lượng TB theo stage ===")
        for tt, d in diag.items():
            for k, v in d.items():
                print(f"  t={tt:4d} {k:16s} {' '.join('—' if x is None else f'{x:.3f}' for x in v)}")
        out["attention"] = diag

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(out, f, indent=2, ensure_ascii=False, default=float)
        print(f"  -> {a.out}")
    print(f"[eval] xong {fmt_time(time.time() - t0)}", flush=True)


if __name__ == "__main__":
    main()
