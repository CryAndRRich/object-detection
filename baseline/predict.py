#!/usr/bin/env python3
"""Dump dự đoán của một baseline detectron2 (BASELINE0–2) rồi chấm bằng bộ chấm chung
(`scoring.py`, cùng thước đo với ALPHA / BETA và docs/SCORE.md) — docs/BASELINES.md mục 3.3.

- Mỗi ảnh ghi TOÀN BỘ đầu ra của model (sau hậu xử lý chuẩn của chính model), box pixel ảnh gốc.
  DiffusionDet: `USE_NMS False` (200 / 300 box thô như ALPHA); `--num-proposals` / `--steps` quét được
  không cần train lại. Sparse R-CNN: 300 proposal học được. Faster R-CNN: sau NMS nội bộ 0,5, tối đa
  `TEST.DETECTIONS_PER_IMAGE` (300).
- Chấm với từng ngân sách `--budgets` (mặc định 200 box điểm cao nhất + mọi box), cả hai thứ tự hậu
  xử lý (top-k trước / NMS trước), in hàng markdown cho docs/SCORE.md.

Chạy (từ object-detection/baseline/, `export OBJDET_DATA_ROOT=../data`), ~5–15 phút mỗi lượt ⇒ nền:

  LOG=/mnt/disk1/aiotlab/haitn/log/baselines/baseline0_predict_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- predict.py --config-file configs/baseline0_diffusiondet.yaml \\
      --weights ../weights/detection/baseline0/best.pth --split test --num-proposals 200 300 --steps 1 4 \\
      --out-dir /mnt/disk1/aiotlab/haitn/output/baselines --where "A30 server" --train-time 1h10m > $LOG 2>&1 &
  echo "PID $! -> $LOG"
"""

import argparse
import datetime
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # object-detection/

from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.data import build_detection_test_loader  # noqa: E402
from detectron2.modeling import build_model  # noqa: E402

from baseline.diffusiondet import DiffusionDetDatasetMapper  # noqa: E402
from baseline.objdet import register_all  # noqa: E402
from baseline.objdet.datasets import ce130_image_root  # noqa: E402
from baseline.scoring import iid_of_path, load_gt, write_dump  # noqa: E402
from baseline.tools.score_predictions import parse_budgets, score_dump_file  # noqa: E402
from baseline.train_net import build_cfg, check_num_classes  # noqa: E402


def fmt_s(sec):
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


@torch.no_grad()
def run_model(cfg, weights, dataset, limit=None, seed=0, log=print):
    """-> (pred {iid: {boxes_xyxy, scores}}, iteration của checkpoint)."""
    model = build_model(cfg)
    model.eval()
    extra = DetectionCheckpointer(model).load(weights)
    iteration = extra.get("iteration") if isinstance(extra, dict) else None
    if iteration is not None:                     # checkpoint lưu chỉ số 0-based của iter cuối -> số iter đã train
        iteration += 1
    loader = build_detection_test_loader(cfg, dataset, mapper=DiffusionDetDatasetMapper(cfg, is_train=False))
    n_total = min(len(loader.dataset), limit) if limit else len(loader.dataset)
    torch.manual_seed(seed)                      # DiffusionDet lấy mẫu box nhiễu -> tái lập được
    pred, t0 = {}, time.time()
    for batched in loader:
        for inp, out in zip(batched, model(batched)):
            inst = out["instances"].to("cpu")
            iid = iid_of_path(inp["file_name"])
            if iid in pred:
                raise KeyError(f"ảnh {iid} xuất hiện hai lần trong {dataset}")
            pred[iid] = {"boxes_xyxy": inst.pred_boxes.tensor.numpy(), "scores": inst.scores.numpy()}
        n = len(pred)
        if n % max(n_total // 10, 1) == 0 or n >= n_total:
            el = time.time() - t0
            log(f"    [predict {n:5d}/{n_total}] {el / n * 1000:.0f} ms/ảnh | {fmt_s(el)} | "
                f"còn ~{fmt_s(el / n * (n_total - n))}", flush=True)
        if n >= n_total:
            break
    return pred, iteration


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config-file", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--num-proposals", type=int, nargs="+", default=None,
                    help="DiffusionDet: số box lúc suy luận (mặc định 200 = N của SCORE.md)")
    ap.add_argument("--steps", type=int, nargs="+", default=[1], help="DiffusionDet: số bước DDIM")
    ap.add_argument("--budgets", nargs="+", default=["200", "all"])
    ap.add_argument("--out-dir", required=True, help="thư mục kết quả (/mnt/disk1/aiotlab/haitn/output/baselines)")
    ap.add_argument("--data-root", default=None, help="all_phase2_V2/ để chấm (mặc định theo OBJDET_DATA_ROOT)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ chạy N ảnh đầu (kiểm nhanh)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="mặc định cuda nếu có")
    ap.add_argument("--no-score", action="store_true")
    ap.add_argument("--where", default="—", help="nơi chạy / GPU lúc TRAIN, cho hàng SCORE.md")
    ap.add_argument("--train-time", default="—", help="thời lượng train, cho hàng SCORE.md")
    ap.add_argument("--opts", nargs=argparse.REMAINDER, default=[],
                    help="override config dạng KEY VALUE, ĐẶT CUỐI lệnh (vd --opts INPUT.MIN_SIZE_TEST 800)")
    a = ap.parse_args()

    register_all()
    data_root = a.data_root or ce130_image_root()
    base = build_cfg(a.config_file, a.opts)
    arch = base.MODEL.META_ARCHITECTURE
    is_diff = arch == "DiffusionDet"
    if not is_diff and (a.num_proposals or a.steps != [1]):
        ap.error(f"--num-proposals / --steps chỉ dành cho DiffusionDet (đang là {arch})")
    props = a.num_proposals or ([200] if is_diff else [None])
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run = base.BASELINE.NAME or os.path.splitext(os.path.basename(a.config_file))[0]
    dataset = f"ce130_agnostic_{a.split}"
    gt = None if a.no_score else load_gt(data_root, a.split)      # quét GT một lần cho mọi lượt

    for n_prop in props:
        for steps in (a.steps if is_diff else [None]):
            cfg = base.clone()
            cfg.MODEL.WEIGHTS = a.weights
            cfg.MODEL.DEVICE = device
            cfg.DATASETS.TEST = (dataset,)
            if is_diff:
                cfg.MODEL.DiffusionDet.NUM_PROPOSALS = n_prop
                cfg.MODEL.DiffusionDet.SAMPLE_STEP = steps
                cfg.MODEL.DiffusionDet.USE_NMS = False       # box thô như ALPHA; bộ chấm tự NMS
            cfg.freeze()
            check_num_classes(cfg)
            tag = f"{run}_{a.split}" + (f"_N{n_prop}_s{steps}" if is_diff else "")
            print(f"[predict] {tag} | {arch} | weights {a.weights} | {device}", flush=True)
            t = time.time()
            pred, iteration = run_model(cfg, a.weights, dataset, a.limit, a.seed)
            meta = {"run": run, "config": os.path.relpath(os.path.abspath(a.config_file),
                                                         os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "arch": arch, "weights": a.weights, "iter": iteration, "split": a.split,
                    "num_proposals": n_prop, "steps": steps if is_diff else "—",
                    "batch": cfg.SOLVER.IMS_PER_BATCH, "where": a.where, "train_time": a.train_time,
                    "date": datetime.date.today().isoformat(), "predict_sec": round(time.time() - t, 1),
                    "internal_nms": arch == "GeneralizedRCNN", "seed": a.seed, "limit": a.limit}
            path = os.path.join(a.out_dir, f"{tag}.json")
            write_dump(path, meta, pred)
            print(f"  dump -> {path} ({fmt_s(time.time() - t)})", flush=True)
            if not a.no_score:
                score_dump_file(path, data_root, parse_budgets(a.budgets), gt=gt)


if __name__ == "__main__":
    main()
