#!/usr/bin/env python3
"""BASELINE3.2 — finetune Grounding DINO Swin-T trên CE-130 train (72 lớp) bằng Open-GroundingDino.

1. dựng dữ liệu ODVG + label map + val nội bộ (`gdino/data.py`) vào `<output_dir>/data/`;
2. chạy `main.py` của Open-GroundingDino (qua `gdino/launch.py`) với cfg Swin-T `config/cfg_odvg.py` +
   ghi đè trong `finetune:` của config yaml (ngân sách ≈ 12,6 epoch × batch toàn cục 2 như mọi baseline);
3. main.py tự `--resume` từ `<output_dir>/checkpoint.pth` nếu đã có (lưu mỗi epoch).
Chọn checkpoint sau khi train: `gdino/predict.py --split val --weights <các checkpoint*.pth> --select-out`.

Server (không có nvcc -> op PyTorch thuần, chậm; bench trước), từ object-detection/baseline/:

  export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
  LOG=/mnt/disk1/aiotlab/haitn/log/baseline3_2_$(date +%m%d_%H%M).log
  nohup python ../tools/run_on_free_gpu.py -- gdino/train.py --config configs/baseline3_2_gdino_finetune.yaml \\
      > $LOG 2>&1 &
  echo "PID $! -> $LOG"

Kaggle T4×2: `--nproc 2 --output-dir /kaggle/working/ckpt --max-hours <giờ còn lại>`.
"""

import argparse
import glob
import os
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from baseline.gdino.data import prepare  # noqa: E402
from baseline.gdino.runtime import check_repo, check_transformers, load_config, resolve  # noqa: E402

LAUNCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "launch.py")

# Ghi đè cfg_odvg.py cho val nội bộ MỘT lớp "object" (gdino/data.py). use_coco_eval=True của cfg gốc gắn cứng bảng
# 80 lớp COCO trong PostProcess (id_map 0..79 -> pos_map[k]) -> IndexError lúc dựng model (Kaggle 2026-10-01).
# Không truyền qua --options: DictAction biến 'label_list=object' thành CHUỖI "object" (cần list).
OG_CFG_OVERRIDES = {"use_coco_eval": False, "label_list": ["object"]}


def write_og_config(repo, cfg, data_dir):
    """cfg Swin-T của Open-GroundingDino + OG_CFG_OVERRIDES -> `<data_dir>/cfg_odvg_ce130.py` (repo không bị sửa)."""
    with open(os.path.join(repo, cfg["og"]["config"]), encoding="utf-8") as f:
        text = f.read()
    text += "\n\n# --- ghi đè của baseline/gdino/train.py (OG_CFG_OVERRIDES) ---\n" + "".join(
        f"{k} = {v!r}\n" for k, v in OG_CFG_OVERRIDES.items())
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "cfg_odvg_ce130.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def build_command(cfg, og_cfg_file, out_dir, datasets_json, nproc, num_workers, python=sys.executable):
    """-> (cmd, options) — lệnh chạy main.py qua launch.py; `og_cfg_file` từ `write_og_config`."""
    ft = cfg["finetune"]
    if ft["batch_total"] % nproc:
        raise ValueError(f"batch_total {ft['batch_total']} không chia hết cho {nproc} tiến trình")
    options = {"text_encoder_type": cfg["text_encoder"], "batch_size": ft["batch_total"] // nproc,
               "epochs": ft["epochs"], "lr_drop": ft["lr_drop"], "lr": ft["lr"], "lr_backbone": ft["lr_backbone"],
               "max_labels": ft["max_labels"], "save_checkpoint_interval": ft["save_checkpoint_interval"]}
    main_argv = ["-c", og_cfg_file, "--datasets", datasets_json,
                 "--output_dir", out_dir, "--pretrain_model_path", resolve(cfg["weights"]),
                 "--seed", str(ft["seed"]), "--num_workers", str(num_workers),
                 "--options", *[f"{k}={v}" for k, v in options.items()]]
    launcher = [python] if nproc == 1 else [python, "-m", "torch.distributed.run", "--standalone",
                                            f"--nproc_per_node={nproc}"]
    return launcher + [LAUNCH] + main_argv, options


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="../data/all_phase2_V2", help="thư mục all_phase2_V2/")
    ap.add_argument("--output-dir", default=None, help="mặc định finetune.output_dir của config")
    ap.add_argument("--nproc", type=int, default=1, help="số GPU (Kaggle: 2)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="quá giờ thì dừng tiến trình; checkpoint.pth của epoch cuối vẫn còn -> chạy lại để nối tiếp")
    a = ap.parse_args()

    cfg = load_config(a.config)
    if not cfg.get("finetune"):
        ap.error(f"{a.config} không có mục finetune (BASELINE3.1 là zero-shot: dùng gdino/predict.py)")
    repo = check_repo(cfg)
    print(f"[gdino train] transformers {check_transformers()}", flush=True)     # dừng trước khi dựng dữ liệu / DDP
    weights = resolve(cfg["weights"])
    if not os.path.exists(weights):
        raise FileNotFoundError(f"thiếu weight {weights} — tải: wget -O {weights} {cfg['weights_url']}")
    out_dir = os.path.abspath(a.output_dir or resolve(cfg["finetune"]["output_dir"]))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    datasets_json = prepare(a.data_root, os.path.join(out_dir, "data"), cfg["finetune"]["internal_val_images"],
                            cfg["finetune"]["seed"])
    print(f"[gdino train] chuẩn bị dữ liệu {time.time() - t0:.0f} s", flush=True)
    og_cfg_file = write_og_config(repo, cfg, os.path.join(out_dir, "data"))
    cmd, options = build_command(cfg, og_cfg_file, out_dir, datasets_json, a.nproc, a.num_workers)
    resume = os.path.exists(os.path.join(out_dir, "checkpoint.pth"))
    print(f"[gdino train] {cfg['name']} | {a.nproc} tiến trình | {options} | "
          f"{'NỐI TIẾP từ checkpoint.pth' if resume else 'từ weight ' + weights}\n  {' '.join(cmd)}", flush=True)

    env = dict(os.environ, BASELINE_OG_REPO=repo, PYTHONUNBUFFERED="1")
    p = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        rc = p.wait(timeout=a.max_hours * 3600 if a.max_hours else None)
    except subprocess.TimeoutExpired:
        print(f"[gdino train] DỪNG: hết {a.max_hours:.2f} giờ -> dừng; chạy lại đúng lệnh để nối tiếp "
              f"từ checkpoint.pth (mất phần epoch dở)", flush=True)
        os.killpg(p.pid, signal.SIGTERM)
        try:
            p.wait(120)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
        rc = 0
    cks = sorted(glob.glob(os.path.join(out_dir, "checkpoint*.pth")))
    print(f"[gdino train] returncode {rc} | {(time.time() - t0) / 3600:.2f} giờ | checkpoint: "
          + ", ".join(os.path.basename(c) for c in cks), flush=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
