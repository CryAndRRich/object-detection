# object-detection

Code của dự án `multi_condition` (bối cảnh: [`../CLAUDE.md`](../CLAUDE.md)). Branch `delta`: CE-Loc bài ADD từ DELTA4 trở đi + baseline
định vị ZETA; mọi thứ cũ hơn (ALPHA / BETA / GAMMA, DELTA1–3.x, baseline detector, diffuse2seg, user study) ở branch `new`.

| | nội dung |
|---|---|
| [`ce_localization/`](ce_localization/README.md) | CE-Loc (DELTA4+): chọn box chỗ thêm vật bằng khuếch tán trên toạ độ box; `box_review/` = web rà box tay |
| [`baseline/`](baseline/README.md) | ZETA1–3, 1.1: finetune + chấm BootPlace / CSANet / HiddenObjects bằng đúng code chấm CE-Loc; ZETA7 / 7.1: OracleLoc (mốc oracle, không weight) |
| [`vlm/`](vlm/README.md) | ZETA4–6: RePlan + VLM thương mại / open-source xin 30 box / ảnh, chấm bằng `baseline/score.py`; ZETA4.1, 8.x: finetune SFT LoRA (`finetune.py`) |
| `tools/run_on_free_gpu.py` | chạy một script trên GPU có nhiều bộ nhớ trống nhất (thử lại khi OOM), `--reserve-gib` giữ trước bộ nhớ (`tools/gpu_hold.py`; `--keeper` giữ chỗ cho cả chuỗi lệnh) — dùng chung |
| `tools/latency.py` | latency 1 ảnh — một định nghĩa cho CE-Loc / baseline (cả OracleLoc) / VLM (`celoc` / `zeta` / `vlm`) |
| `tests/` | toàn bộ test: `python -m pytest tests -q` (từ `object-detection/`, test nhanh); `--runslow` thêm trọn luồng train / eval + DDP |
| `notebooks/` | notebook Kaggle (**gitignore**, chỉ ở local): `add_kaggle.ipynb` (CE-Loc), `baseline_kaggle.ipynb` (ZETA1–3), `vlm_kaggle.ipynb` (ZETA6) |
| `data/`, `weights/` | dữ liệu + weight, **không vào git** (zip thủ công lên server) |

Package import theo tên đầy đủ (`ce_localization.models...`); script tự thêm `object-detection/` vào `sys.path` nên vẫn chạy
`python train.py` trong thư mục project.

## `weights/` và `data/`

```
weights/add/<run>/          last.pth, best.pth, history.json, snapshots/epNNN.pth, pretrained/ của từng run (delta8, delta8_1, zeta1 … zeta3, zeta4_1, zeta8_x)
data/all_phase2_V2/         CE-130 (nhánh xoá vật cộng dồn, annotation)
data/samples/               ảnh inpaint + density + annotation của bài (train / test)
data/turn_index.json        chỉ mục (nhánh, lượt) <-> samples (tools/build_turn_index.py)
data/cocount/               CE-CoCount (Image/ Anno/ Anno_with_exam_bbox/)
data/cocount_review.jsonl   bản rà box vật CE-CoCount (`box_review/app.py cocount`) — đầu vào eval `review5`
data/ce130_train.jsonl      box positive gán tay của `samples/train` — đích train DELTA7+
```

Kết quả eval (ngoài repo, `../output/add/`), nhóm theo package: `celoc/<run>/<tập>/` và `baseline/<run>/<tập>/` (`epNNN.json` = snapshot,
`best.json` = best.pth, kèm `*_boxes.json`; tập = `ce130_test_all` | `cocount_review5` | …), `vlm/<model>/<tập>/`; việc chung: hình ở
`viz/`, latency ở `latency/`, kết quả rà box model ở `review/`. Log `../log/add/{celoc,baseline}/<run>/`, `../log/add/vlm/<model>/`.

## Môi trường

Một `requirements.txt`. Server: env `ce-locmodel` (`/mnt/disk1/aiotlab/envs/ce-locmodel/bin/python`), Python 3.10, torch 2.3.0+cu121,
3× A30 24 GB dùng chung; `huggingface_hub` phải `< 1.0`. Server 2 (RTX PRO 5000 48 GB): `/mnt/disk1/aiotlab/haitn/envs/{ce-locmodel,vllm}`
(vLLM: `vlm/README.md`). Local: `ce_localization/.venv-cpu` (torch 2.2 CPU) cho test; `box_review/`
cần thêm `gradio`.

CE-Loc: bài NeurIPS 2026 của chính người dùng; repo gốc + bản cập nhật của tác giả ở `../refs/` (read-only).
