# object-detection

Code của dự án `multi_condition` (bối cảnh: [`../CLAUDE.md`](../CLAUDE.md)). Branch `delta`: chỉ CE-Loc bài ADD từ DELTA4 trở đi;
mọi thứ cũ hơn (ALPHA / BETA / GAMMA, DELTA1–3.x, baseline detector, diffuse2seg, user study) ở branch `new`.

| | nội dung |
|---|---|
| [`ce_localization/`](ce_localization/README.md) | CE-Loc: chọn box chỗ thêm vật bằng khuếch tán trên toạ độ box |
| `tests/` | toàn bộ test: `python -m pytest tests -q` (từ `object-detection/`, test nhanh); `--runslow` thêm trọn luồng train / eval + DDP |
| `tools/run_on_free_gpu.py` | chạy một script trên GPU có nhiều bộ nhớ trống nhất (thử lại khi OOM) — dùng chung |
| `data/`, `weights/` | dữ liệu + weight, **không vào git** (zip thủ công lên server) |

Package import theo tên đầy đủ (`ce_localization.models...`); script tự thêm `object-detection/` vào `sys.path` nên vẫn chạy
`python train.py` trong thư mục project.

## `weights/` và `data/`

```
weights/add/<run>/          last.pth, best.pth, history.json của từng run (delta4, delta5*, delta6*)
data/all_phase2_V2/         CE-130 (nhánh xoá vật cộng dồn, annotation)
data/samples/               ảnh inpaint + density + annotation của bài (train / test)
data/turn_index.json        chỉ mục (nhánh, lượt) <-> samples (tools/build_turn_index.py)
data/density_index.json     chỉ mục density (tools/build_density_index.py)
data/cocount/               CE-CoCount (Image/ Anno/ Anno_with_exam_bbox/)
```

## Môi trường

Một `requirements.txt`. Server: env `ce-locmodel` (`/mnt/disk1/aiotlab/envs/ce-locmodel/bin/python`), Python 3.10, torch 2.3.0+cu121,
3× A30 24 GB dùng chung; `huggingface_hub` phải `< 1.0`. Local: `ce_localization/.venv-cpu` (torch 2.2 CPU) cho test; `box_review/`
cần thêm `gradio`.

CE-Loc: bài NeurIPS 2026 của chính người dùng; repo gốc + bản cập nhật của tác giả ở `../refs/` (read-only).
