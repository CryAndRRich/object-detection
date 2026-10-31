# baseline — EXPERIMENT ZETA: baseline định vị cho bài ADD

Finetune BootPlace / CSANet (bản tác giả bài Add One Take One, `refs/BOOTPLACE`, `refs/CSANet` — chỉ tham khảo, viết lại đúng logic) trên
CE-130 train, chấm CE-130 `test_all` + CE-CoCount `review5` bằng **đúng code chấm của CE-Loc** (`ce_localization/engine/add_eval.py`) ⇒
số so thẳng với DELTA. Quyết định, siêu tham số, lệnh, kết quả: `docs/EXPERIMENT_ZETA.md` (gốc `multi_condition/`). Thư mục riêng của
ZETA (người dùng chốt 2026-10-30), dùng lại (không sửa) dữ liệu / chấm / tiện ích train của `ce_localization`.

| | |
|---|---|
| `finetune.py` | finetune một baseline (1 GPU hoặc `torchrun`): batch toàn cục 64, cộng dồn micro-batch, AMP, `--max-hours` + `--resume` giữa epoch |
| `predict.py` | K box / ảnh (mặc định 30) -> `<out>_raw.jsonl`, rồi chấm như `score.py` |
| `score.py` | chấm jsonl box pixel ảnh gốc -> `<out>.json` + `<out>_boxes.json` (định dạng `eval.py`); `--first 1` = box chính |
| `config/` | `zeta1.yaml` (ZETA1 = BootPlace), `zeta2.yaml` (ZETA2 = CSANet) — một run một config |
| `data.py` | mẫu test (cùng tập / thứ tự với eval CE-Loc) + hình học canvas, mẫu train, chọn ảnh vật mẫu |
| `scoring.py`, `dist.py` | box gốc -> record canvas -> `add_metrics` / prior / dump; tổng + TB grad qua GPU |
| `models/` | `detr`, `bootplace`, `csanet` — tên tham số như bản gốc ⇒ nạp checkpoint công bố |
| `methods/` | mỗi baseline: tiền xử lý, dataset finetune, `Trainer`, suy luận; `__init__` nạp checkpoint |

Test: `python -m pytest tests/baseline -q` từ `object-detection/` (~35 s CPU). Kaggle: `object-detection/notebooks/baseline_kaggle.ipynb`
(gitignore; mỗi Save & Run All một baseline). Eval trên server:

```bash
cd /mnt/disk1/aiotlab/haitn/object-detection/baseline
python ../tools/run_on_free_gpu.py -- predict.py --config config/zeta1.yaml --ckpt ../weights/add/zeta1/best.pth \
    --dataset ce130 --out /mnt/disk1/aiotlab/haitn/output/add/zeta1/ce130_test_all/test.json
```

Đọc số như `ce_localization/README.md` mục "Đọc số", thêm `cnll_median`, `cnll_mean_trim1` (TB bỏ 1 % ảnh tệ nhất), `cnll_gt_median`.
