# ce_localization — CE-Loc (bài ADD, từ DELTA4)

Cho ảnh CE-130 đã xoá vật (inpaint) + tên lớp + box các vật còn lại, sinh box chỗ thêm vật bằng khuếch tán trên toạ độ box. Branch
`delta` chỉ giữ kiến trúc DELTA4 trở đi; code / config / checkpoint các thí nghiệm cũ (ALPHA, BETA, GAMMA, DELTA1–3.x, checkpoint CE-Loc
gốc của bài, baseline, diffuse2seg, user study) ở branch `new`. Kế hoạch + kết quả: `docs/EXPERIMENT_DELTA.md` (gốc `multi_condition/`).

```
ảnh [B,4,512,512] (RGB + density, kiểu bài) -> ResNet18 của bài -> SpatialSoftmax C5 (ô đệm −inf) -> Linear ┐ cond
tên lớp -> CLIP text frozen -> Linear -> Mish                                                          ┘ (FiLM U-Net)
box vật đang có -> token gương qua chính U-Net (t = 0) -> cross-attn ở 6 vị trí U-Net
x_t -> BoxRefiner của tác giả (CLIP ViT frozen + RoIAlign + 4 block RoPE 2D / AdaLN-Zero) -> refined
U-Net 1D nhận [x_t ; refined] -> ε̂ ;  loss = ε-MSE + aux_loss_weight · [MSE(refined, x0) + aux_giou_weight · (1 − GIoU)]
```

| | |
|---|---|
| `train.py` / `eval.py` | điểm vào duy nhất (1 GPU hoặc `torchrun`); docstring đầu file = cách dùng |
| `config/delta/` | `delta4.yaml` đầy đủ; config khác kế thừa qua `base` (`utils/config.py`) — bảng dưới |
| `models/` | `box_policy` (BoxPolicy, sampler mock / ddim, `build_model` / `load_checkpoint`), `unet1d`, `clip_refiner` (port `refs/CE-Loc-update/`), `text` |
| `data/` | `turns` (CE-130 theo (nhánh, lượt), split như CE-Loc gốc, cache uint8), `density` (bản `full` cho ảnh gốc), `cocount` (CE-CoCount, lọc box) |
| `engine/` | `add_eval` (suy luận K mẫu, IoU, on_object, C-NLL), `train_utils` (DDP, lịch lr, checkpoint nguyên tử) |
| `tools/` | `build_turn_index`, `build_density_index`, `build_add_cache` (dữ liệu, một lần); `cnll_outliers`, `plot_cnll_worst` (đọc dump eval) |
| `box_review/` | web Gradio rà box dữ liệu bằng tay (local) — docstring `box_review/app.py` |
| `notebooks/add_kaggle.ipynb` | train một run trên Kaggle T4×2 (gitignore, chỉ ở local; tự clone branch `delta`) |

| config | khác | |
|---|---|---|
| `delta4` | — (box linear, val mock, best theo val `mean_iou_any`) | mục 10 |
| `delta5` / `5_1` / `5_3` / `5_2` / `5_4` / `5_5` | box log (`model.box_coder`), val ddim; `aux_giou_weight` 0 / 0,1 / 0,25 / 0,5 / 0,75 / 1,0 | mục 11 |
| `delta6` / `6_1` / `6_2` | `delta4` / `5_2` / `5_4` + `protocol_celoc.yaml`: không val, best = loss train TB / epoch thấp nhất, 120 epoch, lr 1e-4 cosine về 0 | mục 12 |

## Chạy (server, trong `object-detection/ce_localization/`)

`export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache TORCH_HOME=/mnt/disk1/aiotlab/haitn/torch_cache` trước (CLIP tải lúc dựng model, không
nằm trong checkpoint — vân tay `clip_sha` kiểm khớp). Mọi job > 5 phút: `nohup` + PID + log (khuôn trong `CLAUDE.md`).

```bash
# train (Kaggle: notebook; server 1 GPU): bị ngắt thì chạy lại đúng lệnh + --resume
python ../tools/run_on_free_gpu.py -- train.py --config config/delta/delta6.yaml --save-dir ../weights/add/delta6
# eval CE-130 (test_all = toàn bộ samples/test, 6.122 mẫu) — ghi <out>.json + <out>_boxes.json
python ../tools/run_on_free_gpu.py -- eval.py --ckpt ../weights/add/delta6/best.pth --split test_all --add-samplers mock ddim \
    --out /mnt/disk1/aiotlab/haitn/output/add/ce130_test_all/test_delta6.json
# eval CE-CoCount review5 (box vật rà tay, file >= 5 vật) + đổi cỡ box
python ../tools/run_on_free_gpu.py -- eval.py --ckpt ../weights/add/delta6/best.pth --dataset cocount --add-samplers mock ddim \
    --cocount-review ../../output/add/box_review/cocount.jsonl --cocount-min-obj 5 --obj-size mean exemplar \
    --out /mnt/disk1/aiotlab/haitn/output/add/cocount/review5/test_delta6.json
```

Đo 2026-10-23 (A30, batch 16 ảnh × 30 box, mock): CE-130 test 4.948 mẫu 11 phút (GPU rảnh) – 23 phút (GPU dùng chung), đỉnh 1,2 GB;
tập train 19.998 mẫu 47 phút – 1h41m, đỉnh 5,4 GB (ảnh tới 1.228 vật). Mặc định nay batch 32 (bộ nhớ ~gấp đôi); GPU còn ít chỗ thì
hạ `--batch-size`.

## Đọc số (`engine/add_eval.py`)

| chỉ số | đo gì |
|---|---|
| `cnll` | **C-NLL như bài Add One Take One**: mỗi ảnh lấy box C-NLL thấp nhất trong 30, TB trên ảnh (>= 5 vật cùng lớp). Đặc trưng [w, h, IoU lớn nhất với vật, TB khoảng cách 3 tâm gần nhất], quan hệ tính với vật cả hai lớp (CE-CoCount). `cnll_gt` = cùng cách trên lỗ GT |
| `best_iou@30_*` / `hit50@30_*` | oracle best-of-30 với lỗ bất kỳ (`any`) / lỗ mới nhất (`latest`, chỉ CE-130) |
| `mean_iou_any` / `box_hit50_any` | IoU từng box (không oracle) — chọn best.pth theo val (DELTA4 / 5.x) |
| `hole_cover`, `on_object`, `in_image`, `degenerate` | độ phủ nhiều lỗ; box đè vật (IoA >= 0,5); box trong ảnh; box suy biến |
| `prior` | cùng chỉ số cho 30 lỗ train ngẫu nhiên (không nhìn ảnh) |

TB C-NLL bị vài ảnh vật xếp hàng đều chi phối (Gaussian suy biến): dump `<out>_boxes.json` giữ C-NLL TỪNG box mọi ảnh để tính trung vị /
bỏ 1 % (`tools/cnll_outliers.py`) hay vẽ ca tệ (`tools/plot_cnll_worst.py`) không cần chạy lại model. Log train là khử nhiễu 1 bước từ
GT đã thêm nhiễu — số báo cáo chỉ lấy từ `eval.py`.
