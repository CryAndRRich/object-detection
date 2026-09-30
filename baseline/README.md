# baseline — detector baseline trên CE-130

Thước đo cho ALPHA / BETA: detector chuẩn trên **cùng dữ liệu, cùng bộ chấm, cùng ngân sách 200 box/ảnh**
với bảng `docs/SCORE.md`. Kế hoạch, lý do, kết quả: `docs/BASELINES.md` (gốc `multi_condition/`). Mục tiêu
cuối vẫn là CE-Loc — baseline chỉ là thước.

| run | model | config | script |
|---|---|---|---|
| BASELINE0 | DiffusionDet R50-FPN (ICCV 2023), train lại công thức D.1 | `configs/baseline0_diffusiondet.yaml` | `train_net.py` / `predict.py` |
| BASELINE1 | Sparse R-CNN R50-FPN (CVPR 2021), 300 proposal | `configs/baseline1_sparsercnn.yaml` | 〃 |
| BASELINE2 | Faster R-CNN R50-FPN (NeurIPS 2015) | `configs/baseline2_fasterrcnn.yaml` | 〃 |
| BASELINE3a | Grounding DINO Swin-T (ECCV 2024), zero-shot, prompt = tên lớp | `configs/baseline3a_gdino_zeroshot.yaml` | `gdino/predict.py` |
| BASELINE3b | Grounding DINO Swin-T finetune CE-130 train | `configs/baseline3b_gdino_finetune.yaml` | `gdino/train.py` / `gdino/predict.py` |

Chung cho BASELINE0–2 (`configs/Base-CE130.yaml`): R-50 ImageNet (torchvision), 12.000 iter × batch
toàn cục 2, augmentation như D.1 (cùng `DiffusionDetDatasetMapper`), eval val mỗi 2.000 iter,
`model_best.pth` theo `oracle_recall` val. AMP **tắt**.

## Cấu trúc

| | |
|---|---|
| `train_net.py` | train / eval detectron2 cho mọi `META_ARCHITECTURE` (DiffusionDet / SparseRCNN / GeneralizedRCNN); `--resume`, `--max-hours` |
| `predict.py` | dump dự đoán test / val (box pixel ảnh gốc) rồi chấm; DiffusionDet quét `--num-proposals` / `--steps` |
| `scoring.py`, `tools/score_predictions.py` | **bộ chấm chung**: dump -> bản ghi kiểu ALPHA -> `ce_localization.engine.evaluate.score` (cả top-k-trước lẫn NMS-trước, trần oracle, recall theo độ dày / cỡ, chỉ số điểm) + hàng markdown cho `docs/SCORE.md` |
| `diffusiondet/` | model DiffusionDet vendored (chuyển từ `object-detection/diffusiondet/diffusiondet/`, không sửa) |
| `sparsercnn/` | model Sparse R-CNN vendored từ `PeizeSun/SparseR-CNN@0e5028d` (MIT), chỉ sửa import `util` sang `diffusiondet.util` (trùng hệt) |
| `gdino/` | Grounding DINO qua Open-GroundingDino: `runtime.py` (dựng model, shim op, suy luận), `data.py` (ODVG), `train.py`, `launch.py`, `predict.py` |
| `objdet/` | đăng ký dataset detectron2, `CE130BoxQualityEvaluator` (chọn checkpoint), mMR CrowdHuman |
| `tools/` | `convert_ce130.py` (CE-130 -> COCO json), `visualize_ce130_coco.py`, `score_predictions.py`, `convert_crowdhuman.py`, `summarize.py` |
| `configs/benchmarks/` | 3 benchmark DiffusionDet cũ (COCO-minitrain / VOC / CrowdHuman) — lưu trữ, không chạy lại; số liệu `docs/old/ROUND_1_ARCHIVE.md`; baseline công bố `published_baselines.yaml` |
| `third_party/` | (gitignore) Open-GroundingDino clone ghim commit — `refs/repos/` chỉ để đọc |
| `checkpoints/`, `notebooks/` | (gitignore) OUTPUT_DIR từng baseline; `notebooks/baseline_kaggle.ipynb` + notebook DiffusionDet cũ |

Test: `python -m pytest tests/baseline -q` từ `object-detection/` (phần detectron2 tự skip nếu không có
detectron2; có thì chạy trọn luồng train -> resume -> predict -> chấm của cả 3 kiến trúc trên CE-130 giả,
~3 phút CPU).

## Chuẩn bị (server)

Mọi lệnh ở `/mnt/disk1/aiotlab/haitn/object-detection/baseline`, sau `git pull`. Trước mỗi job:

```bash
cd /mnt/disk1/aiotlab/haitn/object-detection/baseline
df -h /mnt/disk1
export OBJDET_DATA_ROOT=../data HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache TORCH_HOME=/mnt/disk1/aiotlab/haitn/torch_cache
```

1. **detectron2** (BASELINE0–2) — env `ce-locmodel` hiện không import được; dựng lại theo
   `../README.md` mục "Môi trường" (source nhánh `main`, không có `nvcc` thì `detectron2._C` không build,
   vẫn chạy được). Ghi commit để Kaggle cài đúng bản: `git -C /mnt/disk1/aiotlab/haitn/d2src/detectron2 rev-parse HEAD`.
2. **Open-GroundingDino** (BASELINE3) — clone ghim commit + phụ thuộc (bỏ bước build op: server không có
   `nvcc`, `gdino/runtime.py` tự dùng bản PyTorch thuần):
   ```bash
   git clone https://github.com/longzw1997/Open-GroundingDino.git third_party/Open-GroundingDino
   git -C third_party/Open-GroundingDino checkout d248268ac9cab808d4aa2691f4a76972ec5d9ab4
   pip install addict yapf==0.40.1 supervision==0.6.0 jsonlines timm colorlog submitit   # requirements.txt của repo, trừ torch
   mkdir -p ../weights/gdino && wget -O ../weights/gdino/groundingdino_swint_ogc.pth \
       https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
   ```
3. **COCO json CE-130** (BASELINE0–2) — quét annotation + đọc header ảnh 3 split, ước 5–15 phút trên đĩa
   dùng chung ⇒ nền:
   ```bash
   LOG=/mnt/disk1/aiotlab/haitn/log/convert_ce130_$(date +%m%d_%H%M).log
   nohup python tools/convert_ce130.py --ce130-root ../data/all_phase2_V2 --mode class-agnostic > $LOG 2>&1 &
   echo "PID $! -> $LOG"
   ```
   Kiểm log: 1.911 / 908 / 779 ảnh, 71.767 / 38.289 / 37.812 box.
4. **Test** — `cd .. && python -m pytest tests/baseline -q && cd baseline` (có detectron2: ~5 phút CPU ⇒
   chạy nền như mọi lệnh > 5 phút).

## Chạy (server)

Mọi lệnh ước > 5 phút ⇒ `nohup` + PID + log; ước tính phải bench trước khi tin (`SOLVER.MAX_ITER 50`).

**Train BASELINE K ∈ {0, 1, 2}** — ước 1–1,5 giờ A30 (D.1: 1h07m) + 6 lần eval val (~1–2 phút mỗi lần):
```bash
K=0; CFG=configs/baseline0_diffusiondet.yaml     # K=1: baseline1_sparsercnn ; K=2: baseline2_fasterrcnn
LOG=/mnt/disk1/aiotlab/haitn/log/baseline${K}_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- train_net.py --num-gpus 1 --config-file $CFG > $LOG 2>&1 &
echo "PID $! -> $LOG"          # bị ngắt: chạy lại đúng lệnh + --resume
```
Checkpoint ở `checkpoints/baseline${K}/` (`model_best.pth` theo `oracle_recall` val, `model_final.pth`).

**Dump + chấm test** — 5–15 phút mỗi lượt (quét GT 1 split vài phút + suy luận 779 ảnh):
```bash
O=/mnt/disk1/aiotlab/haitn/output/baselines
LOG=/mnt/disk1/aiotlab/haitn/log/baseline${K}_predict_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- predict.py --config-file $CFG \
    --weights checkpoints/baseline${K}/model_best.pth --split test --out-dir $O \
    --where "A30 server" --train-time <thời lượng train> > $LOG 2>&1 &     # BASELINE0: thêm --num-proposals 200 300 --steps 1 4
echo "PID $! -> $LOG"
```
Mỗi lượt ghi `<run>_test[_N{N}_s{bước}].json` (dump) + `..._metrics.json`, in hàng `SCORE.md:` cho từng
ngân sách (`B200` = 200 box điểm cao nhất — hàng chính; `Ball` = mọi box). Override config: `--opts KEY VALUE`
ở CUỐI lệnh. Chấm lại một dump: `tools/score_predictions.py --pred <dump.json>`.

**BASELINE3a** (không train) — Swin-T, op PyTorch thuần trên server: ước 10–30 phút cho 779 ảnh:
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/baseline3a_predict_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- gdino/predict.py --config configs/baseline3a_gdino_zeroshot.yaml \
    --split test --out-dir /mnt/disk1/aiotlab/haitn/output/baselines --where "zero-shot" > $LOG 2>&1 &
echo "PID $! -> $LOG"
```

**BASELINE3b** — chạy Kaggle là chính (build được CUDA op). Server: op PyTorch thuần, ước 4–8 giờ, bench trước:
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/baseline3b_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- gdino/train.py --config configs/baseline3b_gdino_finetune.yaml > $LOG 2>&1 &
echo "PID $! -> $LOG"          # ngắt: chạy lại đúng lệnh -> main.py tự nối tiếp từ checkpoint.pth
# chọn checkpoint trên val (oracle_recall), rồi test checkpoint đó:
python gdino/predict.py --config configs/baseline3b_gdino_finetune.yaml --split val \
    --weights checkpoints/baseline3b/checkpoint0009.pth checkpoints/baseline3b/checkpoint.pth \
    --out-dir /mnt/disk1/aiotlab/haitn/output/baselines --select-out /mnt/disk1/aiotlab/haitn/output/baselines/baseline3b_select.json
```

## Kaggle T4×2

`notebooks/baseline_kaggle.ipynb` (gitignore, chỉ ở local), khuôn `ce_localization/notebooks/train_kaggle.ipynb`:
đổi `RUN` ở ô 1 (`baseline0` / `baseline1` / `baseline2` / `baseline3a` / `baseline3b`) và `BRANCH`. Mỗi lần
Save & Run All **một** baseline, cả 2 GPU (detectron2 `--num-gpus 2`, batch toàn cục vẫn 2; Grounding DINO
`torch.distributed.run`), toàn notebook ≤ 11 giờ (`--max-hours`), output `last.pth` + `best.pth` + `results/`
(dump, metrics, log). Input: dataset `ce130-gt.zip` của ALPHA; Internet On (R-50, weight GD, BERT).

## DiffusionDet: sửa gì so với repo gốc, bẫy đã gặp

9/13 file model y nguyên từng byte (kể cả `detector.py`, `head.py`, `loss.py`). Ba chỗ sửa đều **ngoài**
đường chạy R50: `timm.models.layers` -> `timm.layers`; `indexing="ij"` cho `torch.meshgrid`; bỏ nhánh
torchvision < 0.7 trong `util/misc.py` (so version sai nên luôn true).

- **Số class phải khớp dataset** — `check_num_classes` raise nếu lệch (khoá theo kiến trúc).
- ⛔ **KHÔNG bật AMP.** `scale_clamp ≈ 8,74` cho phép mỗi stage nhân box tới 6250×; qua 6 stage box trung
  gian lên 1e18 — fp16 tràn ở 65504 -> NaN -> `assert x2 >= x1`.
- **Inference batch 1** — `ddim_sample` giả định batch 1 khi box renewal.
- **`SAMPLE_STEP > 1`: bước cuối KHÔNG vào pool ensemble** (`img = x_start; continue` chạy trước
  `ensemble_coord.append`) — 4 bước gộp 3 × N box.
- `USE_NMS True` mặc định (NMS 0,5 trong `detector.py`) — `predict.py` tắt để có N box thô như ALPHA.
- `DiffusionDetDatasetMapper(is_train=False)` **xoá GT** — evaluator đọc GT từ `DatasetCatalog`.
- `hooks.BestCheckpointer` gốc của detectron2 trỏ `last_checkpoint` sang `model_best.pth` -> `--resume`
  nối tiếp sai chỗ; `train_net.py` dùng `BestCheckpointerKeepLast` (có test).
- COCO-minitrain có nhiều bản — chỉ split của `giddyyupp/coco-minitrain` so được với 27,7 AP.
- CrowdHuman dùng `fbox` (baseline Table 7 là fbox).

## D.1 (2026-09-09, cũ) và dữ liệu CE-130

D.1 = DiffusionDet R50 trên CE-130 class-agnostic, 12.000 iter, batch 2, LR 8,84e-6, 1× A30, ~1h07m, chấm
bằng COCOEvaluator / `measure_box_quality_ce130.py` (đã xoá — thay bằng `predict.py` + bộ chấm chung):

| N (test) | oracle_recall | mean_bestIoU | score_AUC | AP | AP50 |
|---|---|---|---|---|---|
| 300 | 0,6734 | 0,5974 | 0,9371 | 34,22 | 58,13 |
| 1000 | 0,7853 | 0,6763 | 0,9483 | 37,43 | 63,65 |
| 3000 | 0,8349 | 0,7062 | 0,9489 | 38,27 | 65,22 |

BASELINE0 train lại đúng công thức này trong pipeline mới; số so với các ALPHA lấy từ bộ chấm chung.

**Chất lượng dữ liệu CE-130** (không sửa, mọi thí nghiệm đọc cùng dữ liệu):
- **Lô annotation HỎNG ở test**: 16 ảnh (15 id dạng `62xx`) có 855 box > 50 % diện tích ảnh = 4,2 % GT
  test; giữ nguyên. Xem: `tools/visualize_ce130_coco.py ... --suspect-only`.
- **Các branch cùng ảnh bất đồng GT** ở `fixed_annotation.json` (86,5 % ảnh val, 79,7 % test): converter
  chọn branch chỉ số nhỏ nhất (khớp `scan_ce130` của `ce_localization`), chênh ~0,03 % số box.
- Mỗi ảnh CE-130 đúng một lớp ⇒ class-agnostic hợp lệ; lớp 3 split rời nhau (72 / 28 / 28, kế thừa FSC-147).

## Ghi công

DiffusionDet: Chen, Sun, Song, Luo — ICCV 2023 (CC-BY-NC 4.0, `../LICENSE`). Sparse R-CNN: Sun et al. —
CVPR 2021 (MIT, `sparsercnn/LICENSE`). Grounding DINO: Liu et al. — ECCV 2024; Open-GroundingDino: Long, Li
(MIT, clone riêng). Faster R-CNN / detectron2: Ren et al. NeurIPS 2015; Wu et al., detectron2.
