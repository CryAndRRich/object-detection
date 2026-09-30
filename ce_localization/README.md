# ce_localization — CE-Loc

Chọn box cho count editing trên CE-130: cho ảnh + tên lớp, sinh N box của lớp đó bằng khuếch tán
trên toạ độ box. Mục tiêu cuối là CE-Loc; detect trên CE-130 là phương tiện học cơ chế "điều kiện
hoá box theo ảnh + text" (xem `CLAUDE.md` ở gốc `multi_condition/`).

**Một thí nghiệm = một config** (`config/<nhóm>/<tên>.yaml`), cùng `train.py` / `eval.py`. Idea mới
thì thêm config, và nếu cần thì thêm module vào đúng package (`data/`, `models/`, `engine/`) kèm khoá
config bật nó — **không** tạo thư mục hay script riêng theo tên thí nghiệm.

## Kiến trúc

```
ảnh [B,3|4,T,T] (letterbox góc trên-trái, T = 1024; kênh 4 = density nếu có)
  -> R-50 (mọi conv train, BN đóng băng) + FPN P2..P5
box nhiễu [B,N,4] (N = 200) --6 stage, mỗi stage:
    RoIAlign(P2..P5, box) -> 1 token / box
    tầng decoder Diffusion Policy: self-attn giữa N token -> cross-attn memory [t ; text ; (ảnh)] -> FFN
    head DiffusionDet: score + delta -> box mới (detach trước stage sau)
Loss DiffusionDet (SimOTA, focal + L1 + GIoU) ở cả 6 stage; suy luận DDIM 1 hoặc nhiều bước.
```

## Cấu trúc

| | |
|---|---|
| `train.py` / `eval.py` | điểm vào duy nhất (1 GPU hoặc `torchrun`; `--resume`, `--max-hours`, `--bench`, `--nan-debug`) |
| `config/alpha/`, `config/beta/` | config từng thí nghiệm (bảng dưới) |
| `data/` | `dataset` (quét CE-130, letterbox, đích box / điểm), `density` (giải mã jet, chỉ mục, chọn bản), `points` (đỉnh density, cỡ giả kNN) |
| `models/` | `backbone`, `roi`, `memory`, `head`, `detector`, `text` (CLIP ViT-B/32 frozen) |
| `engine/` | `diffusion`, `criterion` (SimOTA + loss, chế độ box / điểm), `evaluate` (suy luận + chỉ số), `train_utils`, `nan_debug` |
| `utils/` | hình học box (torch / numpy), toán khuếch tán, chấm điểm numpy, checkpoint ghi nguyên tử, grad theo nhóm, log |
| `tools/` | `build_density_index`, `build_density_points`, `visualize_data` (xem đầu vào bằng mắt), `check_data_facts`, `inspect_spatial_softmax`, `inspect_dp_spatial_softmax` |
| `celoc_paper/` | CE-Loc GỐC của bài viết lại đúng công thức (nạp strict `weights/celoc/best_paper.pth`) + encoder Diffusion Policy — để so sánh / soi mô hình gốc |
| `notebooks/` | `train_kaggle.ipynb` (một config trên T4×2), `celoc_paper_kaggle.ipynb` — gitignore, chỉ ở local |
| `checkpoints/` | không vào git |

Test ở `object-detection/tests/ce_localization/` theo module: `test_data`, `test_models`,
`test_engine`, `test_train_eval` (trọn luồng mọi loại config), `test_celoc_paper`, `test_dp_vision`;
dữ liệu giả dùng chung ở `helpers.py`.

## Thí nghiệm (config)

| config | khác `alpha/alpha0.yaml` ở | kế hoạch + kết quả |
|---|---|---|
| `alpha/alpha0.yaml` | — (memory `[t ; text]`) | `docs/EXPERIMENT_ALPHA.md` |
| `alpha/alpha1.yaml` | `model.memory: spatial_softmax` (+ 1 token SpatialSoftmax(P5)) | 〃 |
| `alpha/alpha2.yaml` | `model.memory: grid` (lưới ô P5 16×16 + PE 2D) | 〃 |
| `alpha/alpha3_1.yaml` | `model.in_channels: 4`, `data.density: full` (density kênh 4, cố định bản đầy đủ nhất) | 〃 mục 5 |
| `alpha/alpha3_2.yaml` | như trên, `data.density: mix` (1/3 đầy đủ · 1/3 thiếu vật · 1/3 trống) | 〃 mục 5 |
| `alpha/alpha3_2_36k.yaml` | như `alpha3_2`, train 36k iter (steps 27k / 33k) | 〃 mục 12.5 |
| `beta/beta0.yaml` | `data.targets: point` (đích = box giả từ điểm density, box GT chỉ để chấm) | `docs/EXPERIMENT_BETA.md` |

## Chạy

Mọi lệnh chạy trong `object-detection/ce_localization/` trên server, sau
`export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache TORCH_HOME=/mnt/disk1/aiotlab/haitn/torch_cache`.

| việc | thời lượng (A30) | dạng |
|---|---|---|
| test (`cd .. && python -m pytest tests/ce_localization -q`) | ~15 phút CPU server | **nền** |
| train 12k iter + 6 lần eval val | ~1h15–1h45 | **nền** |
| eval test (1 + 4 bước, chẩn đoán) | 5–10 phút mỗi lượt | **nền** |
| chỉ mục density (một lần) | ~13 phút | **nền** |

**Train** (`--save-dir` bắt buộc; `last.pth` mỗi 1000 iter, ghi nguyên tử):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/<tên>_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- train.py --config config/alpha/alpha0.yaml \
    --save-dir checkpoints/<tên> > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Bị ngắt thì chạy lại **đúng lệnh đó + `--resume`**. Không có `--resume` mà `last.pth` đã có thì
train DỪNG ngay, không ghi đè. Kaggle: `torchrun --standalone --nproc_per_node=2 train.py ...`.

**Eval** (config lấy từ checkpoint; model 4 kênh thêm `--density full|partial|empty`):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/<tên>_eval_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- eval.py --ckpt checkpoints/<tên>/best.pth \
    --split test --num-proposals 200 --top-k 100 --nms --oracle-score --steps 1 4 --attn-diag 20 \
    --out /mnt/disk1/aiotlab/haitn/output/<tên>_test_N200.json > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Có NMS thì JSON báo cả hai thứ tự hậu xử lý: `steps{k}` (top-k trước) và `steps{k}_nmsfirst`.

**Density** (config có `data.density`): dựng chỉ mục một lần
```bash
python tools/build_density_index.py --samples ../data/samples --ce130 ../data/all_phase2_V2 \
    --out ../data/density_index.json --workers 8
```

## CE-Loc gốc (`celoc_paper/`)

**Soi SpatialSoftmax của CE-Loc gốc** (chỉ cần torch/torchvision/matplotlib, CPU được; 100 ảnh
có hình ~4–7 phút mỗi chế độ):
```bash
for m in original inpainted_1 inpainted_2; do
  python tools/inspect_spatial_softmax.py --image $m --n 100 --out ../../output/spatial_softmax/density_paper/$m
  # weight nodensity: thêm --ckpt ../weights/celoc/best_nodensity.pth, --out .../nodensity/$m
done
```

**Soi SpatialSoftmax của Diffusion Policy** (Push-T; checkpoint `../weights/diffusion_policy/epoch=1850-test_mean_score=0.898.ckpt`,
data `../data/pusht/pusht_cchi_v7_replay.zarr`; cần thêm `zarr<3`, `dill`, `omegaconf`; CPU ~35 giây):
```bash
python tools/inspect_dp_spatial_softmax.py --episode 116 --out ../../output/spatial_softmax/diffusion_policy
```

**Train lại CE-Loc gốc, có / không density.** Công thức suy từ checkpoint gốc: AdamW lr 5e-5
wd 0,01, cosine `T_max` 300 theo epoch, batch toàn cục 32, best = loss train nhỏ nhất;
`--stop-epoch 120` dừng sớm mà giữ lịch. Eval báo cả sampler `mock` của bài lẫn `ddpm` đúng, IoU
đúng (`best_iou`) lẫn công thức sai của bài (`best_iou_orig`), mốc `prior`; `--viz N` vẽ box.
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/celoc_nodensity_$(date +%m%d_%H%M).log
nohup python ../tools/run_on_free_gpu.py -- celoc_paper/train.py --save-dir checkpoints/celoc_nodensity \
    --no-density --stop-epoch 120 > $LOG 2>&1 &
echo "PID $! -> $LOG"
# nhiều GPU (Kaggle): cache rồi DDP
python celoc_paper/build_cache.py --data <samples>/train --out <cache> [--no-density]
torchrun --standalone --nproc_per_node=2 celoc_paper/train.py --save-dir <out> --cache-dir <cache> --stop-epoch 120 ...
```
Ngữ cảnh + kết quả: `docs/SPATIAL_SOFTMAX.md`.

## Đọc số

| chỉ số | đo gì |
|---|---|
| `oracle_recall` | GT được ít nhất một box phủ (IoU ≥ 0,5) — chất lượng BOX, không dùng score. Chọn checkpoint bằng nó |
| `score_AUC` | box khớp GT có score cao hơn box còn lại không — chất lượng XẾP HẠNG |
| `--oracle-score` | trần AP khi score = IoU thật, cùng box. Trần ≫ thật: sửa xếp hạng; trần cũng thấp: sửa box |
| `oracle_recall_per_stage` | recall theo stage; phẳng = các stage sau không tinh chỉnh thêm |
| `density_recall`, `size_recall` | recall tách theo số vật / ảnh và theo cỡ vật (quy về canvas 512) |
| `iou_matched` (log train) | chỉ để đọc — **không** chọn checkpoint (mù với GT không box nào chạm tới) |

⚠️ Chỉ số trong log train là **khử nhiễu 1 bước từ GT đã thêm nhiễu**, cao hơn suy luận thật. Số
báo cáo chỉ lấy từ `eval.py` (DDIM từ nhiễu thuần).
