# EXPERIMENT A — vòng 2

Thiết kế đầy đủ: [`../../../docs/EXPERIMENT_A_PLAN.md`](../../../docs/EXPERIMENT_A_PLAN.md).
Cơ sở paper: [`../../../docs/LITERATURE_SURVEY.md`](../../../docs/LITERATURE_SURVEY.md).

Code vòng 2 nằm ở **file riêng**, không sửa file vòng 1. `box_transformer.py`,
`detector.py`, `criterion.py`, `train.py`, `eval.py` giữ nguyên để còn đối chiếu.

| file mới | vai trò |
|---|---|
| `models/dit_blocks.py` | `update_box`, `clamp_to_valid`, `build_cross_mask`, `RegionGate`, `DiTBlock` |
| `models/detector_a.py` | `BoxDiT`, `CELocDetectorA`, `build_model_a` |
| `models/criterion_a.py` | `DeepSetCriterion` — CỘNG loss các tầng |
| `train_a.py` / `eval_a.py` | điểm vào train / eval |
| `tools/gate_delta_direction.py` | **CỬA CHẶN — chạy TRƯỚC khi train** |
| `config/round2_experiment_a.yaml` | N=30, SimOTA, roi_k=3 |
| `tests/test_experiment_a.py` | 19 test |

Dùng lại nguyên từ vòng 1: `roi_sampler.py`, `clip_encoder.py`, `data/`, `utils/`,
và các hàm phụ trợ trong `train.py` / `eval.py`.

---

## Kiến trúc, một hình

```
ảnh  -> CLIP ViT-B/16 frozen -+-> patch_raw [B,1024,768] ----> cho RoI
                              +-> Linear(768->256) --+
text -> CLIP text  frozen -------> Linear(768->256) --+--> memory [B,1025,256]

x_T ~ N(0,I) [B,30,4]                               <- LUỒNG CHÍNH
 |
 +- 4 bước DDIM, mỗi bước 6 tầng DiTBlock:
 |    (1) r <- gate(r, roi(patch_raw, x))    lấy ảnh TẠI x hiện tại
 |    (2) seq = [h ; r + mark(x)]            2N token
 |    (3) self-attn có mask, adaLN theo t    r KHÔNG đọc được h
 |    (4) chỉ r cross-attn vào memory        h lấy ảnh hoàn toàn qua r
 |    (5) x <- update_box(x, box_delta(h))   CỘNG DỒN
 |
 +-> 30 box + 30 score
```

`h` = "tôi ở đâu" (hình học), `r` = "ở đó có gì" (ảnh). `h` không đụng `memory` nên
mismatch giữa không gian toạ độ và không gian đặc trưng **biến mất khỏi luồng**.

---

## Chạy trên server

Mọi lệnh chạy từ `object-detection/count_editing/CE-LocModel`.
Log vào `/mnt/disk1/aiotlab/haitn/log/`, checkpoint vào
`/mnt/disk1/aiotlab/haitn/checkpoints/`.

### Bước 0 — test (chạy được ở local, không cần GPU)

```bash
python -m pytest tests/test_experiment_a.py -q
```
Phải thấy **19 passed**. (Đừng chạy `pytest tests/` trần — các file test vừa là script
`main()` vừa có wrapper `test_*`, lệnh trần từng báo "no tests ran" mà vẫn exit 0.)

### Bước 1 — CỬA CHẶN, chạy trước khi train

Trả lời câu hỏi duy nhất chống đỡ cả thiết kế: `Linear(256->4)` có đoán được **hướng
dịch** về GT không? Vài phút, chặn được ~10 giờ A30 nếu trượt.

```bash
python tools/run_on_free_gpu.py -- tools/gate_delta_direction.py \
    --cache /mnt/disk1/aiotlab/haitn/cache \
    --split val \
    --out /mnt/disk1/aiotlab/haitn/log/round2_gate_delta.json
```

**Đọc kết quả:**
- **TIÊU CHÍ: `cosine > 0,5` ở `d = 1` ô.** Dưới ngưỡng ⇒ cộng dồn vô nghĩa, **DỪNG**,
  không train.
- Cột `xáo` và `r=0` phải **thấp hơn hẳn** cột `cosine`. Nếu xấp xỉ nhau thì `Linear`
  chỉ học prior của phân bố delta, **không dùng ảnh** ⇒ kết quả vô giá trị.
- Cột `nhỏ<1ô` dự kiến tệ nhất (18–29 % số box, RoI 3×3 thoái hoá thành 1×1). Nếu **chỉ**
  nhóm này hỏng thì thiết kế vẫn dùng được, chỉ giới hạn ở box lớn.

### Bước 2 — cache patch token (nếu chưa có)

Cache dùng chung với vòng 1, **không cần build lại** nếu đã có. Nếu chưa:

```bash
for S in train val; do
  python tools/run_on_free_gpu.py -- tools/build_cache.py \
      --config config/round2_experiment_a.yaml --split $S \
      --out /mnt/disk1/aiotlab/haitn/cache
done
```
Khoảng 6,0 GB cho train (1.911 ảnh × 2 phiên bản × 1024 token × 768 fp16).

### Bước 3 — chạy thử ngắn trước khi chạy dài

```bash
python tools/run_on_free_gpu.py -- train_a.py \
    --config config/round2_experiment_a.yaml \
    --cache /mnt/disk1/aiotlab/haitn/cache \
    --save-dir /mnt/disk1/aiotlab/haitn/checkpoints/round2_a_smoke \
    --limit 64 --epochs 2 --log-every-n-batch 5
```
Kiểm: loss hữu hạn, `recall theo tầng` in ra đủ 6 số, không cảnh báo lạ.

### Bước 4 — train thật (nền, kèm PID + logfile)

```bash
LOG=/mnt/disk1/aiotlab/haitn/log/round2_a_$(date +%m%d_%H%M).log
nohup python tools/run_on_free_gpu.py -- train_a.py \
    --config config/round2_experiment_a.yaml \
    --cache /mnt/disk1/aiotlab/haitn/cache \
    --save-dir /mnt/disk1/aiotlab/haitn/checkpoints/round2_a \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
```

Theo dõi:
```bash
tail -f $LOG
```

### Bước 5 — eval

```bash
# Ở N=30, cùng thang với lúc train
python tools/run_on_free_gpu.py -- eval_a.py \
    --ckpt /mnt/disk1/aiotlab/haitn/checkpoints/round2_a/best.pt \
    --cache /mnt/disk1/aiotlab/haitn/cache --split val \
    --out /mnt/disk1/aiotlab/haitn/log/round2_a_eval_val_n30.json

# Ở N=300, số để BÁO CÁO (trần recall ~97 % thay vì 82 %)
python tools/run_on_free_gpu.py -- eval_a.py \
    --ckpt /mnt/disk1/aiotlab/haitn/checkpoints/round2_a/best.pt \
    --cache /mnt/disk1/aiotlab/haitn/cache --split val --num-proposals 300 \
    --out /mnt/disk1/aiotlab/haitn/log/round2_a_eval_val_n300.json
```

---

## Đọc số thế nào

| chỉ số | dùng để | cảnh báo |
|---|---|---|
| **`oracle_recall`** | **chọn checkpoint** | matcher KHÔNG nhìn thấy nó ⇒ không bị đánh lừa |
| `recall theo tầng` | **chỉ số CHÍNH của A** | đường **phẳng** ⇒ cộng dồn không mang lại gì |
| `label_stability` | biến nền | vòng 1 đo được **0,018** (>98 % nhãn đổi mỗi epoch) |
| `mean_bestIoU` | chất lượng box | tách khỏi chất lượng score |
| `iou_matched` | ❌ **không dùng chọn checkpoint** | mù với GT mà không box nào chạm tới; đã đánh lừa **hai lần** |

**Trần `oracle_recall` ở N=30**: 83,8 / 82,4 / 76,1 % (train/val/test) — GT bị cắt cụt
trên 34–50 % số ảnh. **Không so với vòng 1 (N=300, trần ~97 %).**

---

## Ba điều đã biết trước, đừng ngạc nhiên

1. **Loss lớn hơn vòng 1 khoảng 6 lần** — vì CỘNG 6 tầng thay vì chia trung bình
   (DiffusionDet/DETR/V-DETR đều cộng). `loss_mean` trong log là con số so được với
   vòng 1. Nếu **phân kỳ** thì hạ `lr` xuống `5e-5`, **đừng** quay lại chia trung bình.

2. **SimOTA phủ ít GT hơn Hungarian khi `n_gt` lớn** — đo trên dữ liệu giả với N=30:
   `n_gt=30` chỉ khớp 16/30 (Hungarian sẽ khớp đủ 30), vì SimOTA dùng nhiều proposal
   cho một GT. Nếu `oracle_recall` kém bất thường thì đây là nghi phạm đầu tiên; đối
   chứng bằng `matcher.method: "hungarian"` trong config.

3. **Box nhỏ hơn 1 ô lưới (18–29 %) gần như không có tín hiệu kích thước** — lưới RoI
   3×3 nằm gọn trong một patch. Không trị được ở độ phân giải 32×32; cần đổi backbone,
   ngoài phạm vi A.

---

## Nếu A không hơn vòng 1

Ba đối chứng, mỗi cái có điều kiện kích hoạt riêng — **không phải lộ trình**:

| đối chứng | khi nào chạy | cách |
|---|---|---|
| **gốc cố định** | recall theo tầng phẳng hoặc giảm | mọi tầng hồi quy từ `x_t` thay vì cộng dồn (V-DETR/D-FINE cố ý chọn cách này) |
| **Hungarian** | `oracle_recall` kém, nghi SimOTA | `matcher.method: "hungarian"` |
| **denoising query** | `label_stability` vẫn ~0,018 | nhãn qua `arange` cố định, không cần matcher |

Nếu A chạy tốt thì **không đụng tới cả ba**, chuyển sang việc mà `CLAUDE.md` ghi là
ĐIỀU KIỆN CẦN: **metric cho nhánh add**.
