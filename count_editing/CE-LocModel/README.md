# EXPERIMENT A — vòng 2

Thiết kế đầy đủ: [`../../../docs/EXPERIMENT_A_PLAN.md`](../../../docs/EXPERIMENT_A_PLAN.md).
Cơ sở paper: [`../../../docs/LITERATURE_SURVEY.md`](../../../docs/LITERATURE_SURVEY.md).

**Code vòng 1 đã xoá.** Không có file `*_a.py` song song — mỗi việc đúng một file.

| file | vai trò |
|---|---|
| `models/dit_blocks.py` | `update_box`, `clamp_to_valid`, `build_cross_mask`, `RegionGate`, `DiTBlock` |
| `models/detector.py` | `BoxDiT`, `CELocDetector`, `build_model` |
| `models/criterion.py` | `SetCriterion` — CỘNG loss các tầng, matcher mỗi tầng |
| `models/roi_sampler.py` | `RoIFeatureSampler` — giữ nguyên từ vòng 1, đã đo kỹ |
| `models/clip_encoder.py` | CLIP frozen — giữ nguyên |
| `train.py` / `eval.py` | điểm vào |
| `tools/gate_delta_direction.py` | **CỬA CHẶN — chạy TRƯỚC khi train** |
| `config/experiment_a.yaml` | N=30, SimOTA, roi_k=3 |
| `tests/test_experiment_a.py` | 19 test |

Đã xoá khỏi repo: `box_transformer.py` (thân vòng 1), 6 file test của A/B/C1/E1/A.2, và
5 cửa chặn vòng 1 đã chạy xong (`check_keypoint_head`, `check_local_softargmax`,
`check_vertex_rpe`, `check_exemplar_signal`, `measure_size_regression`) — kết quả của
chúng đã nằm trong [`ROUND_1_ARCHIVE.md`](../../../docs/old/ROUND_1_ARCHIVE.md).

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

**Quy ước đường dẫn trên server** — giữ tách bạch để tải về local không lẫn:

| loại | nơi để | ví dụ |
|---|---|---|
| **log chạy** (stdout của job) | `/mnt/disk1/aiotlab/haitn/log/` | `round2_a_0923_1400.log` |
| **kết quả** (json, npy, …) | `/mnt/disk1/aiotlab/haitn/output/` | `round2_gate_delta.json` |
| **checkpoint** | thẳng trong repo: `checkpoints/` | `checkpoints/round2_a/best.pt` |

`checkpoints/` và `output/` đều đã nằm trong `.gitignore` nên không lo commit nhầm.
Checkpoint để trong repo để cấu trúc server và local trùng nhau — tải về giữ nguyên
đường dẫn.

**Thời lượng ước tính** — mọi thứ > 5 phút đều đưa dạng `nohup` nền kèm PID + logfile:

| việc | thời lượng | dạng chạy |
|---|---|---|
| test | ~4 phút | trực tiếp |
| cửa chặn (nhanh, 1 cấu hình) | ~9 phút | nền |
| **cửa chặn (đầy đủ, 20 cấu hình)** | **~20 phút** | **nền** |
| build cache (nếu cần) | 5–15 phút | nền |
| **train 300 epoch** | **~10–20 giờ** | **nền** |
| eval | ~5–10 phút | nền |

### Bước 0 — test (chạy được ở local, không cần GPU)

```bash
python -m pytest tests/test_experiment_a.py -q
```
Phải thấy **91 passed** (19 test của A + 72 test hạ tầng). (Đừng chạy `pytest tests/` trần — các file test vừa là script
`main()` vừa có wrapper `test_*`, lệnh trần từng báo "no tests ran" mà vẫn exit 0.)

### Bước 1 — CỬA CHẶN, chạy trước khi train

Trả lời câu hỏi duy nhất chống đỡ cả thiết kế: `Linear(256->4)` có đoán được **hướng
dịch** về GT không? Chặn được ~10–20 giờ A30 nếu trượt.

> ⚠️ **Lần chạy 2026-09-23 (cosine 0,086) KHÔNG dùng để phán quyết** — bản cửa chặn đó có
> hai lỗi đo, cả hai đều kéo điểm xuống. Đã sửa; phải chạy lại bản mới.
>
> 1. **Nhiễu khuếch tán nuốt biến `d`.** `add_diffusion_noise` tự nó dịch tâm box **2,12 ô**
>    ở `t=249` và **6,64 ô** ở `t=999`, trong khi `d` cố ý gây ra chỉ 0,5–4 ô. Ở `t=999`
>    (`alpha_bar = 0`) tương quan còn **−0,009** nên cả 4 mức `d` cho kết quả **trùng khít** —
>    bảng 16 hàng thực chất chỉ có 4 điểm độc lập. Câu hỏi bị đổi thành **định vị tuyệt đối**,
>    đúng câu cửa chặn soft-argmax vòng 1 đã trượt. ⇒ thêm `t = -1` và cột `dịch thật`.
> 2. **`sampler.out` đóng băng ngẫu nhiên** thành nút thắt 2304→256 mà model thật không có
>    (ở đó lớp này **được học**). Trên tín hiệu tuyến tính hoàn hảo, phép chiếu ấy kéo cosine
>    **0,966 → 0,233** — xấp xỉ đúng con số bảng cũ. ⇒ chấm thêm cột `cos2304` **trước** nút
>    thắt và lấy chính cột đó làm tiêu chí.
>
> Đã loại trừ, *không* phải lỗi: 400 bước AdamW đủ hội tụ (0,896 so với trần lstsq 0,900).

**~20 phút** (20 cấu hình: 4 mức `d` × 5 mức `t`) ⇒ chạy nền. Chi phí gần như chỉ nằm ở
lần đọc cache nguội đầu tiên (~8m30s); các hàng sau ~2 giây mỗi hàng.

```bash
LOG=/mnt/disk1/aiotlab/haitn/log/gate_delta_$(date +%m%d_%H%M).log
nohup python tools/run_on_free_gpu.py -- tools/gate_delta_direction.py \
    --split val \
    --out /mnt/disk1/aiotlab/haitn/output/round2_gate_delta.json \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
```

Muốn biết NGAY có đạt ngưỡng không (**~9 phút**, gần như toàn bộ là đọc cache) — đúng
**hàng phán quyết**:

```bash
LOG=/mnt/disk1/aiotlab/haitn/log/gate_quick_$(date +%m%d_%H%M).log
nohup python tools/run_on_free_gpu.py -- tools/gate_delta_direction.py \
    --split val --d-cells 1.0 --timesteps -1 \
    --out /mnt/disk1/aiotlab/haitn/output/round2_gate_quick.json \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
```

**Đọc kết quả:**
- **HÀNG PHÁN QUYẾT là `t = -1`, `d = 1,0`.** Chỉ hàng đó hỏi đúng câu *"box lệch 1 ô, có
  biết lệch hướng nào không?"*. Các hàng `t ≥ 0` bị nhiễu dịch thêm 2–7 ô (xem cột
  `dịch thật`) nên hỏi sang chuyện khác.
- **TIÊU CHÍ: `cos2304 > 0,5`.** Đọc cột **2304** (trước nút thắt), **không** phải cột 256 —
  xem hộp cảnh báo ở trên.
- Cột `lstsq` là **trần tuyến tính chính xác**. `lstsq ≈ cosine` ⇒ AdamW đã hội tụ, loại bỏ
  nghi ngờ *"train chưa đủ"*. `lstsq ≫ cosine` ⇒ tăng `--epochs` rồi chạy lại.
- Cột `xáo` và `r=0` phải **thấp hơn hẳn** cột `cosine`. Nếu xấp xỉ nhau thì `Linear`
  chỉ học prior của phân bố delta, **không dùng ảnh** ⇒ kết quả vô giá trị.
- Cột `nhỏ<1ô` dự kiến tệ nhất (18–29 % số box, RoI 3×3 thoái hoá thành 1×1). Nếu **chỉ**
  nhóm này hỏng thì thiết kế vẫn dùng được, chỉ giới hạn ở box lớn.

### Bước 2 — cache patch token

**Cache đã có sẵn từ vòng 1, KHÔNG cần build lại**: `../../data/cache_clip/`
(tức `object-detection/data/cache_clip/`, nằm trong repo, đã có trong `.gitignore`).

Đã kiểm 2026-09-23 — khớp với config vòng 2:

| | train | val |
|---|---|---|
| shape | `[1911, 2, 1024, 768]` | `[908, 2, 1024, 768]` |
| dung lượng | 6,0 GB | 2,9 GB |
| image_size | 512 | 512 |
| CLIP | `openai/clip-vit-base-patch16` | như trên |
| số lớp | 72 | 28 |

Dùng được vì vòng 2 **không đổi** encoder, độ phân giải hay cách chuẩn hoá — chỉ đổi
phần sau CLIP.

Nếu vì lý do nào đó phải build lại, `train.py` và cửa chặn đều kiểm cache ngay lúc khởi
động và in sẵn lệnh; hoặc chạy tay:

Sinh cache (mỗi split một lệnh, chạy nền):
```bash
LOG=/mnt/disk1/aiotlab/haitn/log/cache_val_$(date +%m%d_%H%M).log
nohup python tools/run_on_free_gpu.py -- tools/build_cache.py \
    --config config/experiment_a.yaml --split val \
    --out ../../data/cache_clip \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
```
Đổi `val` thành `train` cho split kia.

| split | ảnh | dung lượng | thời gian ước tính (A30) |
|---|---|---|---|
| train | 1.911 | ~6,0 GB | ~10–15 phút |
| val | 908 | ~2,9 GB | ~5–8 phút |

(ảnh × 2 phiên bản gốc/lật × 1024 token × 768 chiều × fp16)

Theo dõi: `tail -f $LOG` — in tiến độ kèm **thời gian đã chạy và ETA** mỗi 20 batch.

### Bước 3 — chạy thử ngắn trước khi chạy dài

```bash
python tools/run_on_free_gpu.py -- train.py \
    --config config/experiment_a.yaml \
    --save-dir checkpoints/round2_a_smoke \
    --limit 64 --epochs 2 --log-every-n-batch 5
```
Kiểm: loss hữu hạn, `recall theo tầng` in ra đủ 6 số, không cảnh báo lạ.

### Bước 4 — train thật (nền, kèm PID + logfile)

```bash
LOG=/mnt/disk1/aiotlab/haitn/log/round2_a_$(date +%m%d_%H%M).log
nohup python tools/run_on_free_gpu.py -- train.py \
    --config config/experiment_a.yaml \
    --save-dir checkpoints/round2_a \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
```

Theo dõi:
```bash
tail -f $LOG
```

### Bước 5 — eval

**~5–10 phút mỗi lần** ⇒ chạy nền. Hai lần, ở hai giá trị N:

```bash
LOG=/mnt/disk1/aiotlab/haitn/log/eval_n30_$(date +%m%d_%H%M).log
nohup python tools/run_on_free_gpu.py -- eval.py \
    --ckpt checkpoints/round2_a/best.pt --split val \
    --out /mnt/disk1/aiotlab/haitn/output/round2_a_eval_val_n30.json \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
```

Rồi ở N=300 — số để **BÁO CÁO** (trần recall ~97 % thay vì 82 %):

```bash
LOG=/mnt/disk1/aiotlab/haitn/log/eval_n300_$(date +%m%d_%H%M).log
nohup python tools/run_on_free_gpu.py -- eval.py \
    --ckpt checkpoints/round2_a/best.pt --split val --num-proposals 300 \
    --out /mnt/disk1/aiotlab/haitn/output/round2_a_eval_val_n300.json \
    > $LOG 2>&1 &
echo "PID $! -> $LOG"
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
