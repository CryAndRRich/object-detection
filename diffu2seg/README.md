# diffu2seg — Diffuse2Seg training-free

Port phần **training-free** của **Diffuse2Seg** ([arXiv 2609.06491](https://arxiv.org/abs/2609.06491),
6 Sep 2026 — CARIAD/VW + TU Berlin).

> **Mục đích: KHẢO SÁT CƠ CHẾ, chưa gắn vào pipeline CE-Loc.**
> Đây là công cụ để xem mask/box thực tế ra sao, **không** phải ứng viên cho
> toán tử `T` của [../../docs/01-bai-toan.md](../../docs/01-bai-toan.md).

**Không có phần train Mask2Former** (bước 3 của paper) — bỏ hoàn toàn.

## Hai đường chạy, hai mục đích khác nhau

| | dữ liệu | cấu hình | metric | tool |
|---|---|---|---|---|
| **A. Reproduce** | **PACO-LVIS val** | **của paper**: canvas 1120, `grid_r` 140, stride 6 | **AR₁₀₀₀** | `tools/run_paco.py` |
| A'. phụ | COCO val2017 | như trên | oracle_recall | `tools/run_coco.py` |
| **B. CE-130** | CE-130 | **đã chỉnh**: canvas 512, `grid_r` 64, stride 3 | oracle_recall | `tools/run_stage1.py` |

Đường A không chỉnh gì cho hợp dữ liệu — chỉnh rồi thì kết quả nói về cách ta chỉnh,
không nói về phương pháp. Đường B chỉnh, và mỗi chỗ chỉnh đều kèm số đo (mục
"Khác paper ở đâu").

⚠️ **Các đường KHÔNG so với nhau.** Khác dữ liệu, khác cấu hình, khác cả metric.

### Vì sao là PACO, và vì sao chỉ có PACO

Bảng training-free của paper (Table 1 + 2) dùng 5 bộ. Đã tra từng bộ:

| bộ | AR₁₀₀₀ của paper | lấy được không |
|---|---|---|
| **PACO** | **13,6** | ✅ ann 128 MB + 2410 ảnh COCO train2017 (~0,5 GB) |
| SA-1B | 20,7 | ❌ chỉ tải theo shard nguyên 10 GB; tập val 1000 ảnh của họ không công bố |
| ADE20K | 22,5 | ❌ bản có instance cần tài khoản MIT CSAIL được duyệt. Bản tải tự do `ADEChallengeData2016` là **SEMANTIC** (pixel → 1 trong 150 class, không có instance) nên không tính được AR instance |
| EntitySeg | 23,8 | ❌ bản low-res đã 11,4 GB, chia 3 shard không tách val |
| UVO | 29,9 | ❌ video Kinetics-400, cần xin quyền + tự trích frame |

### ⚠️ Đọc kết quả PACO cho đúng

1. **PACO là part segmentation**: **65,6 %** mục tiêu là bộ phận (`chair:apron`),
   34,4 % là vật nguyên. Một cái ghế đồng thời là 1 mask OBJECT và ~8 mask PART —
   đúng thứ 6 mức granularity sinh ra để bắt. `run_paper.py` in riêng `AR_OBJECT`
   và `AR_PART`, cộng bảng số cụm/số mask theo từng height, để thấy các mức có
   làm gì không.
2. **Trần độ phân giải 90,7 %** ở `grid_r=140` (cạnh ngắn trung vị 4,23 ô, p10 1,09).
   Mask hoàn hảo cũng không vượt được.
3. **AR₁₀₀₀ ≠ `oracle_recall`.** AR trung bình recall trên 10 ngưỡng IoU 0,5→0,95,
   ghép **một-một**, tính trên **mask**. `oracle_recall` là số hạng đầu (t=0,5), cho
   phép một box phủ nhiều GT, tính trên **box** — **luôn cao hơn**. Chỉ AR₁₀₀₀ so được
   với 13,6.

### Dữ liệu PACO

```
data/paco/
├── paco_lvis_v1_val.json    25 MB   2410 ảnh, 31919 annotation
└── images/                 367 MB   2410 ảnh COCO train2017 (KHÔNG phải val2017)
```

Đã kiểm toàn bộ: 31917 mask giải mã sạch, 0 mask rỗng, 0 GT ngoài vùng ảnh.

⚠️ **PACO dùng hai encoding, tách đúng theo loại**: OBJECT là **polygon**, PART là
**RLE nén** — và cả hai đều mang `iscrowd=0` nên phép thử iscrowd quen thuộc không
phân biệt được. Vì thế `pycocotools` là dependency (chỉ loader này cần; import trễ
nên CE-130 không đụng tới).

---

## Cơ chế trong 30 giây

```
ảnh 512×512
  │  SD2 VAE encode → MỘT bước denoise (latent KHÔNG nhiễu hoá)
  │  hook mọi self-attention, trộn 2 layer decoder cao nhất
  ▼
A  (4096 × 4096) — ĐỒ THỊ trên patch token, mỗi hàng tổng = 1
  │  A[i,j] = "token i chú ý tới token j bao nhiêu"
  │  ⚠️ KHÔNG phải feature map: nó nói QUAN HỆ giữa các điểm,
  │     không nói thuộc tính của từng điểm
  ▼
441 hạt one-hot trên lưới đều (cách 3 ô), bỏ hạt rơi vùng pad
  │  p-Laplacian: mực loang trên đồ thị, TỰ TẮT ở chỗ dốc (biên vật)
  ▼
f  (441 × 4096) soft object map
  │  ngưỡng → connected component chứa hạt → box outer-edge → dedup
  ▼
boxes (M, 4) cxcywh ∈ [0,1]
```

**Không có tham số học được nào.** Mask là **nghiệm của một bài toán tối ưu trên
đồ thị**, giải bằng vòng lặp số học. SD2 đóng băng, chỉ để cung cấp đồ thị.

### Vì sao `p < 2` là toàn bộ đóng góp của paper

Số hạng trơn là `(1/p)·Σᵢ gᵢᵖ` với `gᵢ` là độ dốc tại token `i`:

| | hành vi |
|---|---|
| `p = 2` | phạt **bình phương** độ dốc → tối ưu là **san đều** → biên bị bào mòn. `γ` sụp thành `2A`, cả thuật toán còn **một dòng** khuếch tán tuyến tính |
| `p < 2` | phạt **dưới bình phương** → tối ưu **chấp nhận vài chỗ dốc đứng** để phẳng lì chỗ khác → biên sống sót |

Cùng nguyên lý `L1` vs `L2` trong hồi quy. **Có đo được trên CE-130 hay không là
câu hỏi của cửa chặn 1** — cạnh ngắn vật trung vị chỉ 4,65 ô lưới.

---

## Tham số: `config/paper.py` vs `config/stage1.py`

**`config/paper.py` không lệch paper một giá trị nào.** Mỗi dòng trong file ghi kèm
mục của paper nó đến từ đâu:

| | giá trị | nguồn |
|---|---|---|
| canvas / `grid_r` | 1120 / **140** | §4.1 "resize the input images to 1120×1120" |
| timestep | 150 | §4.1 |
| layer + trọng số | 2 layer cuối decoder, **w₁=0,85 / w₂=0,15** | §3.3 + §4.1 |
| `τ_att` | 0,55 | §4.1 |
| prompt stride | 6 | §4.1 |
| `p` / `λ` / `τ_prop` | 1,6 / 1e−5 / 1e−4 | §4.1 |
| **L mức** | **6, log-spaced [0,186 ; 2,99]** | §A.1 |
| **`τ_IoU`** (NMS) | **0,9** | §A.1 |
| **`A_min`** | **100 px** | §A.1 |
| **`N_max`** | **1000** | §3.4 |

⚠️ **`w₁=0,85 / w₂=0,15` là trọng số LAYER**, không phải trộn timestep — chỗ này
từng bị chép nhầm thành 0,5/0,5 của M2N2.

**`config/stage1.py` là đường CE-130**, cố ý lệch, mỗi chỗ lệch kèm số đo:

| tham số | paper | CE-130 | lý do (số đo) |
|---|---|---|---|
| `grid_r` | 140 | **64** | A: 1,54 GB → **0,07 GB** (22× nhẹ). `8×64 = 512` = canvas chuẩn dự án |
| `prompt_stride` | 6 ô | **3 ô** | đo 200 ảnh val: s=3 phủ **96,6 %** box; s=6 bước qua vật nhỏ (box nhỏ nhất trung vị **2,24 ô**) |
| readout | Algorithm 2 | **1 ngưỡng + CC** | GĐ1 để cửa chặn 1 đo RIÊNG cơ chế lan truyền |
| lọc vùng pad | không có | **có** | ảnh CE-130 cao 384 cố định, rộng tới 1918 → pad ~**29 %** canvas. Paper dùng ảnh vuông |
| `g_eps` | **không có** | `1e-8` | `g=0` ở vùng phẳng, `p−2 = −0,4` → `0^(−0,4) = inf` → NaN |
| `mask_rel_floor` | **không có** | `0,05` | xem "Bốn thứ paper không viết" dưới |

### ⚠️ GĐ1 và GĐ2 là hai READOUT KHÁC NHAU, không phải "có/không có cluster"

| | GĐ1 (`stage1.py`) | GĐ2 = Algorithm 2 (`paper.py`) |
|---|---|---|
| ngưỡng từng map | **có** (quantile) | **KHÔNG BAO GIỜ** |
| gộp | không | trung bình trong cụm KL |
| thứ tự | threshold → CC | **upsample → argmax qua cụm** → CC |
| kết quả | các blob độc lập | **PHÂN HOẠCH** ảnh |
| khử trùng | dedup IoU 0,70 | **NMS diện tích giảm dần, 0,9, cap 1000** |
| mask ở đâu | lưới latent (140) | **độ phân giải ảnh gốc** |

Đọc GĐ1 như "GĐ2 bỏ phần cluster" là sai — hai readout khác nhau về bản chất, và
chỉ GĐ2 mới so được với số của paper.

---

## ⚠️ Bốn thứ paper KHÔNG viết, thiếu là hỏng

**1. Khai triển matmul.** Tính `g` theo nghĩa đen dựng tensor `(K,N,N)` = **29 GB**
ở `K=441, N=4096`. A30 có 24 GB → không bao giờ chạy. Phải khai triển:
```
Σⱼ Aᵢⱼ(fⱼ−fᵢ)² = (A f²)ᵢ − 2fᵢ(A f)ᵢ + fᵢ²(A·1)ᵢ
```
**2. Tử/mẫu số cũng phải khai triển** — chỗ thứ hai, dễ sót hơn vì nhìn như một
phép trung bình có trọng số thường. `γᵢⱼ = Aᵢⱼ(gᵖᵢ+gᵖⱼ)` nên
`Σⱼ γᵢⱼfⱼ = gᵖᵢ(A f)ᵢ + (A(gᵖ⊙f))ᵢ`. Tổng **4 matmul mỗi vòng**.

**3. Clamp `g`.** Xem bảng trên.

**4. `mask_rel_floor` — lỗ hổng bắt được khi test.** Quantile là ngưỡng **tương
đối** nên luôn chọn ~10 % ô, **kể cả từ một map không có tín hiệu gì**. Đo được:
hạt rơi trên nền lan truyền về `max 0.0000`, nhưng `q90` của chính nó là `1.5e-05`,
nên `f > q90` vẫn trả về ô seed → thành một box 1×1 hoàn hảo. **2 vật thành 58 box.**
Floor hỏi thêm câu tuyệt đối: có ô nào đạt 5 % ô mạnh nhất ảnh không.

Cả bốn đều có test, kèm **negative control** (tắt guard thì test phải fail).

---

## ⚠️ Checkpoint: SD 1.5, KHÔNG phải SD2 như paper

Đo 2026-09-15: **cả dòng `stabilityai/stable-diffusion-2*` đã bị khoá trên HuggingFace.**
`stable-diffusion-2`, `-2-base`, `-2-1`, `-2-1-base` đều trả **HTTP 401** *"Invalid username
or password"* cho một repo từng công khai — đo từ **ba máy khác nhau** (Mac local, server
aiotlab, và một máy thứ ba). `401` chứ không phải `404` nghĩa là repo còn tồn tại nhưng đã
thành gated/private. Không phải lỗi mạng của ta, và thêm token cũng không gỡ được nếu chưa
xin quyền.

**SD 1.5 khác SD2 ở đâu** (đo từ `unet/config.json` của cả hai):

| | SD 1.5 | SD2 | ảnh hưởng |
|---|---|---|---|
| `sample_size` | 64 | 64 | **giống** → input 512, lưới latent 64×64 |
| `block_out_channels` | `[320,640,1280,1280]` | giống | **giống** |
| `up_block_types` | 3× `CrossAttnUpBlock2D` | giống | **giống** → `up_blocks.3.attentions.{0,1,2}` vẫn đúng |
| `cross_attention_dim` | **768** | 1024 | không chạm — ta hook `attn1`, không dùng cross-attn |
| `attention_head_dim` | **8** | 5 | không chạm — ta trung bình trên mọi head |

Nên `d2s/attention.py` chạy **không sửa một dòng nào**. M2N2 xác nhận điều này: hai file
aggregator SD1/SD2 của họ khác nhau **đúng 2 chỗ** — tên repo mặc định và
`attention_resolution` mặc định, toàn bộ logic hook giống hệt.

⚠️ **Thứ PHẢI đo lại**: `t=150` là giá trị Diffuse2Seg tinh chỉnh **cho SD2**. Thang timestep
của SD1.5 không nhất thiết đặt đặc trưng tốt nhất ở cùng chỗ → cửa chặn 0 có cờ `--timesteps`
để quét, dùng nó trước khi chốt.

### Tải checkpoint

```bash
# Ở LOCAL
cd object-detection/weights
hf download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir diffu2seg/stable-diffusion-v1-5 \
  --exclude "*.ckpt" "*.bin" "*.safetensors.index.json"

ls diffu2seg/stable-diffusion-v1-5/
# phải thấy: model_index.json  unet/  vae/  text_encoder/  tokenizer/  scheduler/
```

Rồi zip, tự upload lên server, giải nén vào
`/mnt/disk1/aiotlab/haitn/object-detection/weights/diffu2seg/stable-diffusion-v1-5/` — đúng
quy ước `weights/` của dự án (không `scp`/`rsync`).

`config.local_model_dir` trỏ sẵn vào đó: **có thư mục hợp lệ thì tự dùng, không có thì rơi về
tên repo HF**. Code kiểm `model_index.json` chứ không chỉ kiểm thư mục tồn tại — một thư mục
rỗng do giải nén hỏng sẽ bị bắt ngay thay vì chết sau đó với thông báo khó hiểu.

---

## Chạy

### Test trước (CPU, không cần GPU/SD)

```bash
python -m pytest tests/ -q       # ⭐ đây là phép thử CHÍNH
python tests/run_all.py          # bản tóm tắt theo checklist, chạy THÊM
```

⚠️ Dùng `python -m pytest`, **không** `pytest tests/` — cạm bẫy #9 của CLAUDE.md.

⚠️ **`run_all.py` KHÔNG thay được `pytest`, và đã chứng minh.** Nó gọi `main()`
của từng suite, mà `main()` có nhánh skip riêng khi thiếu dữ liệu; pytest gọi
**thẳng** từng hàm `test_*`, bỏ qua `main()`. Ngày 2026-09-15 `run_all.py` báo
13/13 xanh ở local trong khi `pytest` trên server cho **5 FAILED** — suite COCO,
vì server không có `data/coco/`. Đã sửa bằng `pytestmark` ở tầng module, nhưng
bài học giữ nguyên: **chạy cả hai, lệch nhau thì tin `pytest`**.

> **Bộ test xanh KHÔNG có nghĩa là phương pháp chạy được.** Nó chạy trên affinity
> giả lập, chỉ chứng minh phần toán và phần ghép nối đúng. SD2 thật có tách được
> vật trên CE-130 hay không là việc của cửa chặn 0.

### A. Diffuse2Seg training-free ĐẦY ĐỦ — cấu hình paper, đo AR₁₀₀₀

```bash
export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
LOG=/mnt/disk1/aiotlab/haitn/log/d2s_paper_$(date +%Y%m%d_%H%M%S).log
nohup python tools/run_on_free_gpu.py -- tools/run_paper.py \
    --dataset paco --limit 50 \
    --out /mnt/disk1/aiotlab/haitn/log/d2s_paper_paco.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

Đây là đường đầy đủ: **Bước 1 + Bước 2**, cả Algorithm 1 lẫn Algorithm 2.
`tools/run_paco.py` là bản GĐ1 (một mức), giữ lại để so — **không** phải đường
reproduce.

Nhìn hình trước khi tin số:

```bash
python tools/run_on_free_gpu.py -- tools/visualize_masks.py \
    --dataset paco --limit 8 --out-dir /mnt/disk1/aiotlab/haitn/log/d2s_paco_viz
```

Mốc in sẵn trong log: **Diffuse2Seg 13,6 | CutLER 10,7 | DiffSeg 9,8 | M2N2 9,6 |
UnSAM 9,3** — tất cả đều **có** multi-granularity.

### A'. COCO val2017 — chạy được nhưng không đối chiếu được

```bash
export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
LOG=/mnt/disk1/aiotlab/haitn/log/d2s_coco_$(date +%Y%m%d_%H%M%S).log
nohup python tools/run_on_free_gpu.py -- tools/run_coco.py --limit 50 \
    --out /mnt/disk1/aiotlab/haitn/log/d2s_coco.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

Nhìn hình trước khi tin số:

```bash
python tools/run_on_free_gpu.py -- tools/visualize_masks.py \
    --dataset coco --limit 8 --out-dir /mnt/disk1/aiotlab/haitn/log/d2s_coco_viz
```

**Chi phí**: `A` là (19600, 19600) fp32 = **1,54 GB** (so với 0,07 GB ở `grid_r=64`),
cộng SD1.5 fp16. Vừa A30 24 GB — nhưng **chỉ vì `compute_g` khai triển thành matmul**;
dạng literal ở đây là 815 TB. Ước tính ~20–40 s/ảnh nên mặc định `--limit 50`.

#### ⚠️ Hai thứ KHÔNG reproduce được

1. **Không có AP mask như bảng của paper.** Bảng đó là số của Mask2Former **đã train**
   trên pseudo-mask (bước 3) — ta bỏ hẳn bước train. Training-free không có score,
   không có ranking, nên **AP không định nghĩa được**. Đo được: `oracle_recall`,
   `mean_bestIoU`.
2. **SD 1.5 chứ không phải SD2.** Và quan trọng hơn cả việc đổi model: `t=150` là giá
   trị paper tinh chỉnh **cho SD2**. Thang timestep của SD1.5 không nhất thiết đặt đặc
   trưng tốt nhất ở cùng chỗ. Đây là nguồn sai khác thật, phải nói ra khi báo cáo.

#### Dữ liệu

`data/coco/` đã có sẵn trong repo: `annotations/instances_val2017.json` (19 MB) +
`val2017/` (5000 ảnh). Loader dùng **4952** ảnh — bỏ 48 ảnh chỉ có vùng `iscrowd`
hoặc không có annotation nào, vì tính chúng là 0 sẽ kéo mọi trung bình xuống vì lý do
không liên quan đến phương pháp.

⚠️ **COCO có ảnh dọc, CE-130 thì không.** CE-130 cao đúng 384 px và rộng ít nhất bằng
thế nên pad **luôn ở đáy**; COCO 427×640 pad ở **phải**. Vì thế có `data/coco_val.py`
riêng và tham số `valid_w` xuyên suốt `prompts`/`boxes`/`pipeline` (mặc định `1.0`,
nên hành vi trên CE-130 không đổi — có negative control trong test).

### Cửa chặn 0 — SD2 có tách được vùng không? (~5 phút)

```bash
export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
LOG=/mnt/disk1/aiotlab/haitn/log/d2s_gate0_$(date +%Y%m%d_%H%M%S).log
nohup python tools/run_on_free_gpu.py -- tools/check_attention_separates.py \
    --split val --limit 30 --timesteps 50 150 300 500 \
    --out /mnt/disk1/aiotlab/haitn/log/d2s_gate0.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

`run_on_free_gpu.py` chọn GPU trống nhất (server có 3× A30 dùng chung). `--timesteps`
quét t trong một lần chạy, ảnh và seed giữ nguyên nên **t là biến duy nhất**.

| ngưỡng | |
|---|---|
| **KHÔNG ĐẠT** | `AUC < 0,65` **HOẶC** chênh so lưới-đều-mù-ảnh `< 0,05` → **dừng** |
| **ĐẠT** | `AUC ≥ 0,78` **VÀ** chênh `≥ 0,10` |

Mốc **lưới đều mù ảnh** là quan trọng nhất: CE-130 vật nhỏ và dày nên một ma trận
chỉ phụ thuộc khoảng cách (không nhìn ảnh) cũng cho AUC cao. Tool tự đo mốc này.

### Cửa chặn 1 — `p=1,6` có thật hơn `p=2,0`? (~40–70 phút)

```bash
export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
LOG=/mnt/disk1/aiotlab/haitn/log/d2s_gate1_p_$(date +%Y%m%d_%H%M%S).log
nohup python tools/check_plaplacian_vs_p2.py --split val --limit 100 \
    --p-values 2.0 1.8 1.6 1.4 \
    --out /mnt/disk1/aiotlab/haitn/log/d2s_gate1_p.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

**Đây là cửa chặn cốt lõi.** Nếu `p<2` không hơn `p=2` thì `compute_g` là công vô
ích — giữ nhánh `p=2` một dòng, **xoá `compute_g`**, ghi kết luận âm. Đó là **kết
quả hợp lệ**, không phải thất bại.

### GĐ1 đầy đủ (~2–4 giờ) — chỉ khi cửa chặn 1 ĐẠT/XÁM

```bash
export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
LOG=/mnt/disk1/aiotlab/haitn/log/d2s_stage1_val_$(date +%Y%m%d_%H%M%S).log
nohup python tools/run_stage1.py --split val \
    --out /mnt/disk1/aiotlab/haitn/log/d2s_stage1_val.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

### Kiểm bằng mắt — làm TRƯỚC khi tin bất kỳ số nào

```bash
python tools/visualize_masks.py --split val --limit 8 \
    --out-dir /mnt/disk1/aiotlab/haitn/log/d2s_viz
```

Bài học §5 của `docs/02`: visualize bắt được **2 lỗi lớn mà toàn bộ test và 3
vòng rà soát code bỏ sót**.

---

## Đọc số cho đúng

**KHÔNG CÓ AP50, và đó là tính chất chứ không phải thiếu sót.** Training-free nên
không có score, không có ranking → AP không định nghĩa được. Bộ metric đúng là
`oracle_recall` + `mean_bestIoU` (bỏ qua score hoàn toàn).

**`oracle_recall` là TRẦN, không phải AP** — đừng so nó với cột AP50 của A/C1/E1.

**Cửa chặn đo trên `val`; số A/C1/E1/D.1 trong docs đo trên `test`** (30 vs 21
box/ảnh, và test có lô annotation hỏng chiếm 4,2 % GT) → **không so trực tiếp**.

**Trần do độ phân giải.** `r=64` → 1 ô = 8 px. Vật có cạnh ngắn dưới 1 ô **không
biểu diễn được**, bất kể `p`. Ảnh 1918×384 (aspect 4,99) bị thu `0,267×` nên vật
trung vị còn **1,30×1,07 ô**, round-trip IoU ~**0,35 dù mask hoàn hảo**. Hiếm
(**3,5 %** ảnh val có aspect > 2,0) nhưng `run_stage1.py` in trần này cùng kết
quả — nếu `oracle_recall` sát trần thì nút thắt là **độ phân giải**, không phải `p`.

⚠️ **D.1 (DiffusionDet, AP50 58,13) là trần của DỮ LIỆU, KHÔNG phải mục tiêu nên
nhắm** — xem `docs/01-bai-toan.md` mục 4.3.

---

## Log khi chạy nền

Mọi tool chạy dài in **cùng một khuôn**, cùng `fmt_time` với `CE-LocModel/train.py`:

```
  [   7/50  14.0%] 000000119876.jpg  427x640  n_gt= 10 pool= 847 -> n_pred= 612
                   hit@50=  4 | AR_1000= 8.31 | 28.4s/ảnh | elapsed 3m18s | ETA 20m21s
```

Mỗi **ảnh** một dòng (không phải mỗi 10 hay 25 ảnh) — job chạy vài giờ thì
`tail -f` phải thấy nó còn sống.

Đầu log in đủ để tái lập: timestamp, toàn bộ tham số theo từng bước, `model_source`,
device + tên GPU, `command`, và **cảnh báo nếu có cờ nào lệch khỏi `config/paper.py`**.
Sau ảnh đầu: `max_memory_allocated`, `n_prompts`, số cụm/mask theo từng mức.

Cuối log có **bảng phân rã thời gian** — một con số "s/ảnh" không cho biết nút thắt
ở đâu, mà ba khâu này chi phí rất khác nhau:

```
THỜI GIAN — 23m40s cho 50 ảnh (28.4s/ảnh)
    SD forward + affinity                 5m10s   21.8 %  (  6.20s/ảnh)
    lan truyền + Algorithm 2             14m50s   62.7 %  ( 17.80s/ảnh)
    tính IoU + AR                         3m00s   12.7 %  (  3.60s/ảnh)
    còn lại (đọc ảnh, giải mã GT)           40s    2.8 %  (  0.80s/ảnh)
  ⚙️  ngoại suy cả tập 2410 ảnh: 19h00m44s
```

Dòng ngoại suy để quyết định có chạy cả tập hay không **trước khi** bắt đầu.

---

## Rà soát đối chiếu paper + M2N2 (2026-09-15)

Rà từng dòng, đối chiếu Algorithm 1/2 của paper và code gốc M2N2.

### Khớp, đã kiểm bằng phép đo

| chỗ | cách kiểm |
|---|---|
| Algorithm 1 dòng 10–12 | khai triển đại số `Σⱼγᵢⱼfⱼ = gpᵢ(Af)ᵢ + (A(gp⊙f))ᵢ`; chỉ số `f @ A.T` = `Σⱼ Aᵢⱼfⱼ` kiểm trên ma trận **không đối xứng** |
| Algorithm 2 dòng 1–14 | KL khai triển == định nghĩa ngây thơ, **max diff 1,1e−15** |
| hook attention | `diff` với `refs/repos/m2n2/` → **giống hệt từng ký tự** |
| `change_temperature` | khớp `m2n2/src/utils.py`, thêm clamp chống `log(0)` |
| thứ tự thao tác | M2N2: extract → temperature → IPF. Ta: extract → temperature → renormalise row (bỏ IPF, có lý do) |
| `z ← VAE(I)`, không thêm nhiễu | Algorithm 1 dòng 2–3; code dùng `latent_dist.mode()` |
| ánh xạ lưới→pixel khi pad | tính tay: `valid_w=0,5`, lưới 4, ảnh 8px → 2 cột đầu phủ cả ảnh ✓ |

### Sai lệch tìm ra và đã sửa

1. **`w_up = 0,5/0,5`** (chép từ M2N2) → paper §4.1 là **`w₁=0,85 / w₂=0,15`**.
2. **NMS chậm 110×** — `np.stack` dựng lại mảng trong vòng lặp, không lọc hộp bao.
   Đo thật: 3000 mask 640×480 mất **~8775 s → 80 s**. Chỉ lộ ra ở quy mô thật.
3. **`mask_iou_matrix` VỠ khi 0 mask** — `reshape(0, -1)` ném ValueError. Ảnh mà mọi
   cụm đều dưới `a_min=100px` là chuyện **có thật**, và nó sẽ giết job giữa chừng.
4. **`_upsample_nearest` float64** → float32: 1,30 GB → 0,65 GB, lặp cho mỗi mức.

### Thông tin lấy thêm được từ paper

- **Hướng quét `t`**: §5 nói *"recall degrades across timesteps"* và `t=150` chọn để
  *"maximize mAP while keeping strong mAR"*. Ta chỉ đo AR → nếu AR thấp, quét
  **t < 150**, không phải lớn hơn.
- **Bảng 4a** (số mức): 3 → AR 16,3 | 5 → 17,7 | 6 → 18,1
- **Bảng 4b** (`τ_IoU`): 0,5 → 15,4 | 0,7 → 17,0 | 0,8 → 17,8 | 0,9 → **18,5**

### Chi phí đo thật (CPU, mask 640×480)

| | |
|---|---|
| `mask_iou_matrix` 1000×279 | 4,3 s, RSS đỉnh 2,8 GB |
| upsample 529 cụm | 6,0 s, 0,65 GB — **mỗi** mức, 6 mức |
| NMS 300 / 1000 / 3000 mask | 4 / 32 / 80 s |

---

## Ghi công

**Diffuse2Seg** — Hümmer, Sicking, Hüger, Gottschalk (CARIAD SE / TU Berlin),
arXiv 2609.06491. **Repo chính thức không tồn tại** (paper không có link code,
GitHub search trả về 0) → `d2s/plaplacian.py` tự viết lại từ Algorithm 1.

**M2N2** — Karmann & Urfalioglu, CVPR 2025
([vivoTechResearch/m2n2](https://github.com/vivoTechResearch/m2n2)).
`d2s/attention.py` **copy từ `refs/repos/m2n2/` rồi sửa** (4 chỗ: né `cv2`, thêm
`prompt_text`, nhận list timestep, `exit()` → `raise`). Không import từ `refs/repos/`.

**Không dùng của M2N2**: `matrix_ipf` (IPF chỉ phục vụ chuỗi Markov, p-Laplacian
có số hạng neo nên không cần), `create_semantic_markov_map_from_start_state`,
`flood_fill_with_min_threshold` (numba → thay bằng `scipy.ndimage.label`), JBU,
toàn bộ `m2n2_model.py` (4 hàm chấm điểm đọc nhãn người dùng bấm — vô nghĩa khi
không có người dùng).

Metric + box-ops **copy** từ `count_editing/CE-LocModel/` (sub-project không
import chéo nhau).
