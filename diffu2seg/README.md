# diffu2seg — Diffuse2Seg training-free trên CE-130

Port phần **training-free** của **Diffuse2Seg** ([arXiv 2609.06491](https://arxiv.org/abs/2609.06491),
6 Sep 2026 — CARIAD/VW + TU Berlin) cho chạy trên dữ liệu CE-130.

> **Mục đích: KHẢO SÁT CƠ CHẾ, chưa gắn vào pipeline CE-Loc.**
> Đây là công cụ để xem mask/box thực tế ra sao, **không** phải ứng viên cho
> toán tử `T` của [../../docs/01-bai-toan.md](../../docs/01-bai-toan.md).

**Không có phần train Mask2Former** (bước 3 của paper) — bỏ hoàn toàn.

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

## Khác paper ở đâu, và vì sao

| tham số | paper | ở đây | lý do (số đo trên CE-130) |
|---|---|---|---|
| `grid_r` | 140 | **64** | A: 1,54 GB → **0,07 GB** (22× nhẹ). `8×64 = 512` = canvas chuẩn dự án |
| `prompt_stride` | 6 ô | **3 ô** | đo 200 ảnh val: s=3 phủ **96,6 %** box; s=6 bước qua vật nhỏ (box nhỏ nhất trung vị **2,24 ô**) |
| lọc vùng pad | không có | **có** | ảnh CE-130 cao 384 cố định, rộng tới 1918 → pad ~**29 %** canvas. Paper dùng ảnh vuông |
| `g_eps` | **không có** | `1e-8` | `g=0` ở vùng phẳng, `p−2 = −0,4` → `0^(−0,4) = inf` → NaN |
| `mask_rel_floor` | **không có** | `0,05` | xem "Ba thứ paper không viết" dưới |
| timestep | 2 cái trộn | **1 cái** | trộn là **biến thứ hai**; quy tắc dự án: mỗi bước đổi một biến |

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
python tests/run_all.py          # 7 suite, 68 test
# hoặc
python -m pytest tests/ -q
```

⚠️ Dùng `python -m pytest`, **không** `pytest tests/` — xem cạm bẫy #9 của CLAUDE.md.

> **Bộ test xanh KHÔNG có nghĩa là phương pháp chạy được.** Nó chạy trên affinity
> giả lập, chỉ chứng minh phần toán và phần ghép nối đúng. SD2 thật có tách được
> vật trên CE-130 hay không là việc của cửa chặn 0.

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
