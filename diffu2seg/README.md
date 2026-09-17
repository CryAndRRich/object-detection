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

## KẾT QUẢ — PACO val, 150 ảnh (2026-09-16)

`run_paper.py --dataset paco --limit 150`, **cấu hình paper không lệch một tham số
nào** trừ `t` và `w1` (xem mục quét bên dưới), SD 1.5, A30, 41m37s (16,7 s/ảnh).

| | ta | paper | các baseline của paper |
|---|---|---|---|
| **AR₁₀₀₀** | **11,80** | **13,6** | CutLER 10,7 · DiffSeg 9,8 · M2N2 9,6 · UnSAM 9,3 |
| AR_S / AR_M / AR_L | 4,83 / 24,01 / 33,75 | — | n = 1446 / 539 / 160 |
| AR OBJECT / PART | 17,58 / 10,47 | — | n = 772 / 1373 |
| recall@0,50 / @0,75 | 27,79 / 8,76 | — | |

**Đạt 87 % con số của paper, vượt cả ba baseline training-free** dù chạy SD 1.5
thay SD2. Lần đo trước (50 ảnh) cho 10,36; chênh lệch chủ yếu là **sai số lấy mẫu**,
không phải cải thiện — xem mục quét.

### ⚠️ Quét `t` × `w1`: 16 cấu hình KHÔNG phân biệt được

Quét 150 ảnh đầu (10h07m), rồi **xác nhận lại trên 150 ảnh KHÁC** (ảnh 150–300, 2h39m):

| t | w1 | ảnh 0–150 | ảnh 150–300 |
|---|---|---|---|
| 150 | 0,15 | 11,66 (hạng 2) | **12,95 (hạng 1)** |
| 50 | 0,85 | 11,25 (hạng 3) | 12,63 (hạng 2) |
| 50 | 0,15 | **11,80 (hạng 1)** | 12,60 (hạng 3) |
| 150 | 0,85 (paper) | 11,17 (hạng 4) | 12,26 (hạng 4) |

- biên độ **giữa các cấu hình**: 0,63 p.p. (sweep) / 0,69 p.p. (confirm)
- dịch chuyển **giữa hai tập ảnh**: **+1,14 p.p.** — cả bốn cấu hình đều lên
- bootstrap 5000 lần trên 150 ảnh: recall@50 có sd **3,55 p.p.**, CI 95% rộng 13,5 p.p.

**Tập ảnh dịch mạnh gấp đôi khoảng cách giữa các cấu hình ⇒ `t` không phân biệt được
ở n=150.** Cấu hình thắng ở sweep tụt xuống hạng 3 khi đổi tập. Đây là lý do phải có
bước xác nhận trên tập rời.

Thứ **có** sống sót: **`w1` = 0,15 đứng nhất ở cả 4 giá trị `t` trong sweep gốc** — 16
điểm nhất quán, không phải một cặp sát nhau. Tức `up_blocks.3.attentions.1` mang tín
hiệu tốt hơn `.0` trên PACO, **ngược** với 0,85/0,15 mà paper đo trên SA-1B và ngược
với 0,5/0,5 của M2N2. `t=300` tệ ở mọi `w1`, khớp ghi chú "quét t XUỐNG chứ không lên".

⇒ **Khuyến nghị: dùng `t=150` của paper** (không có bằng chứng để đổi), `w1=0,15`.

### Chẩn đoán — sáu mức granularity CÓ hoạt động

```
   height   cụm/ảnh  mask/ảnh
    0.186      79.3     210.6
    0.324      59.0     178.0
    0.565      41.9     142.3
    0.984      28.3     104.2
    1.716      17.7      71.3
    2.990       9.6      38.8
```

Số cụm giảm đều 79,3 → 9,6 qua 6 mức. **Dải `[0,186 ; 2,99]` của paper khớp tốt với
thang KL của SD 1.5** — đây là rủi ro lớn nhất khi đổi model, và nó đã không xảy ra.

Hội tụ **100 %** (n_iter trung vị 28/1000, max 50). NMS 745,3 vào → 496,0 giữ.
Cap `N_max=1000` chỉ chạm ở **2/150 ảnh** — không phải nút thắt.

### Ba chỗ mất điểm, theo thứ tự đáng làm

**1. PACO hỏi PART, phương pháp trả OBJECT.** AR_part 10,47 vs AR_object 17,58.
65,6 % annotation của PACO là bộ phận. Quan hệ số vật/ảnh với recall (cả 150 ảnh):

| n_gt | số ảnh | recall TB | % tổng GT |
|---|---|---|---|
| 1–3 | 42 | 0,544 | 4,3 % |
| 4–10 | 49 | 0,445 | 14,5 % |
| 11–30 | 45 | 0,331 | 34,5 % |
| 31+ | 14 | 0,219 | 46,6 % |

Tương quan `log(n_gt)` vs recall: **−0,398**. Ảnh có ≥11 GT chiếm **39 % số ảnh nhưng
81 % tổng GT** ⇒ AR₁₀₀₀ bị chi phối bởi nhóm ảnh dày đặc phần lớn là PART — đúng nhóm
mà một phương pháp phân vùng theo **vật** yếu nhất. Đây là phần lớn khoảng cách tới 13,6,
và **không phải lỗi implement**.

**2. Biên mask thô ở mức ô 8 px.** recall 27,79 (IoU 0,5) → 8,76 (IoU 0,75), mất 2/3.
Đúng chỗ **CascadePSP** (Step 3, ta bỏ) sinh ra để sửa.

**3. Độ phân giải `r=140`** — chỉ giải thích ~18 %: trần TB có trọng số 0,818 vs recall
thực 0,278, khoảng cách 0,540 là chất lượng mask chứ không phải độ phân giải.

| trần độ phân giải | số ảnh | recall@50 |
|---|---|---|
| < 0,50 | 2 | 0,018 |
| 0,50–0,80 | 17 | 0,157 |
| 0,80–0,95 | 18 | 0,314 |
| ≥ 0,95 | 113 | 0,480 |

### Nhìn ảnh xác nhận chẩn đoán

`d2s_viz_spread/0.00_000000017235.png` (nhà tắm, recall **0,00**): phân vùng rất hợp lý
— bồn rửa, bệ toilet, khăn, mặt bàn, từng ô gạch đều tách đúng. Nhưng PACO chỉ annotate
**2 vật**, 1 là PART. 479 mask, không cái nào đạt IoU 0,5 với đúng 2 GT đó.

`0.03_000000029558.png` (thả diều): **30 GT, 21 là PART** — tay áo, ống quần, vành mũ.
Model tách người/diều/bãi cỏ sạch sẽ, nhưng nó phân theo **vật**, GT hỏi **bộ phận**.

⇒ **Mask tốt hơn con số 11,80 gợi ý.** Con số thấp phản ánh lệch giữa thứ đo và thứ
sinh ra, không phải chất lượng phân vùng.

### Biến đáng quét tiếp, theo thứ tự

1. **CascadePSP** — nhắm thẳng vào chỗ mất điểm #2, là chỗ duy nhất còn nhiều điểm.
2. **`--limit` lớn hơn** — mọi kết luận về `t` hiện đều nằm trong nhiễu ở n=150.
   Cả tập 2410 ảnh mất ~11h09m.
3. **`t` và `w1`: KHÔNG đáng quét thêm** ở n=150 — đã chứng minh nằm trong nhiễu.

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

## ⚠️ Bộ nhớ ở `r=140` — ba lần vỡ, ba nguyên nhân khác nhau

Ở `r=64` (M2N2) mọi thứ nhỏ nên không ai thấy vấn đề. Ở `r=140` của paper,
`N = 19 600` và `N²` fp16 = **0,77 GB**, `(B,H,N,N)` 8 head = **6,15 GB**. Đỉnh đo
được đi từ **23,87 GB → 13,86 GB** qua ba lần sửa, mỗi lần một nguyên nhân khác:

**1. `x.float()` trên cả tensor** (2026-09-15). M2N2 viết
`torch.mean(torch.mean(x.float(), dim=1), dim=0)` — ở `r=140` dựng bản fp32
**12,3 GB**. Sửa: cộng dồn từng head vào bộ đệm fp32 `(N,N)` = 1,54 GB.

**2. Wrap MỌI block attention** (2026-09-16). `sd2_inject_attention_wrappers` wrap
cả 32 module `Attention` của UNet, mà wrapper thay SDPA của PyTorch bằng bản viết
tay — và bản viết tay **bắt buộc** dựng `(B,H,N,N)` tường minh (SDPA gốc/flash
attention không bao giờ dựng). `down_blocks.0` có **cùng 19 600 token** với
`up_blocks.3`, nên mỗi ảnh cấp phát rồi vứt **3 tensor 6,15 GB** thừa
(`down_blocks.0.attentions.{0,1}` và `up_blocks.3.attentions.2`, đều `weight=0`).
Sửa: `active_paths` — chỉ wrap block có trọng số khác 0 (2/5 thay vì 5/5).

**3. Cả `(B,H,N,N)` sống suốt lúc callback chạy.** Callback được gọi TỪ TRONG
`scaled_dot_product_attention` nên `x` chưa được giải phóng. Sửa: vòng theo head,
mỗi lúc chỉ một `(N,N)` fp16 sống (6,15 → 0,77 GB). Bật khi tensor
> `CHUNK_THRESHOLD_ELEMS` (4e8); dưới ngưỡng giữ đường cũ vì một matmul lớn nhanh hơn.
Kết quả **giống hệt**: softmax chạy trên chiều cuối nên độc lập theo head, và
`attn_weight @ value` cũng tách được theo head — không phải xấp xỉ.

Thêm `torch.cuda.empty_cache()` + giải phóng `A` giữa các ảnh trong `visualize_best.py`.
Đỉnh giữ nguyên 13,86 GB suốt 40 ảnh ⇒ không rò rỉ.

### ⚠️ Hai lỗi ÂM THẦM sinh ra trong lúc sửa — cả hai đều có test

**a. Chia `B*H` của riêng lần gọi.** Callback cũ chia cho `B*H` của tensor nhận
được. Khi SDPA gọi từng head thì `B*H == 1` ⇒ attention tích luỹ **gấp 8 lần**.
Shape đúng, không NaN, mask vẫn sinh ra, chỉ sai thang. Sửa: callback cộng **tổng
thô** + đếm head; chia trung bình và nhân trọng số dời sang `_finish_layer()` gọi ở
ranh giới layer. Đo được tỉ lệ đúng **8,00**.

**b. `.float()` trả VIEW khi dtype đã đúng.** `head = x[b,h].float()` chỉ **sao chép
khi dtype đổi**. Với `x` đã fp32 nó trả về view, nên `_raw_acc` trỏ thẳng vào
`x[0,0]`; các head sau `add_` vào bộ đệm = **ghi đè lên `x[0,0]`**, rồi SDPA dùng
chính `x` đó cho `attn_weight @ value`. Kết quả: **head 0 sai, head 1–7 đúng**
(đo: lệch 4,97 ở phần tử đầu, trùng khít ở phần tử cuối).
⚠️ **Đường chạy thật KHÔNG lộ lỗi này** vì `x` là fp16 → `.float()` có sao chép.
Chỉ hiện khi fp32. Sửa: `.to(torch.float32, copy=True)`.

`tests/test_sdpa_chunked.py` (4 test) phủ cả hai, gồm negative control cho mỗi lỗi.
⚠️ Bản test ĐẦU TIÊN **chép lại** vòng lặp head trong chính test rồi so với nhánh
một-lần — tức kiểm bản chép, không kiểm code thật, và xanh giả. Vì thế
`CHUNK_THRESHOLD_ELEMS` là **thuộc tính lớp** chứ không phải hằng số viết thẳng:
test hạ ngưỡng để chạy ĐÚNG nhánh chunked của code thật.

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
    --out /mnt/disk1/aiotlab/haitn/output/d2s_paper_paco.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

Đây là đường đầy đủ: **Bước 1 + Bước 2**, cả Algorithm 1 lẫn Algorithm 2.
`tools/run_paco.py` là bản GĐ1 (một mức), giữ lại để so — **không** phải đường
reproduce.

Nhìn hình trước khi tin số:

```bash
python tools/run_on_free_gpu.py -- tools/visualize_masks.py \
    --dataset paco --limit 8 --out-dir /mnt/disk1/aiotlab/haitn/output/d2s_paco_viz
```

Mốc in sẵn trong log: **Diffuse2Seg 13,6 | CutLER 10,7 | DiffSeg 9,8 | M2N2 9,6 |
UnSAM 9,3** — tất cả đều **có** multi-granularity.

### A'. COCO val2017 — chạy được nhưng không đối chiếu được

```bash
export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
LOG=/mnt/disk1/aiotlab/haitn/log/d2s_coco_$(date +%Y%m%d_%H%M%S).log
nohup python tools/run_on_free_gpu.py -- tools/run_coco.py --limit 50 \
    --out /mnt/disk1/aiotlab/haitn/output/d2s_coco.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

Nhìn hình trước khi tin số:

```bash
python tools/run_on_free_gpu.py -- tools/visualize_masks.py \
    --dataset coco --limit 8 --out-dir /mnt/disk1/aiotlab/haitn/output/d2s_coco_viz
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
    --out /mnt/disk1/aiotlab/haitn/output/d2s_gate0.json > "$LOG" 2>&1 &
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
    --out /mnt/disk1/aiotlab/haitn/output/d2s_gate1_p.json > "$LOG" 2>&1 &
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
    --out /mnt/disk1/aiotlab/haitn/output/d2s_stage1_val.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

### Kiểm bằng mắt — làm TRƯỚC khi tin bất kỳ số nào

```bash
python tools/visualize_masks.py --split val --limit 8 \
    --out-dir /mnt/disk1/aiotlab/haitn/output/d2s_viz
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

## Quét tham số — `t` và `w1`

Cả hai đều được paper chọn để tối ưu **mAP**, mà ta **chỉ đo AR**:

| | paper | lý do paper chọn | hướng quét |
|---|---|---|---|
| `t` | 150 | *"maximizes mAP while keeping strong mAR"*; §5: *"recall degrades across timesteps"* | **t nhỏ hơn** |
| `w1` | 0,85 | §A.1: *"lies on the **precision** plateau"* | cả hai phía |

⚠️ `0,85/0,15` **không phải "giá trị của SD2"** — đó là số Diffuse2Seg tự đo trên
SA-1B holdout (M2N2 dùng 0,5/0,5). Nhưng phép đo đó làm trên SD2, ta chạy SD 1.5.

```bash
export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache
LOG=/mnt/disk1/aiotlab/haitn/log/d2s_sweep_$(date +%Y%m%d_%H%M%S).log
nohup python tools/run_on_free_gpu.py -- tools/sweep_t_w1.py \
    --limit 20 --timesteps 50 100 150 300 --w1 1.0 0.85 0.5 0.15 \
    --out /mnt/disk1/aiotlab/haitn/output/d2s_sweep.json > "$LOG" 2>&1 &
echo "PID $! -> $LOG"
```

**Vì sao quét được**: đo thật cho thấy SD chỉ chiếm **11,5 %** thời gian (2,3 s/ảnh),
còn lan truyền + Algorithm 2 chiếm 82,9 %. Nên `t` phải chạy lại SD, nhưng `w1`
thì **không** — tool trích từng layer riêng một lần (`extract_per_layer`) rồi trộn
ngoài. 16 cấu hình × 20 ảnh ≈ **1 giờ 40 phút**.

⚠️ **Thứ tự chuẩn hoá quan trọng**: đường chạy thật trộn layer **trước** rồi chuẩn
hoá tổng **một lần**. Chuẩn hoá từng layer rồi mới trộn cho kết quả **khác** (đo:
max diff 7,8e-3) — nên `extract_per_layer` trả tensor **thô**. Đã kiểm khớp tuyệt
đối (diff = 0) ở 4 giá trị `w1`.

⚠️ **20 ảnh để CHỌN, không để BÁO số.** Chọn cấu hình trên cùng tập dùng để đánh
giá là overfit tập đó — xác nhận lại bằng `--start` khác trước khi tin.

## Nhìn ảnh — `visualize_best.py`

```bash
python tools/run_on_free_gpu.py -- tools/visualize_best.py \
    --from-json /mnt/disk1/aiotlab/haitn/output/d2s_paper_paco.json \
    --limit 30 --pick spread \
    --out-dir /mnt/disk1/aiotlab/haitn/output/d2s_viz
```

Đọc `per_image` của JSON, xếp ảnh theo `recall@50`, rồi lấy `--pick`:

| | |
|---|---|
| `spread` | đều khắp dải — **mặc định**, trung vị khớp cả tập (kiểm: 0,27 vs 0,27) |
| `best` | 30 ảnh tốt nhất — trung vị 0,49, **gần gấp đôi cả tập** |
| `worst` | 30 ảnh tệ nhất — để tìm chỗ hỏng |

⚠️ `best` là **mẫu chọn lọc thiên vị**: 30 ảnh tốt nhất trong 50 luôn trông thuyết
phục, kể cả khi trung vị thật chỉ 0,27. Tool in cả hai trung vị lên đầu log và
ghi recall vào **tên file**, nên `ls` đã là bảng xếp hạng.

Ba panel mỗi ảnh: ảnh gốc | mask pred (mỗi instance một màu) | box pred so GT
(lá đậm = GT tìm thấy, lá nhạt đứt nét = trượt, cam = pred khớp, đỏ = pred thừa).

---

## Nơi ghi file trên server

| loại | thư mục |
|---|---|
| **log** của job nền (`> "$LOG"`) | `/mnt/disk1/aiotlab/haitn/log/` |
| **kết quả** — `.json`, ảnh visualize (`--out`, `--out-dir`) | `/mnt/disk1/aiotlab/haitn/output/` |

Tách hai thứ vì chúng có vòng đời khác nhau: log là thứ đọc khi job đang chạy
hoặc khi nó vỡ, còn `.json` là kết quả cần giữ và đối chiếu về sau.

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
