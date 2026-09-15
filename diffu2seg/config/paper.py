"""CẤU HÌNH PAPER — mọi giá trị trích thẳng từ arXiv 2609.06491.

Không một con số nào ở đây là lựa chọn của dự án. Mỗi dòng ghi kèm mục của
paper nó đến từ đâu, để ai cũng kiểm lại được mà không phải đọc cả bài.

    tham số          giá trị        nguồn
    model            SD2            §4.1 "we mainly use the SD2 model"
    timestep         150            §4.1 "one step of denoising at timestep 150"
    canvas           1120x1120      §4.1 "resize the input images to 1120x1120"
    r (derived)      140            canvas / 8 (VAE stride)
    layers           2 layer cuối   §3.3 "only the self-attention tensors from the
                     ở decoder      first two layers at the highest resolution of
                                    the UNet decoder"
    w1 / w2          0.85 / 0.15    §4.1 "aggregation weights of w1 = 0.85 and
                                    w2 = 0.15"
    tau_att          0.55           §4.1 "temperature scaling with tau_att = 0.55"
    prompt spacing   6              §4.1 "distance between prompts to six pixels"
    p                1.6            §4.1 "propagation gradient exponent to p = 1.6"
    lambda           1e-5           §4.1 "anchoring penalty lambda = 1e-5"
    tau_prop         1e-4           §4.1 "stopping criterion tau_prop = 1e-4"
    L (mức)          6              §A.1 "L = 6 log-spaced thresholds"
    h range          [0.186, 2.99]  §A.1 "between hmin = 0.186 and hmax = 2.99"
    tau_IoU (NMS)    0.9            §A.1 "NMS at tau_IoU = 0.9"
    A_min            100 px         §A.1 "noise filtering at Amin = 100 px"
    N_max            1000           §3.4 "set to 1000"

                    BỐN CHỖ KHÔNG THỂ TRUNG THÀNH TUYỆT ĐỐI

1. **SD 1.5 thay SD2.** Mọi repo `stabilityai/stable-diffusion-2*` trả HTTP 401
   từ 2026-09-15 (đo từ ba máy). Đường hook không đổi: ta lấy `attn1`
   (self-attention ảnh<->ảnh) và trung bình trên MỌI head, còn SD1.5 và SD2 chỉ
   khác `cross_attention_dim` (768 vs 1024) và `attention_head_dim` (8 vs 5) —
   không cái nào chạm đường đó. M2N2 xác nhận: hai aggregator SD1/SD2 của họ
   khác nhau đúng 2 dòng.
   ⚠️ NHƯNG `t=150` LÀ GIÁ TRỊ TINH CHỈNH CHO SD2. Thang timestep của SD1.5
   không nhất thiết đặt đặc trưng tốt nhất ở cùng chỗ. Đây là sai khác thật.

2. **1120x1120 nằm NGOÀI độ phân giải gốc của cả hai model.** Paper nói rõ với
   SD2 (train ở 1024): "outside the native resolution of SD2 but still yields
   good segmentation results". SD 1.5 train ở **512**, nên 1120 là 2,2x — xa hơn
   nhiều so với 1,09x của SD2. Không có cách nào vừa giữ r=140 vừa ở trong vùng
   train của SD1.5. `config/paper_sd15_native.py` là biến thể cho ai muốn đo
   xem điều này ảnh hưởng bao nhiêu.

3. **Không có CascadePSP.** Paper gọi nó là "an optional refinement step".
   Table 1 (bảng so pseudo-label) đo KHÔNG có refinement: "comparing them
   against the baselines described in Sec. 4.1 without mask-refinement
   post-processing". Nên bỏ nó vẫn so được với Table 1/2.

4. **Không có bước 3 (train Mask2Former).** Ngoài phạm vi dự án, và không cần
   cho AR_1000 của nhãn.
"""

from .base import Diffu2SegConfig

cfg = Diffu2SegConfig(
    # --- Bước 1: trích self-attention -------------------------------------
    canvas=1120,                 # §4.1 -> grid_r = 140
    timesteps=(150,),            # §4.1
    # ⚠️ NẾU AR THẤP, QUÉT t XUỐNG chứ không lên. §5 (Fig. 5b): "our approach's
    # recall degrades across timesteps [...] because earlier time steps focus
    # more on fine texture, which favors our objective of segmenting objects at
    # varying granularities, whereas later steps capture coarse semantics".
    # Họ chọn 150 vì nó "maximizes mAP while keeping strong mAR" — tức 150 là
    # đánh đổi nghiêng về ĐỘ CHÍNH XÁC, t nhỏ hơn cho RECALL cao hơn. Ta chỉ đo
    # AR (không đo mAP), nên hướng quét đúng là t < 150.
    timestep_weights=(1.0,),     # một timestep
    w_down_0=0.0, w_down_1=0.0,  # §3.3: chỉ layer decoder độ phân giải cao nhất
    w_up_0=0.85,                 # §4.1 w1
    w_up_1=0.15,                 # §4.1 w2
    w_up_2=0.0,                  # §3.3 "first two layers" -> layer thứ ba không dùng
    tau_att=0.55,                # §4.1
    prompt_text="",              # class-agnostic

    # --- Bước 2a: lan truyền p-Laplacian ----------------------------------
    p=1.6,                       # §4.1
    lam=1e-5,                    # §4.1
    tau_prop=1e-4,               # §4.1
    max_iter=1000,               # paper không nêu trần; xem ghi chú base.py
    prompt_stride_cells=6,       # §4.1
    min_valid_frac=1.0,          # paper không có khái niệm pad (ảnh vuông)

    # --- Bước 2b: gộp map + NMS (GĐ2) --------------------------------------
    stage=2,
    n_levels=6,                  # §A.1 L = 6. Bảng 4a: 3 mức -> AR 16,3 |
                                 # 5 -> 17,7 | 6 -> 18,1. Tăng dần, chưa bão hoà.
    kl_h_min=0.186,              # §A.1 hmin
    kl_h_max=2.99,               # §A.1 hmax
    nms_iou=0.9,                 # §A.1 tau_IoU. Bảng 4b: 0,5 -> AR 15,4 |
                                 # 0,7 -> 17,0 | 0,8 -> 17,8 | 0,9 -> 18,5.
                                 # Càng lỏng càng tốt cho recall — đúng như
                                 # multi-granularity đòi hỏi (mask lồng nhau).
    min_area_px=100,             # §A.1 Amin
    max_masks=1000,              # §3.4 Nmax
).validate()
