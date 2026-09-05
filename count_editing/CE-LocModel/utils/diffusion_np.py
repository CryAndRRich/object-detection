"""Phần toán diffusion — numpy thuần, KHÔNG phụ thuộc torch.

NGUỒN CHÂN LÝ cho `q_sample` / DDIM / `prepare_diffusion_concat`. Bản torch phải
port cơ học từ đây kèm test `allclose(torch_fn, np_fn)`.

BỐN LỖI SỐ HỌC CỦA VÒNG 1 — mỗi cái đều có test âm bản trong tests/test_math.py:

  1. KHÔNG clamp `pred_x_start` trước khi dùng.
     Hệ số 1/sqrt(alpha_bar) đạt 157x ở t=999 (>10x trên 33 % timestep) nên box
     trong [-1,1] văng ra ±301, phá cả cost matrix lẫn loss.
     (DiffusionDet clamp ở detector.py:181)

  2. KHÔNG tính lại `pred_noise` từ `x_start` ĐÃ clamp.
     Bỏ bước này làm sai số DDIM tăng từ ~2e-7 lên ~5,4.
     (DiffusionDet: detector.py:182)

  3. Khởi tạo x_T nhân thêm `snr_scale`.
     Phải là N(0, I) std 1,0 — ở t=T-1 thì alpha_bar = 4e-5.
     (DiffusionDet dùng `randn` trơn: detector.py:197)

  4. Loss trên epsilon dưới set-matching.
     Vô nghĩa: sau khi matcher gán prediction p sang GT g, không tồn tại epsilon
     nào vừa thuộc p vừa ứng với g. `SetCriterionDynamicK` chỉ có L1+GIoU trên x0.

GIỚI HẠN ĐÃ ĐO của việc sửa placeholder w/h (docs §(a)): model nhìn thấy
`x_t = q_sample(x_start)`, không phải `x_start`. Clamp kéo mọi thứ về median 0,5
khi t lớn (t=999: median w giải mã = 0,500; 79 % có w>0,3). Nên sửa placeholder
CHỈ có tác dụng ở t nhỏ — vẫn cần, nhưng đừng kỳ vọng quá.
"""

import numpy as np

from .box_ops_np import decode_diffusion, encode_diffusion

__all__ = [
    "cosine_alphas_cumprod",
    "linear_alphas_cumprod",
    "q_sample",
    "predict_noise_from_start",
    "ddim_step",
    "ddim_time_pairs",
    "make_placeholders",
    "prepare_diffusion_concat",
]


# --------------------------------------------------------------------------
# Beta schedule
# --------------------------------------------------------------------------

def cosine_alphas_cumprod(num_timesteps=1000, s=0.008):
    """Cosine schedule (Nichol & Dhariwal), đúng như DiffusionDet dùng.

    Đo được vòng 1: cosine cho 3,70x AP so với linear. Lý do: ở t lớn linear chỉ
    còn 5,8 % tín hiệu (t=749) nên box gần như thuần nhiễu, matcher gán bừa và
    gradient nhiễu gấp 4,6 lần.

        sqrt(alpha_bar) còn lại   t=249    t=499    t=749
        cosine                    0,92     0,70     0,38
        linear                    0,72     0,28     0,058
    """
    t = np.arange(num_timesteps + 1, dtype=np.float64)
    f = np.cos(((t / num_timesteps) + s) / (1.0 + s) * np.pi / 2.0) ** 2
    ab = f / f[0]
    betas = np.clip(1.0 - (ab[1:] / ab[:-1]), 0.0, 0.999)
    return np.cumprod(1.0 - betas)


def linear_alphas_cumprod(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """Linear schedule — CHỈ để đối chiếu trong test, KHÔNG dùng để train."""
    betas = np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)
    return np.cumprod(1.0 - betas)


# --------------------------------------------------------------------------
# Forward process
# --------------------------------------------------------------------------

def q_sample(x_start, t, noise, alphas_cumprod):
    """x_t = sqrt(ab_t) * x_0 + sqrt(1 - ab_t) * eps.

    `t` là MỘT giá trị vô hướng cho cả ảnh (không phải mỗi box một t) — N box là
    *một* mẫu từ phân phối tập box; cho mỗi box một t thì lúc inference không tái
    tạo được. DiffusionDet: `torch.randint(..., (1,))`.
    """
    ab = float(alphas_cumprod[int(t)])
    return np.sqrt(ab) * np.asarray(x_start, dtype=np.float64) + np.sqrt(1.0 - ab) * np.asarray(
        noise, dtype=np.float64
    )


def predict_noise_from_start(x_t, t, x_start, alphas_cumprod):
    """Suy ngược eps từ (x_t, x_0). Nghịch đảo giải tích của `q_sample`.

    Đây là lý do dự đoán x_0 KHÔNG mất gì so với dự đoán eps: biết x_t và t thì
    hai đại lượng xác định lẫn nhau qua một phương trình tuyến tính.
    """
    ab = float(alphas_cumprod[int(t)])
    sqrt_recip = np.sqrt(1.0 / ab)
    sqrt_recipm1 = np.sqrt(1.0 / ab - 1.0)
    return (sqrt_recip * np.asarray(x_t, dtype=np.float64) - np.asarray(x_start, dtype=np.float64)) / sqrt_recipm1


# --------------------------------------------------------------------------
# Reverse process (DDIM)
# --------------------------------------------------------------------------

def ddim_time_pairs(num_timesteps=1000, sampling_steps=4):
    """[(T-1, T-2), ..., (1, 0), (0, -1)] — giống DiffusionDet detector.py:191."""
    times = np.linspace(-1, num_timesteps - 1, sampling_steps + 1)
    times = list(reversed(times.astype(int).tolist()))
    return list(zip(times[:-1], times[1:]))


def ddim_step(x_t, x_start_raw, t, t_next, alphas_cumprod, snr_scale=2.0, eta=1.0, noise=None):
    """Một bước DDIM. Trả về (x_next, x_start_đã_clamp).

    Thứ tự BẮT BUỘC (lỗi 1 + 2 ở đầu file):
      1. clamp x_start về miền hợp lệ
      2. TÍNH LẠI pred_noise từ x_start ĐÃ clamp
      3. mới đi bước DDIM

    Khi `t_next < 0` thì trả thẳng x_start (bước cuối, không thêm nhiễu).
    """
    s = float(snr_scale)
    # (1) clamp — qua decode/encode để dùng đúng một định nghĩa miền hợp lệ
    x_start = encode_diffusion(decode_diffusion(x_start_raw, s), s)

    if t_next < 0:
        return x_start, x_start

    # (2) tính LẠI pred_noise từ x_start đã clamp
    pred_noise = predict_noise_from_start(x_t, t, x_start, alphas_cumprod)

    ab_t = float(alphas_cumprod[int(t)])
    ab_next = float(alphas_cumprod[int(t_next)])
    sigma = eta * np.sqrt((1 - ab_t / ab_next) * (1 - ab_next) / (1 - ab_t))
    c = np.sqrt(max(1 - ab_next - sigma ** 2, 0.0))

    if noise is None:
        noise = np.random.standard_normal(np.shape(x_t))
    x_next = x_start * np.sqrt(ab_next) + c * pred_noise + sigma * noise
    return x_next, x_start


# --------------------------------------------------------------------------
# prepare_diffusion_concat — pad/crop GT lên N proposal
# --------------------------------------------------------------------------

def make_placeholders(n, median_wh=None, valid_h=1.0, rng=None):
    """Sinh `n` box giả trong hệ chuẩn cxcywh [0,1].

    Gốc DiffusionDet: `randn/6 + 0.5` cho CẢ 4 chiều — là Gaussian N(0,5; 1/6),
    comment gốc ghi "3sigma = 1/2". Hai sửa cho CE-130:

    SỬA 1 — w/h theo dữ liệu. Gốc cho w/h ~ 0,5 = nửa ảnh, trong khi vật CE-130
      median 0,069 => placeholder to gấp 7,3x. Với N=100 và ~37,6 GT thì 62 %
      slot là placeholder, model học prior "box thì to và ở giữa". Thay bằng
      log-normal quanh trung vị vật thật CỦA CHÍNH ẢNH ĐÓ (7,3x -> ~0,8x).

    SỬA 1b — chặn cy trong vùng ảnh thật. Đo được 13,7 % placeholder có tâm rơi
      vào vùng pad (ảnh nặng nhất 92,2 %). Rẻ vì mọi ảnh CE-130 cao đúng 384px
      nên luôn W>=H, tức chỉ pad ở DƯỚI -> một ngưỡng vô hướng là đủ.

    Tâm cx giữ nguyên Gaussian gốc (ảnh luôn rộng hết canvas).
    """
    rng = np.random.default_rng() if rng is None else rng
    out = np.empty((n, 4), dtype=np.float64)

    out[:, 0] = rng.standard_normal(n) / 6.0 + 0.5                      # cx: gốc
    out[:, 1] = np.clip(rng.standard_normal(n) / 6.0 + 0.5, 0.0, 1.0)   # cy: gốc
    out[:, 1] *= max(valid_h, 1e-6)                                      # SỬA 1b

    if median_wh is None:                                                # gốc DiffusionDet
        out[:, 2:] = np.clip(rng.standard_normal((n, 2)) / 6.0 + 0.5, 1e-4, None)
    else:                                                                # SỬA 1
        mw, mh = float(median_wh[0]), float(median_wh[1])
        sigma = 0.4  # độ tản của log-normal, ~1 bát phân
        out[:, 2] = np.clip(mw * np.exp(rng.standard_normal(n) * sigma), 1e-4, 1.0)
        out[:, 3] = np.clip(mh * np.exp(rng.standard_normal(n) * sigma), 1e-4, 1.0)
    return out


def prepare_diffusion_concat(
    gt_boxes, num_proposals, t, alphas_cumprod, snr_scale=2.0,
    valid_h=1.0, adapt_placeholder=True, rng=None,
):
    """GT [M,4] cxcywh[0,1] -> (x_t [N,4], noise [N,4], is_gt [N] bool).

    Pad bằng placeholder khi M < N, crop ngẫu nhiên khi M > N. `is_gt` đánh dấu
    slot nào là GT thật (chỉ những slot đó vào loss toạ độ qua matcher).

    N=100 cắt mất GT ở 8-11 % ảnh (test 11,2 %) -> eval nên dùng N=300. Kiến trúc
    cho phép vì đã bỏ pos_emb theo chỉ số.
    """
    rng = np.random.default_rng() if rng is None else rng
    gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    m = gt.shape[0]

    if m == 0:
        x_start_norm = make_placeholders(num_proposals, None, valid_h, rng)
        is_gt = np.zeros(num_proposals, dtype=bool)
    elif m < num_proposals:
        med = (np.median(gt[:, 2]), np.median(gt[:, 3])) if adapt_placeholder else None
        ph = make_placeholders(num_proposals - m, med, valid_h, rng)
        x_start_norm = np.concatenate([gt, ph], axis=0)
        is_gt = np.zeros(num_proposals, dtype=bool)
        is_gt[:m] = True
    else:
        idx = rng.permutation(m)[:num_proposals]
        x_start_norm = gt[idx]
        is_gt = np.ones(num_proposals, dtype=bool)

    x_start = encode_diffusion(x_start_norm, snr_scale)
    noise = rng.standard_normal((num_proposals, 4))
    x_t = np.clip(q_sample(x_start, t, noise, alphas_cumprod), -snr_scale, snr_scale)
    return x_t, noise, is_gt
