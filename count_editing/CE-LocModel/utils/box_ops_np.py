"""Phép toán trên bounding box — numpy thuần, KHÔNG phụ thuộc torch.

File này là NGUỒN CHÂN LÝ cho mọi biến đổi toạ độ. Bản torch (`box_ops.py`) phải
port cơ học từ đây và có test `allclose(torch_fn, np_fn)`.

HỆ CHUẨN DUY NHẤT: cxcywh trong [0, 1] trên canvas vuông (mặc định 512).
- `w > 0` luôn đúng, đọc thẳng ra "rộng bao nhiêu phần ảnh".
- `snr_scale` CHỈ tồn tại bên trong phần diffusion, không rò ra ngoài.

Chuỗi 6 bước (xem docs/thiet-ke-ce-loc-vong-2.md §4.7):

    all_bboxes: xyxy pixel tuyệt đối trên ảnh gốc (W x H)
      (1) xyxy -> cxcywh
      (2) x scale = min(target/W, target/H)      # pad ở DƯỚI, không dịch toạ độ
      (3) / target -> [0, 1]
    ===> cxcywh [0,1]  (HỆ CHUẨN)
      (4) (x*2 - 1) * snr_scale                  # chỉ trong diffusion
    ===> x_start, dùng cho q_sample / DDIM
      (5) clamp(±snr) -> /snr -> (x+1)/2         # về lại [0,1]
      (6) cxcywh -> xyxy                         # chỉ để tính GIoU

LỖI VÒNG 1 (đừng lặp): giải mã extent là `(norm+1)/2`, KHÔNG phải `(norm+1)/4`.
Chia 2 lần thì box còn nửa chiều rộng, IoU giữa đúng và sai chỉ 0,25.
"""

import numpy as np

__all__ = [
    "compute_scale",
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "scale_to_canvas",
    "canvas_to_pixel",
    "encode_diffusion",
    "decode_diffusion",
    "box_iou",
    "generalized_box_iou",
    "pairwise_l1",
    "flip_horizontal",
    "filter_degenerate",
]


# --------------------------------------------------------------------------
# Bước (1) và (6): đổi định dạng, KHÔNG đổi thang đo
# --------------------------------------------------------------------------

def xyxy_to_cxcywh(boxes):
    """[..., 4] (x1,y1,x2,y2) -> (cx,cy,w,h). Giữ nguyên đơn vị."""
    b = np.asarray(boxes, dtype=np.float64)
    x1, y1, x2, y2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], axis=-1)


def cxcywh_to_xyxy(boxes):
    """[..., 4] (cx,cy,w,h) -> (x1,y1,x2,y2). Giữ nguyên đơn vị."""
    b = np.asarray(boxes, dtype=np.float64)
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=-1)


# --------------------------------------------------------------------------
# Bước (2)+(3): pixel ảnh gốc -> hệ chuẩn [0,1]
# --------------------------------------------------------------------------

def compute_scale(W, H, target=512):
    """Hệ số resize giữ tỉ lệ. Mọi ảnh CE-130 cao đúng 384px nên luôn W >= H,
    tức new_w == target và pad LUÔN ở dưới (xem data/README.md §8)."""
    return min(target / float(W), target / float(H))


def scale_to_canvas(boxes_xyxy_px, W, H, target=512):
    """xyxy pixel trên ảnh gốc (W,H) -> cxcywh [0,1] trên canvas target x target.

    Trả về (boxes_cxcywh_norm, scale, valid_h) với `valid_h = new_h / target`
    là ranh giới vùng ảnh thật (phần dưới là pad).
    """
    s = compute_scale(W, H, target)
    b = np.asarray(boxes_xyxy_px, dtype=np.float64).reshape(-1, 4) * s
    n = xyxy_to_cxcywh(b) / float(target)
    valid_h = int(H * s) / float(target)
    return n, s, valid_h


def canvas_to_pixel(boxes_cxcywh_norm, W, H, target=512):
    """Nghịch đảo `scale_to_canvas`: cxcywh [0,1] -> xyxy pixel trên ảnh gốc."""
    s = compute_scale(W, H, target)
    b = np.asarray(boxes_cxcywh_norm, dtype=np.float64).reshape(-1, 4) * float(target)
    return cxcywh_to_xyxy(b) / s


# --------------------------------------------------------------------------
# Bước (4) và (5): hệ chuẩn <-> không gian diffusion
# --------------------------------------------------------------------------

def encode_diffusion(boxes_norm, snr_scale=2.0):
    """cxcywh [0,1] -> x_start cho diffusion: (x*2 - 1) * snr_scale.

    Áp cho CẢ 4 chiều — giống hệt DiffusionDet (`detector.py:396`). Vật nhỏ cho
    giá trị ÂM ở w/h (w=0,0686 -> -1,73); điều đó BÌNH THƯỜNG, không phải lỗi.
    """
    b = np.asarray(boxes_norm, dtype=np.float64)
    return (b * 2.0 - 1.0) * float(snr_scale)


def decode_diffusion(x, snr_scale=2.0):
    """x_start -> cxcywh [0,1]. Clamp TRƯỚC khi chia (giống DiffusionDet).

    LỖI VÒNG 1: chia thêm 2 lần nữa ở đây làm box còn nửa kích thước.
    Đúng là `(x/snr + 1) / 2`, không phải `(x/snr + 1) / 4`.
    """
    s = float(snr_scale)
    v = np.clip(np.asarray(x, dtype=np.float64), -s, s)
    return (v / s + 1.0) / 2.0


# --------------------------------------------------------------------------
# IoU / GIoU — nhận xyxy, dùng chung cho matcher lẫn loss lẫn test
# --------------------------------------------------------------------------

def _area_xyxy(b):
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def box_iou(boxes1, boxes2):
    """IoU từng cặp. boxes1 [N,4], boxes2 [M,4] xyxy -> (iou [N,M], union [N,M])."""
    b1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    a1, a2 = _area_xyxy(b1)[:, None], _area_xyxy(b2)[None, :]

    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    union = a1 + a2 - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """GIoU từng cặp, xyxy. Kết quả trong [-1, 1].

    GIoU có gradient cả khi hai box RỜI NHAU (khác IoU thuần = 0) — quan trọng
    với CE-130 vì vật rất nhỏ (median 0,41 % diện tích ảnh) nên phần lớn cặp
    prediction-GT lúc đầu train là rời nhau.

    Yêu cầu box hợp lệ (x2>=x1, y2>=y1). Box lộn ngược cho giá trị vô nghĩa —
    vòng 1 từng ra GIoU 5e5 vì giải mã sai. Dùng `filter_degenerate` trước.
    """
    b1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    assert (b1[:, 2:] >= b1[:, :2]).all(), "boxes1 có box lộn ngược (x2<x1 hoặc y2<y1)"
    assert (b2[:, 2:] >= b2[:, :2]).all(), "boxes2 có box lộn ngược (x2<x1 hoặc y2<y1)"

    iou, union = box_iou(b1, b2)

    lt = np.minimum(b1[:, None, :2], b2[None, :, :2])
    rb = np.maximum(b1[:, None, 2:], b2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    enclosing = wh[..., 0] * wh[..., 1]

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(enclosing > 0, iou - (enclosing - union) / enclosing, iou)


def pairwise_l1(boxes1, boxes2):
    """Khoảng cách L1 từng cặp trên 4 toạ độ. [N,4] x [M,4] -> [N,M]."""
    b1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 4)
    b2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 4)
    return np.abs(b1[:, None, :] - b2[None, :, :]).sum(-1)


# --------------------------------------------------------------------------
# Augmentation + vệ sinh dữ liệu
# --------------------------------------------------------------------------

def flip_horizontal(boxes_cxcywh_norm):
    """Lật ngang trong hệ chuẩn [0,1]: cx -> 1 - cx. w/h/cy giữ nguyên.

    Chỉ flip NGANG. Không rotate: rotate 5-30 độ làm box axis-aligned phình
    x1,20-x1,98, quá lớn với vật CE-130 (median 0,069 x 0,061); rotate 90 độ thì
    phá giả định "luôn pad ở dưới" (chỉ 9,6 % ảnh vuông).
    """
    b = np.asarray(boxes_cxcywh_norm, dtype=np.float64).copy()
    b[..., 0] = 1.0 - b[..., 0]
    return b


def filter_degenerate(boxes_xyxy, min_size=0.0):
    """Bỏ box có w hoặc h <= min_size. Trả về (boxes_sạch, mask_giữ).

    Đo được 14/37.110 box train của CE-130 bị degenerate. Phải lọc TRƯỚC khi
    vào matcher VÀ trước khi tính GIoU.
    """
    b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
    keep = (b[:, 2] - b[:, 0] > min_size) & (b[:, 3] - b[:, 1] > min_size)
    return b[keep], keep
