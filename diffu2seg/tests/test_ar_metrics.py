#!/usr/bin/env python3
"""AR_1000 + nâng mask lên độ phân giải ảnh gốc.

Chạy trên CPU, không cần GPU/SD/dữ liệu. Suite này khoá đúng hai chỗ sai ÂM
THẦM của đường PACO — sai mà vẫn ra một con số trông hợp lý:

  1. HOÁN VỊ TRỤC khi nâng (M,r,r) -> (M,H,W). Mask lệch 90° vẫn có IoU hợp lệ
     với GT, vẫn cho AR ra số, không assert nào bắt.
  2. GHÉP NHIỀU-MỘT thay vì MỘT-MỘT. Trên PACO một cái ghế và 8 bộ phận của nó
     chồng lấn nặng; nếu một mask hình-cái-ghế được tính là tìm thấy cả 9 GT thì
     AR phồng lên gấp nhiều lần mà không có dấu hiệu gì.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.ar_metrics import (IOU_THRESHOLDS, SIZE_BANDS,  # noqa: E402
                              ar_one_image, summarise_ar, _greedy_match)
from utils.mask_ops import mask_iou_matrix, masks_to_original  # noqa: E402

R, W, H = 140, 640, 480
VW, VH = 1.0, 480.0 / 640.0          # ảnh ngang 640x480 trên canvas vuông


def test_thresholds_are_the_coco_set():
    """0,50 -> 0,95 bước 0,05, đúng 10 ngưỡng."""
    assert len(IOU_THRESHOLDS) == 10
    assert abs(IOU_THRESHOLDS[0] - 0.50) < 1e-9
    assert abs(IOU_THRESHOLDS[-1] - 0.95) < 1e-9


def test_full_grid_covers_the_whole_real_image():
    o = masks_to_original(np.ones((1, R, R), bool), VW, VH, W, H)
    assert o.shape == (1, H, W)
    assert o.all(), "mask phủ kín lưới phải phủ kín vùng ảnh thật"


def test_upsampling_preserves_orientation():
    """[NEGATIVE CONTROL] Ô (row=10, col=100) phải rơi vào TRÊN-PHẢI.

    Đây là test bắt hoán vị trục. Nếu masks_to_original đảo x/y thì ô này rơi
    xuống DƯỚI-TRÁI, IoU vẫn tính được, AR vẫn ra số — chỉ là số vô nghĩa.
    """
    m = np.zeros((1, R, R), bool)
    m[0, 10, 100] = True
    ys, xs = np.where(masks_to_original(m, VW, VH, W, H)[0])
    assert len(xs), "ô đơn biến mất khi nâng độ phân giải"
    assert xs.mean() > W * 0.6, "cột 100/140 phải nằm bên PHẢI — nghi hoán vị trục"
    assert ys.mean() < H * 0.3, "hàng 10/140 phải nằm TRÊN — nghi hoán vị trục"


def test_padding_region_is_dropped_not_stretched():
    """Ảnh dọc pad ở phải: lưới bên phải valid_w không được kéo vào ảnh."""
    vw, vh = 0.75, 1.0
    m = np.zeros((1, R, R), bool)
    m[0, :, int(0.9 * R):] = True          # hoàn toàn nằm trong vùng pad
    o = masks_to_original(m, vw, vh, 480, 640)
    assert not o.any(), "mask nằm trọn trong vùng pad phải biến mất, không bị kéo giãn"


def test_mask_iou_matches_hand_computation():
    p = np.zeros((1, 10, 10), bool); p[0, :, :5] = True
    g = np.zeros((1, 10, 10), bool); g[0, :, 2:7] = True
    # giao 3 cột, hợp 7 cột
    assert abs(mask_iou_matrix(p, g)[0, 0] - 3.0 / 7.0) < 1e-12


def test_mask_iou_identity_and_disjoint():
    a = np.zeros((2, 20, 20), bool)
    a[0, :10, :10] = True
    a[1, 10:, 10:] = True
    iou = mask_iou_matrix(a, a)
    assert np.allclose(np.diag(iou), 1.0)
    assert iou[0, 1] == 0.0 and iou[1, 0] == 0.0


def test_mask_iou_chunking_gives_same_answer():
    """Chia lô chỉ để tiết kiệm bộ nhớ, không được đổi kết quả."""
    rng = np.random.default_rng(0)
    p = rng.random((37, 12, 12)) > 0.5
    g = rng.random((11, 12, 12)) > 0.5
    assert np.allclose(mask_iou_matrix(p, g, chunk=4),
                       mask_iou_matrix(p, g, chunk=1000))


def test_matching_is_one_to_one():
    """[NEGATIVE CONTROL] Một prediction KHÔNG được nhận hai GT.

    Chỗ này quan trọng riêng với PACO: một cái ghế và các bộ phận của nó chồng
    lấn nặng, nên ghép nhiều-một sẽ thổi phồng AR một cách âm thầm.
    """
    best = _greedy_match(np.array([[0.9, 0.8]]))
    assert (best > 0).sum() == 1, "1 prediction đã được ghép cho 2 GT"
    assert best[0] == 0.9, "phải ghép cặp có IoU cao nhất trước"


def test_matching_prefers_globally_best_pair():
    """Greedy trên danh sách cặp đã sắp, không phải argmax theo từng GT."""
    iou = np.array([[0.60, 0.55],
                    [0.50, 0.95]])
    best = _greedy_match(iou)
    assert abs(best[1] - 0.95) < 1e-12
    assert abs(best[0] - 0.60) < 1e-12


def test_ar_is_always_below_recall_at_50():
    """AR là trung bình 10 ngưỡng, recall@0,5 là số hạng LỚN NHẤT.

    Test này tồn tại để chặn việc trích oracle_recall như thể nó là AR_1000:
    hai số đo hai thứ khác nhau và AR luôn nhỏ hơn.
    """
    iou = np.array([[0.60, 0.00], [0.00, 0.92]])
    hits, n_gt, _ = ar_one_image(iou, np.array([2000.0, 2000.0]))
    ar = (hits / n_gt).mean()
    assert ar < hits[0] / n_gt


def test_no_predictions_does_not_crash():
    """⭐ [ĐÃ TỪNG VỠ] Ảnh không sinh mask nào phải chạy trót lọt.

    `np.zeros((0,H,W)).reshape(0, -1)` ném "cannot reshape array of size 0 into
    shape (0,newaxis)". Đây KHÔNG phải trường hợp phòng xa: một ảnh mà mọi cụm
    đều dưới a_min = 100 px sẽ trả về đúng 0 mask, và khi đó cả job chết giữa
    chừng sau khi đã chạy hàng chục ảnh trên GPU.
    """
    gt = np.zeros((3, 48, 64), dtype=bool)
    gt[0, :10, :10] = True
    gt[1, 20:30, 20:30] = True
    gt[2, 40:, 40:] = True

    iou = mask_iou_matrix(np.zeros((0, 48, 64), dtype=bool), gt)
    assert iou.shape == (0, 3)
    hits, n_gt, _ = ar_one_image(iou, np.array([100.0, 200.0, 300.0]))
    assert n_gt == 3 and hits.sum() == 0, "GT vẫn phải vào mẫu số"

    # chiều ngược lại: có prediction nhưng không GT
    pred = np.zeros((5, 48, 64), dtype=bool)
    pred[0, :5, :5] = True
    assert mask_iou_matrix(pred, np.zeros((0, 48, 64), dtype=bool)).shape == (5, 0)

    # và nâng độ phân giải từ tập rỗng.
    # ⚠️ Chữ ký là (masks, valid_w, valid_h, W, H) — W TRƯỚC H. Phiên bản đầu
    # của test này truyền (48, 64) tưởng là (H, W) và fail trên code đúng.
    assert masks_to_original(np.zeros((0, 8, 8), dtype=bool),
                             1.0, 1.0, 64, 48).shape == (0, 48, 64)


def test_summarise_with_no_gt_returns_zero_not_nan():
    """n_gt = 0 không được sinh NaN — một NaN sẽ lan ra cả bản tổng kết."""
    z = np.zeros(len(IOU_THRESHOLDS), dtype=np.int64)
    res = summarise_ar(z, 0, {k: z.copy() for k in SIZE_BANDS},
                       {k: 0 for k in SIZE_BANDS})
    assert res["AR_1000"] == 0.0
    for k in SIZE_BANDS:
        assert res[f"AR_{k}"] == 0.0


def test_size_bands_follow_coco_definition():
    iou = np.eye(3)
    areas = np.array([500.0, 5000.0, 50000.0])     # S, M, L
    _, _, band = ar_one_image(iou, areas)
    assert band["S"][1] == 1 and band["M"][1] == 1 and band["L"][1] == 1


def test_proposal_cap_truncates_at_1000():
    """Quá 1000 prediction thì cắt; GT chỉ khớp được bởi phần còn lại."""
    P, G = 1500, 1
    iou = np.zeros((P, G))
    iou[1200, 0] = 0.99                  # nằm SAU ngưỡng cắt
    hits, n_gt, _ = ar_one_image(iou, np.array([2000.0]), max_proposals=1000)
    assert hits.sum() == 0, "prediction thứ 1200 không được tính khi cap = 1000"


def test_empty_prediction_still_counts_gt():
    """[NEGATIVE CONTROL] Không có prediction nào -> n_gt vẫn vào mẫu số."""
    hits, n_gt, _ = ar_one_image(np.zeros((0, 3)), np.array([1.0, 2.0, 3.0]))
    assert n_gt == 3 and hits.sum() == 0


def test_summarise_divides_once():
    """Cộng dồn thô rồi chia MỘT LẦN — không trung bình các tỉ lệ mỗi ảnh."""
    hits = np.array([2] * 10)
    res = summarise_ar(hits, 101)
    assert abs(res["AR_1000"] - 2.0 / 101.0) < 1e-12


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
