"""Chỉ số: chất lượng box và AR theo ngưỡng IoU/cỡ vật."""

import os
import sys

import numpy as np

from diffuse2seg.utils.ar_metrics import (IOU_THRESHOLDS, SIZE_BANDS,
                              ar_one_image, summarise_ar, _greedy_match)
from diffuse2seg.utils.mask_ops import mask_iou_matrix, masks_to_original
from diffuse2seg.utils.metrics import quality_one_image, summarise

PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "diffuse2seg")

# ============================================================================
# từ test_metrics.py
# ============================================================================

def _box(cx, cy, w, h):
    return np.array([[cx, cy, w, h]], dtype=np.float64)


def test_empty_prediction_counts_n_gt_not_zero():
    """TRAP 2. An image with no predictions still contributes its GT count.

    NEGATIVE CONTROL: the whole point is that n_gt is 5 and NOT 0. Returning
    (0, 0) would silently delete five missed objects from the denominator.
    """
    gt = np.tile(_box(0.5, 0.5, 0.1, 0.1), (5, 1))
    best, hits, n_gt = quality_one_image(np.zeros((0, 4)), gt)

    assert n_gt == 5, "missed objects must stay in the denominator"
    assert n_gt != 0
    assert hits == 0
    assert best.shape == (5,)
    assert np.all(best == 0.0)


def test_no_gt_contributes_nothing():
    best, hits, n_gt = quality_one_image(_box(0.5, 0.5, 0.1, 0.1), np.zeros((0, 4)))
    assert (len(best), hits, n_gt) == (0, 0, 0)


def test_raw_accumulation_differs_from_mean_of_ratios():
    """TRAP 1. Proves the trap is real, not folklore.

    Image 1: 1 GT, hit.        Image 2: 100 GT, 1 hit.
    Raw:  (1 + 1) / (1 + 100) = 0.0198
    Mean of ratios: (1.0 + 0.01) / 2 = 0.505   <- 25x larger, and wrong
    """
    hits, n_gt, best_all = 0, 0, []

    b1, h1, g1 = 1.0, 1, 1
    best_all.append(np.array([b1]))
    hits += h1
    n_gt += g1

    b2 = np.zeros(100)
    b2[0] = 1.0
    best_all.append(b2)
    hits += 1
    n_gt += 100

    out = summarise(best_all, hits, n_gt)

    assert abs(out["oracle_recall"] - 2.0 / 101.0) < 1e-12
    mean_of_ratios = (1.0 / 1 + 1.0 / 100) / 2
    assert abs(out["oracle_recall"] - mean_of_ratios) > 0.4, "the trap must bite"


def test_oracle_recall_is_score_free():
    """Shuffling prediction order changes nothing: there is no ranking here."""
    gt = np.concatenate([_box(0.25, 0.25, 0.1, 0.1), _box(0.75, 0.75, 0.1, 0.1)])
    pred = np.concatenate(
        [_box(0.25, 0.25, 0.1, 0.1), _box(0.9, 0.1, 0.05, 0.05), _box(0.75, 0.75, 0.1, 0.1)]
    )

    _, hits_a, n_a = quality_one_image(pred, gt)
    _, hits_b, n_b = quality_one_image(pred[::-1], gt)
    assert (hits_a, n_a) == (hits_b, n_b) == (2, 2)


def test_perfect_prediction_gives_recall_one():
    gt = np.concatenate([_box(0.3, 0.3, 0.2, 0.2), _box(0.7, 0.7, 0.2, 0.2)])
    best, hits, n_gt = quality_one_image(gt.copy(), gt)
    assert hits == n_gt == 2
    assert np.allclose(best, 1.0)
    assert abs(summarise([best], hits, n_gt)["mean_bestIoU"] - 1.0) < 1e-12


def test_iou_threshold_is_half():
    """A box overlapping exactly 1/3 is not a hit; a near-exact one is."""
    gt = _box(0.5, 0.5, 0.2, 0.2)
    shifted = _box(0.6, 0.5, 0.2, 0.2)          # IoU = 1/3
    _, hits, _ = quality_one_image(shifted, gt)
    assert hits == 0

    _, hits2, _ = quality_one_image(_box(0.505, 0.5, 0.2, 0.2), gt)
    assert hits2 == 1


def test_no_score_auc_key():
    """Locks the module docstring's promise.

    score_AUC here would be a proxy, and a proxy under that name invites a
    comparison against A/B/C1's 0.4965-0.4988 that it cannot support.
    """
    out = summarise([np.array([1.0])], 1, 1)
    assert "score_AUC" not in out
    assert "score_AUC_n_images" not in out
    assert set(out) == {
        "oracle_recall", "mean_bestIoU", "median_bestIoU",
        "n_gt", "n_hit", "n_pred_total", "n_images",
    }


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_ar_metrics.py
# ============================================================================

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


def main_ar_metrics():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0

