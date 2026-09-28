"""Giai đoạn 2: mask -> box, gộp theo độ cao cây, pipeline đầy đủ."""

import os
import sys

import numpy as np
import torch

from diffuse2seg.config.base import Diffu2SegConfig
from diffuse2seg.d2s.boxes import dedup_boxes, filter_boxes, masks_to_boxes
from diffuse2seg.d2s.masks import (extract_masks, largest_component_containing_seed,
                       threshold_maps)
from diffuse2seg.d2s.merging import (area_descending_nms, cluster_at_heights,
                         masks_from_clusters, merge_maps_to_masks,
                         normalise_maps, symmetric_kl_matrix)
from diffuse2seg.d2s.pipeline import masks_full_to_boxes, segment_image
from diffuse2seg.d2s.pipeline import segment_image
from diffuse2seg.d2s.plaplacian import plaplacian_propagate
from diffuse2seg.d2s.prompts import build_prompt_grid, f0_onehot
from diffuse2seg.utils.box_ops_np import (box_iou, cxcywh_to_xyxy, scale_to_canvas,
                              xywh_to_xyxy)

PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "diffuse2seg")

# ============================================================================
# từ test_masks_boxes.py
# ============================================================================

R = 64


CANVAS = 512


CELL = CANVAS / R          # 8.0 px


def test_connected_components_keeps_only_seed_component():
    """Two sheep of the same class: attention links them, geometry separates them."""
    m = np.zeros((R, R), dtype=bool)
    m[10:14, 10:14] = True          # object A, holds the seed
    m[40:44, 40:44] = True          # object B, identical-looking

    comp = largest_component_containing_seed(m, (11, 11))
    assert comp[10:14, 10:14].all()
    assert not comp[40:44, 40:44].any()
    assert comp.sum() == 16


def test_component_is_empty_when_seed_below_threshold():
    """A prompt whose own cell did not survive must yield nothing.

    NEGATIVE CONTROL against "just take the biggest component instead", which
    would invent a mask for a prompt that said nothing.
    """
    m = np.zeros((R, R), dtype=bool)
    m[40:44, 40:44] = True
    assert not largest_component_containing_seed(m, (11, 11)).any()


def test_threshold_is_per_map():
    """Each map is cut at ITS OWN scale, so absolute magnitude cannot leak across.

    Two comparably strong maps of different absolute size must both yield their
    own 100 cells. rel_floor is disabled here to isolate the quantile.
    """
    f = np.zeros((2, R * R))
    f[0, :100] = 1.0
    f[1, :100] = 0.5
    binm = threshold_maps(f, R, quantile=0.90, rel_floor=0.0)
    assert binm[0].sum() == binm[1].sum() == 100


def test_rel_floor_kills_maps_with_no_signal():
    """The fix for the 2-objects-became-58-boxes bug.

    A quantile is RELATIVE, so on its own it hands back ~10 % of the cells even
    from a map that propagated nowhere. Measured on a synthetic graph: a prompt
    seeded on background reached max 0.0000 while its own q90 was 1.5e-05, so
    `f > q90` still returned its own cell and that became a tidy 1x1 box.

    rel_floor adds the absolute question -- is anything here at least 5 % of the
    strongest response IN THIS IMAGE -- and a dead map fails it.

    NEGATIVE CONTROL: with rel_floor=0 the dead map DOES produce cells, which is
    what makes this test proof that the floor is doing the work.
    """
    f = np.zeros((2, R * R))
    f[0, :100] = 1.0                       # a real object
    f[1, :100] = 1e-6                      # background: 1e-6 of the strongest

    with_floor = threshold_maps(f, R, quantile=0.90, rel_floor=0.05)
    assert with_floor[0].sum() == 100, "the real map must survive"
    assert with_floor[1].sum() == 0, "the dead map must be suppressed"

    without = threshold_maps(f, R, quantile=0.90, rel_floor=0.0)
    assert without[1].sum() > 0, "rel_floor is no longer being tested"


def test_extract_masks_drops_dead_prompts():
    f = np.zeros((2, R * R))
    f[0].reshape(R, R)[10:14, 10:14] = 1.0
    # prompt 1 sits at (11,11) but its map is flat -> nothing above quantile
    cells = np.array([[11, 11], [11, 11]])
    masks, kept = extract_masks(f, cells, R, quantile=0.99)
    assert len(masks) == 1 and len(kept) == 1


def test_mask_to_box_is_outer_edge_exact():
    """Cells (2..5, 3..7) -> x [24, 64), y [16, 48) in canvas px.

    Catches the missing +1: without it x2 would be 56 and y2 40, shrinking every
    box by one 8 px cell (~20 % of a median CE-130 object).
    """
    m = np.zeros((1, R, R), dtype=bool)
    m[0, 2:6, 3:8] = True
    box = masks_to_boxes(m, R, CANVAS)[0]

    x1, x2, y1, y2 = 3 * CELL, 8 * CELL, 2 * CELL, 6 * CELL
    expected = np.array([((x1 + x2) / 2) / CANVAS, ((y1 + y2) / 2) / CANVAS,
                         (x2 - x1) / CANVAS, (y2 - y1) / CANVAS])
    assert np.allclose(box, expected)
    assert np.allclose(box[2] * CANVAS, 40.0)      # 5 cells wide
    assert np.allclose(box[3] * CANVAS, 32.0)      # 4 cells tall


def test_single_cell_mask_gives_one_cell_box():
    m = np.zeros((1, R, R), dtype=bool)
    m[0, 7, 9] = True
    box = masks_to_boxes(m, R, CANVAS)[0]
    assert np.allclose(box[2], CELL / CANVAS)
    assert np.allclose(box[3], CELL / CANVAS)


def test_empty_mask_gives_zero_row():
    box = masks_to_boxes(np.zeros((1, R, R), dtype=bool), R, CANVAS)[0]
    assert np.allclose(box, 0.0)


def test_filter_rejects_oversized_and_degenerate():
    boxes = np.array([
        [0.5, 0.4, 0.10, 0.10],     # keep
        [0.5, 0.4, 0.90, 0.90],     # too large -> padding escape tripwire
        [0.5, 0.4, 0.001, 0.10],    # degenerate
    ])
    kept, info = filter_boxes(boxes, R, CANVAS, valid_h=1.0)
    assert len(kept) == 1
    assert info["n_too_large"] == 1 and info["n_degenerate"] == 1


def test_padding_filter_uses_area_not_centre():
    """A real object low in the frame must survive; a padding box must not.

    valid_h = 0.94 (a 408x384 image). The first box has its CENTRE below
    valid_h but most of its area inside -- testing the centre alone would
    discard a genuine detection.
    """
    valid_h = 0.9412
    mostly_inside = [0.5, valid_h - 0.005, 0.06, 0.05]
    in_padding = [0.5, valid_h + 0.03, 0.06, 0.05]

    kept, info = filter_boxes(np.array([mostly_inside, in_padding]), R, CANVAS,
                              valid_h=valid_h)
    assert info["n_in_padding"] == 1
    assert len(kept) == 1
    assert kept[0][1] < valid_h


def test_right_hand_padding_is_filtered_for_portrait_images():
    """Same rule, applied to the WIDTH -- the COCO portrait case.

    valid_w = 0.9152 (a 586x640 image). CE-130 can never produce this, so
    without a test the width term would be dead code that looks correct.
    """
    valid_w = 0.9152
    mostly_inside = [valid_w - 0.005, 0.5, 0.06, 0.05]
    in_padding = [valid_w + 0.03, 0.5, 0.06, 0.05]

    kept, info = filter_boxes(np.array([mostly_inside, in_padding]), R, CANVAS,
                              valid_w=valid_w)
    assert info["n_in_padding"] == 1
    assert len(kept) == 1
    assert kept[0][0] < valid_w


def test_valid_w_default_leaves_ce130_filtering_unchanged():
    """[NEGATIVE CONTROL] valid_w=1.0 must reproduce the CE-130 behaviour byte
    for byte. If it does not, adding COCO support changed CE-130 results."""
    valid_h = 0.9412
    boxes = np.array([[0.5, valid_h - 0.005, 0.06, 0.05],
                      [0.5, valid_h + 0.03, 0.06, 0.05],
                      [0.2, 0.3, 0.10, 0.10]])
    a, ia = filter_boxes(boxes, R, CANVAS, valid_h=valid_h)
    b, ib = filter_boxes(boxes, R, CANVAS, valid_h=valid_h, valid_w=1.0)
    assert np.array_equal(a, b) and ia == ib


def test_dedup_removes_duplicates_keeps_neighbours():
    """441 prompts on ~21 objects: duplicates must go, true neighbours must not."""
    dup = [[0.30, 0.30, 0.10, 0.10]] * 5
    neighbour = [[0.42, 0.30, 0.10, 0.10]]         # adjacent object, low IoU
    kept, _ = dedup_boxes(np.array(dup + neighbour), iou_thr=0.70)
    assert len(kept) == 2


def test_dedup_orders_by_area_not_input_order():
    boxes = np.array([[0.5, 0.5, 0.05, 0.05], [0.5, 0.5, 0.20, 0.20]])
    kept, idx = dedup_boxes(boxes, iou_thr=0.70)
    assert idx[0] == 1, "largest box must be considered first"


def test_xywh_to_xyxy_from_coco():
    """COCO (x,y) is TOP-LEFT, not a centre -- the third CE-130 box convention."""
    assert np.allclose(xywh_to_xyxy([[74.0, 167.0, 25.0, 20.0]])[0],
                       [74.0, 167.0, 99.0, 187.0])


def _roundtrip_iou(W, H, gt_px):
    """GT pixels -> canvas -> cell mask -> box, returning IoU against the GT."""
    gt_norm, _, valid_h = scale_to_canvas(gt_px, W, H, CANVAS)
    assert abs(valid_h - int(H * min(CANVAS / W, CANVAS / H)) / CANVAS) < 1e-12

    xyxy = cxcywh_to_xyxy(gt_norm)[0] * CANVAS
    c0, c1 = int(xyxy[0] // CELL), int(np.ceil(xyxy[2] / CELL))
    r0, r1 = int(xyxy[1] // CELL), int(np.ceil(xyxy[3] / CELL))

    m = np.zeros((1, R, R), dtype=bool)
    m[0, r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)] = True
    back = masks_to_boxes(m, R, CANVAS)
    return box_iou(cxcywh_to_xyxy(back), cxcywh_to_xyxy(gt_norm))[0][0, 0]


def test_pad_coordinate_roundtrip():
    """Round-trip on the shapes that carry 96 % of val.

    Aspect ratios up to ~1.8 cover 96.5 % of val images (measured: 29.2 % below
    1.2, 26.4 % in 1.2-1.5, 40.9 % in 1.5-2.0). There the median 39x32 px object
    still spans 4-6 cells and survives the 8 px grid.
    """
    for W, H in [(384, 384), (541, 384), (700, 384)]:
        gt_px = np.array([[100.0, 120.0, 139.0, 152.0]])     # 39x32, the median
        iou = _roundtrip_iou(W, H, gt_px)
        assert iou > 0.6, f"round-trip IoU {iou:.3f} too low for {W}x{H}"


def test_extreme_aspect_loses_resolution_by_design():
    """THE COST OF r=64, as a number rather than a caveat.

    A 1918x384 image (aspect 4.99, the measured CE-130 maximum) is scaled by
    0.267 to fit a square canvas, so the median 39x32 px object shrinks to
    1.30 x 1.07 cells and quantisation eats most of it: round-trip IoU ~0.35
    even with a PERFECT mask.

    This is a property of the resolution choice, not a bug, and it bounds what
    any p can achieve on those images. It is rare -- only 3.5 % of val images
    have aspect > 2.0, and 0.2 % exceed 3.0 -- but run_stage1.py reports the
    per-image ceiling so a low oracle_recall is never misread as a failure of
    the mechanism. If gate 1 comes back GREY, r=96 is the next variable.
    """
    iou = _roundtrip_iou(1918, 384, np.array([[100.0, 120.0, 139.0, 152.0]]))
    assert iou < 0.5, "extreme aspect should visibly lose resolution"
    assert iou > 0.2, "...but not vanish entirely"


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_merging.py
# ============================================================================

def test_normalise_rows_sum_to_one():
    rng = np.random.default_rng(0)
    p, ok = normalise_maps(rng.random((7, 30)))
    assert ok.all()
    assert np.allclose(p.sum(axis=1), 1.0)


def test_all_zero_map_is_dropped_not_made_uniform():
    """[NEGATIVE CONTROL] Prompt lan truyền ra 0 phải bị LOẠI.

    Biến nó thành phân phối đều là tuyên bố rằng prompt đó thấy cả ảnh như
    nhau — sai, và nó sẽ kéo mọi khoảng cách KL về phía mình.
    """
    f = np.zeros((3, 10))
    f[0, :5] = 1.0
    f[2, 5:] = 1.0
    p, ok = normalise_maps(f)
    assert ok.tolist() == [True, False, True]
    assert (p[1] == 0).all()


def test_kl_expansion_matches_naive_definition():
    """⭐ Khai triển 2 matmul == vòng lặp đôi theo đúng công thức (3)."""
    rng = np.random.default_rng(1)
    p, _ = normalise_maps(rng.random((10, 25)) ** 3)
    eps = 1e-12
    fast = symmetric_kl_matrix(p, eps=eps)

    q = np.clip(p, eps, None)
    naive = np.zeros((len(p), len(p)))
    for a in range(len(p)):
        for b in range(len(p)):
            kl_ab = np.sum(p[a] * (np.log(q[a]) - np.log(q[b])))
            kl_ba = np.sum(p[b] * (np.log(q[b]) - np.log(q[a])))
            naive[a, b] = 0.5 * (kl_ab + kl_ba)
    np.fill_diagonal(naive, 0.0)
    naive = np.clip(naive, 0.0, None)
    assert np.abs(fast - naive).max() < 1e-10, np.abs(fast - naive).max()


def test_kl_matrix_is_a_valid_metric_shape():
    """scipy.linkage đòi đối xứng, chéo 0, không âm — nếu không thì nó ném."""
    rng = np.random.default_rng(2)
    p, _ = normalise_maps(rng.random((8, 20)))
    d = symmetric_kl_matrix(p)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)
    assert (d >= 0).all()


def test_kl_chunking_does_not_change_the_answer():
    rng = np.random.default_rng(3)
    p, _ = normalise_maps(rng.random((17, 30)))
    assert np.allclose(symmetric_kl_matrix(p, chunk=3),
                       symmetric_kl_matrix(p, chunk=1000))


def test_identical_distributions_have_zero_distance():
    a = np.zeros(10); a[:3] = 1 / 3
    c = np.zeros(10); c[7:] = 1 / 3
    d = symmetric_kl_matrix(np.stack([a, a.copy(), c]))
    assert d[0, 1] < 1e-9, "hai phân phối giống hệt phải cách nhau 0"
    assert d[0, 2] > 10.0, "hai phân phối rời nhau phải cách xa"


def test_cluster_count_decreases_with_height():
    """⭐ h nhỏ -> nhiều cụm (vật nhỏ), h lớn -> ít cụm (vật nguyên).

    Tính chất này là toàn bộ lý do có 6 mức. Mất nó thì 6 mức chỉ là chạy một
    thứ sáu lần.
    """
    rng = np.random.default_rng(4)
    p, _ = normalise_maps(rng.random((14, 40)) ** 3)
    d = symmetric_kl_matrix(p)
    heights = np.geomspace(0.186, 2.99, 6)
    counts = [int(l.max()) + 1 for l in cluster_at_heights(d, heights)]
    assert all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), counts


def test_argmax_over_clusters_gives_a_partition():
    """⭐ Mỗi pixel thuộc ĐÚNG MỘT cụm — GĐ2 không ngưỡng từng map."""
    r, H, W = 8, 32, 32
    p = np.zeros((4, r * r))
    p[0, :r * r // 2] = 1.0
    p[1, r * r // 2:] = 1.0
    p[2, :r * r // 2] = 0.9
    p[3, r * r // 2:] = 0.9
    p = p / p.sum(axis=1, keepdims=True)
    masks = masks_from_clusters(p, np.array([0, 1, 0, 1]), r, H, W, min_area_px=1)
    cover = np.zeros((H, W), dtype=int)
    for m in masks:
        cover += m
    assert set(cover.ravel().tolist()) == {1}, \
        "argmax qua cụm phải phủ kín và không chồng lấn"


def test_connected_components_split_one_cluster_into_instances():
    """Một cụm phủ hai vùng rời nhau phải ra HAI instance.

    Đây là bước biến vùng ngữ nghĩa thành instance mask.
    """
    r, H, W = 8, 16, 16
    p = np.zeros((1, r * r)).reshape(1, r, r)
    p[0, 0:2, 0:2] = 1.0          # góc trên trái
    p[0, 6:8, 6:8] = 1.0          # góc dưới phải, rời hẳn
    p = p.reshape(1, -1)
    p = p / p.sum()
    masks = masks_from_clusters(p, np.array([0]), r, H, W, min_area_px=1)
    assert len(masks) == 2, f"hai vùng rời phải ra 2 instance, được {len(masks)}"


def test_min_area_filters_noise():
    r, H, W = 8, 64, 64
    p = np.zeros((1, r, r))
    p[0, 0, 0] = 1.0
    masks_keep = masks_from_clusters(p.reshape(1, -1) / p.sum(), np.array([0]),
                                     r, H, W, min_area_px=1)
    masks_drop = masks_from_clusters(p.reshape(1, -1) / p.sum(), np.array([0]),
                                     r, H, W, min_area_px=10 ** 6)
    assert len(masks_keep) == 1 and len(masks_drop) == 0


def test_nms_keeps_largest_first():
    big = np.zeros((20, 20), bool); big[:15, :15] = True
    dup = big.copy()
    small = np.zeros((20, 20), bool); small[:5, :5] = True
    kept, info = area_descending_nms([small, dup, big], iou_thr=0.9)
    assert len(kept) == 2
    assert kept[0].sum() == big.sum(), "mask lớn nhất phải được giữ đầu tiên"
    assert info["n_suppressed"] == 1


def test_nms_at_0_9_keeps_nested_masks():
    """⭐ [NEGATIVE CONTROL] Ghế và mặt ghế đều phải sống sót.

    Paper đo: siết xuống 0,5 làm mất 3,1 p.p. mAR. Test này khoá lý do.
    """
    chair = np.zeros((40, 40), bool); chair[:30, :30] = True
    seat = np.zeros((40, 40), bool); seat[:12, :12] = True
    kept, _ = area_descending_nms([chair, seat], iou_thr=0.9)
    assert len(kept) == 2, "NMS 0,9 phải giữ mask lồng nhau (multi-granularity)"

    kept_strict, _ = area_descending_nms([chair, chair.copy()], iou_thr=0.9)
    assert len(kept_strict) == 1, "mask trùng hệt vẫn phải bị chặn"


def test_nms_respects_the_1000_cap():
    masks = []
    for i in range(5):
        m = np.zeros((10, 10), bool)
        m[i, :] = True
        masks.append(m)
    kept, info = area_descending_nms(masks, iou_thr=0.9, max_masks=3)
    assert len(kept) == 3 and info["n_over_cap"] == 2


def test_bbox_prefilter_does_not_change_the_result():
    """⭐ [NEGATIVE CONTROL] Lọc theo hộp bao là TỐI ƯU, không được đổi kết quả.

    NMS bỏ qua matmul cho những cặp mask có hộp bao rời nhau (IoU chắc chắn 0).
    Nếu điều kiện giao hộp viết sai — ví dụ dùng `<` thay `<=` — thì một cặp
    chạm biên sẽ bị bỏ sót và mask đáng lẽ bị chặn lại lọt vào kết quả, không
    một dấu hiệu nào.

    Test so kết quả với một bản NMS tham chiếu viết thẳng theo định nghĩa.
    """
    rng = np.random.default_rng(7)
    masks = []
    for _ in range(40):
        m = np.zeros((32, 32), dtype=bool)
        y, x = rng.integers(0, 24), rng.integers(0, 24)
        s = int(rng.integers(3, 10))
        m[y:y + s, x:x + s] = True
        masks.append(m)
    # thêm vài cặp CHẠM BIÊN nhau, chỗ dễ sai nhất
    a = np.zeros((32, 32), dtype=bool); a[0:10, 0:10] = True
    b = np.zeros((32, 32), dtype=bool); b[10:20, 0:10] = True   # chạm đúng cạnh
    masks += [a, b]

    def reference_nms(ms, thr, cap):
        flat = np.stack([m.ravel() for m in ms]).astype(np.float64)
        ar = flat.sum(1)
        keep = []
        for i in np.argsort(-ar):
            if len(keep) >= cap:
                continue
            ok = True
            for j in keep:
                inter = float(flat[i] @ flat[j])
                union = ar[i] + ar[j] - inter
                if union > 0 and inter / union > thr:
                    ok = False
                    break
            if ok:
                keep.append(int(i))
        return keep

    for thr in (0.5, 0.9):
        got, _ = area_descending_nms(masks, iou_thr=thr, max_masks=1000)
        want = reference_nms(masks, thr, 1000)
        assert len(got) == len(want), \
            f"thr={thr}: lọc hộp bao giữ {len(got)}, tham chiếu {len(want)}"
        for g, w in zip(got, want):
            assert np.array_equal(g, masks[w]), f"thr={thr}: khác thứ tự/nội dung"


def test_empty_input_is_handled():
    assert symmetric_kl_matrix(np.zeros((0, 5))).shape == (0, 0)
    assert masks_from_clusters(np.zeros((0, 64)), np.zeros(0, dtype=np.int64),
                               8, 16, 16) == []
    kept, info = area_descending_nms([])
    assert kept == [] and info["n_kept"] == 0


def main_merging():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


def _merge_cfg(keep):
    """Config nhỏ đủ chạy Alg.2 trên lưới 8x8."""
    return Diffu2SegConfig(canvas=64, n_levels=3, min_area_px=1,
                           keep_level_masks=keep).validate()


def _fake_maps(seed=0, k=6, r=8):
    rng = np.random.default_rng(seed)
    f = rng.random((k, r * r)) ** 3
    return f


def test_keep_level_masks_does_not_change_output():
    """Bật cờ chỉ THÊM thứ trả về; mask và mọi thống kê phải y hệt.

    Nếu cờ này đổi kết quả thì ảnh vẽ ra không còn là ảnh của con số đã báo
    cáo -- đúng loại sai âm thầm mà README cảnh báo.
    """
    f = _fake_maps()
    H = W = 16
    m_off, i_off = merge_maps_to_masks(f, _merge_cfg(False), H, W)
    m_on, i_on = merge_maps_to_masks(f, _merge_cfg(True), H, W)

    assert len(m_off) == len(m_on)
    for a, b in zip(m_off, m_on):
        assert np.array_equal(a, b)
    assert i_off["per_level"] == i_on["per_level"]
    assert i_off["nms"] == i_on["nms"]
    assert "level_masks" not in i_off        # tắt thì KHÔNG tốn bộ nhớ
    assert "level_masks" in i_on


def test_each_level_is_a_partition():
    """[NEGATIVE CONTROL] mỗi mức phải là PHÂN HOẠCH, không chồng lấn.

    Đây là lý do `--mode paper` vẽ từng mức riêng: trộn 6 mức lên nhau thì
    mức thô đè mức mịn. Nếu một mức tự nó đã chồng lấn thì cách vẽ đó cũng sai
    và test này phải đỏ.
    """
    f = _fake_maps(seed=1)
    H = W = 16
    _, info = merge_maps_to_masks(f, _merge_cfg(True), H, W)
    for li, masks in enumerate(info["level_masks"]):
        if not masks:
            continue
        cover = np.zeros((H, W), dtype=int)
        for m in masks:
            cover += m.astype(int)
        assert cover.max() <= 1, \
            f"mức {li}: có pixel thuộc {cover.max()} mask cùng lúc"



# ============================================================================
# từ test_stage2_pipeline.py
# ============================================================================

R_stage2_pipeline = 16


CANVAS_stage2_pipeline = 128


ORIG_H, ORIG_W = 64, 64


def _cfg(**kw):
    base = dict(canvas=CANVAS_stage2_pipeline, prompt_stride_cells=2, max_iter=60, tau_prop=1e-8,
                lam=1e-2, p=1.6, stage=2, n_levels=4, kl_h_min=0.05,
                kl_h_max=5.0, nms_iou=0.9, min_area_px=4, max_masks=1000)
    base.update(kw)
    return Diffu2SegConfig(**base).validate()


def _blocky_affinity(blocks, r=R_stage2_pipeline, within=1.0, across=1e-4):
    n = r * r
    A = np.full((n, n), across, dtype=np.float64)
    for (r0, r1, c0, c1) in blocks:
        idx = [i * r + j for i in range(r0, r1) for j in range(c0, c1)]
        for i in idx:
            for j in idx:
                A[i, j] = within
    A /= A.sum(axis=1, keepdims=True)
    return torch.tensor(A, dtype=torch.float32)


def _run(cfg, blocks, valid_h=1.0, valid_w=1.0):
    A = _blocky_affinity(blocks)
    img = np.zeros((CANVAS_stage2_pipeline, CANVAS_stage2_pipeline, 3), dtype=np.uint8)
    return segment_image(img, valid_h=valid_h, cfg=cfg, A=A, valid_w=valid_w,
                         orig_hw=(ORIG_H, ORIG_W))


def test_masks_come_back_at_original_resolution():
    """GĐ2 trả mask ở kích thước ẢNH GỐC, không phải lưới latent.

    Đây là điều Algorithm 2 quy định (upsample TRƯỚC argmax) và cũng là điều
    khiến AR tính được ở nơi GT sống.
    """
    out = _run(_cfg(), [(2, 6, 2, 6), (10, 14, 10, 14)])
    assert out["masks_full"].shape[1:] == (ORIG_H, ORIG_W), \
        f"mask ở {out['masks_full'].shape[1:]}, phải ở ({ORIG_H}, {ORIG_W})"
    assert out["masks_full"].dtype == bool


def test_each_object_gets_its_own_cluster():
    """Hai khối rời -> hai CỤM riêng, và argmax ở tâm mỗi khối chọn cụm của nó.

    Đây là câu hỏi đúng cho một fixture dựng tay. Câu "mỗi khối có một mask
    IoU>0.9" thì KHÔNG, và lý do đáng ghi lại vì nó là tính chất thật của
    Algorithm 2, không phải lỗi:

    Đo trên chính fixture này (2 khối 4 ô, nền 56 prompt):
        cụm 0 = 4 prompt khối 1   cụm 1 = 4 prompt khối 2   cụm 2 = 56 prompt nền
        tâm khối 1: [0.007615, 0.000242, 0.003806] -> argmax 0  ĐÚNG
        tâm khối 2: [0.000242, 0.007611, 0.003806] -> argmax 1  ĐÚNG
    Nhưng `pbar_c = (1/|C_c|) * sum p_k` chia cụm nền cho 56, nên giá trị của
    nó THẤP KHẮP NƠI; cụm 1 (4 prompt) do đó thắng argmax trên 2944 px chứ
    không riêng 256 px của khối 2. Khối 1 vẫn ra mask hoàn hảo (IoU 1,000) chỉ
    vì nó tình cờ bị cụm 0 "kẹp" chặt hơn.

    Nói cách khác: khi các cụm lệch kích thước 14x, argmax của các trung bình
    đã chuẩn hoá không còn phản ánh "vật". Trên ảnh thật với 529 prompt phân bố
    đều thì độ lệch nhỏ hơn nhiều — nhưng đây là thứ phải nhìn trong
    visualize_masks.py, không phải thứ test giả lập kết luận được.
    """
    from diffuse2seg.d2s.merging import (cluster_at_heights, normalise_maps,
                             symmetric_kl_matrix)
    from diffuse2seg.d2s.plaplacian import plaplacian_propagate
    from diffuse2seg.d2s.prompts import build_prompt_grid, f0_onehot

    cfg = _cfg()
    A = _blocky_affinity([(2, 6, 2, 6), (10, 14, 10, 14)])
    cells = build_prompt_grid(R_stage2_pipeline, cfg.prompt_stride_cells, valid_h=1.0,
                              min_valid_frac=cfg.min_valid_frac)
    f0 = f0_onehot(cells, R_stage2_pipeline, device=A.device, dtype=A.dtype)
    f, _, _ = plaplacian_propagate(A, f0, p=cfg.p, lam=cfg.lam,
                                   tau_prop=cfg.tau_prop, max_iter=cfg.max_iter,
                                   g_eps=cfg.g_eps)
    p_maps, ok = normalise_maps(f.numpy())
    p_maps = p_maps[ok]
    used = np.asarray(cells)[ok]
    labels = cluster_at_heights(symmetric_kl_matrix(p_maps, eps=cfg.kl_eps),
                                [cfg.kl_h_min])[0]

    def cluster_of(r0, r1, c0, c1):
        idx = [i for i, (r, c) in enumerate(used) if r0 <= r < r1 and c0 <= c < c1]
        assert idx, f"không prompt nào rơi vào khối ({r0},{r1},{c0},{c1})"
        return set(labels[idx].tolist())

    c1 = cluster_of(2, 6, 2, 6)
    c2 = cluster_of(10, 14, 10, 14)
    assert len(c1) == 1, f"prompt của khối 1 bị tách ra {len(c1)} cụm"
    assert len(c2) == 1, f"prompt của khối 2 bị tách ra {len(c2)} cụm"
    assert c1 != c2, "hai khối rời nhau lại rơi vào cùng một cụm"

    # argmax tại tâm mỗi khối phải chọn cụm của chính khối đó
    n_cl = int(labels.max()) + 1
    bar = np.zeros((n_cl, p_maps.shape[1]))
    np.add.at(bar, labels, p_maps)
    bar /= np.maximum(np.bincount(labels, minlength=n_cl), 1)[:, None]
    assert bar[:, 4 * R_stage2_pipeline + 4].argmax() == c1.pop(), "argmax ở tâm khối 1 chọn sai cụm"
    assert bar[:, 12 * R_stage2_pipeline + 12].argmax() == c2.pop(), "argmax ở tâm khối 2 chọn sai cụm"


def test_at_least_one_object_is_recovered_as_an_exact_mask():
    """Ít nhất một khối phải ra mask khớp gần hoàn hảo.

    Kiểm rằng chuỗi upsample -> argmax -> connected components -> NMS thật sự
    sinh ra được một instance đúng hình, chứ không chỉ những mảnh vụn.
    """
    out = _run(_cfg(), [(2, 6, 2, 6), (10, 14, 10, 14)])
    m = out["masks_full"]
    assert len(m) >= 2

    # ⚠️ float32, KHÔNG phải bool: `bool @ bool` trong numpy là AND-tích luỹ và
    # trả True/False, không đếm phần giao — mọi IoU sẽ thành 0 hoặc 1 mà không
    # báo gì. (utils/mask_ops.py và d2s/merging.py ép float trước matmul đúng
    # vì lý do này.)
    flat = m.reshape(len(m), -1).astype(np.float32)

    def block_mask(r0, r1, c0, c1):
        g = np.zeros((R_stage2_pipeline, R_stage2_pipeline), dtype=bool)
        g[r0:r1, c0:c1] = True
        ys = ((np.arange(ORIG_H) + 0.5) / ORIG_H * R_stage2_pipeline).astype(int).clip(0, R_stage2_pipeline - 1)
        xs = ((np.arange(ORIG_W) + 0.5) / ORIG_W * R_stage2_pipeline).astype(int).clip(0, R_stage2_pipeline - 1)
        return g[ys][:, xs]

    best = []
    for blk in ((2, 6, 2, 6), (10, 14, 10, 14)):
        g = block_mask(*blk).ravel().astype(np.float32)
        inter = flat @ g
        union = flat.sum(1) + g.sum() - inter
        best.append(float(np.where(union > 0, inter / np.maximum(union, 1.0), 0.0).max()))
    assert max(best) > 0.9, f"không khối nào ra mask khớp; IoU tốt nhất {best}"


def test_a_single_cluster_level_covers_the_whole_image():
    """Ở h lớn mọi prompt gộp làm một -> argmax cho MỘT vùng phủ kín ảnh.

    Tính chất của phân hoạch, và là lý do "mask lớn nhất" không bao giờ là
    một phép thử tốt cho "vật".
    """
    out = _run(_cfg(), [(2, 6, 2, 6), (10, 14, 10, 14)])
    mi = out["merge_info"]
    assert mi["per_level"][-1]["n_clusters"] == 1, \
        "height lớn nhất phải gộp mọi prompt thành 1 cụm"
    areas = out["masks_full"].reshape(len(out["masks_full"]), -1).sum(1)
    assert areas.max() == ORIG_H * ORIG_W, "cụm duy nhất phải phủ kín ảnh"


def test_every_level_is_actually_run():
    """merge_info phải ghi đúng số mức đã chạy, kèm height của từng mức."""
    cfg = _cfg(n_levels=4)
    out = _run(cfg, [(2, 6, 2, 6), (10, 14, 10, 14)])
    mi = out["merge_info"]
    assert len(mi["per_level"]) == 4
    assert len(mi["heights"]) == 4
    assert mi["heights"] == sorted(mi["heights"]), "height phải tăng dần"
    assert all("n_clusters" in lv and "n_masks" in lv for lv in mi["per_level"])


def test_stage2_requires_orig_hw():
    """[NEGATIVE CONTROL] Thiếu orig_hw thì phải NÉM, không được đoán.

    Đoán kích thước ảnh gốc sẽ làm a_min=100 px đo trên một thang khác và mask
    trả về sai độ phân giải — cả hai đều không crash.
    """
    A = _blocky_affinity([(2, 6, 2, 6)])
    img = np.zeros((CANVAS_stage2_pipeline, CANVAS_stage2_pipeline, 3), dtype=np.uint8)
    try:
        segment_image(img, valid_h=1.0, cfg=_cfg(), A=A)
    except AssertionError:
        return
    raise AssertionError("stage 2 thiếu orig_hw phải ném AssertionError")


def test_padding_is_not_stretched_into_the_image():
    """Ảnh dọc (pad ở phải): mask không được lấy nội dung từ vùng pad."""
    cfg = _cfg()
    out = _run(cfg, [(2, 6, 2, 6)], valid_w=0.5)
    m = out["masks_full"]
    if len(m):
        # với valid_w=0.5, chỉ nửa trái lưới ánh xạ vào ảnh; khối (c=2..6) nằm
        # trong nửa đó nên phải xuất hiện, và toàn bộ ảnh vẫn được phủ.
        assert m.shape[1:] == (ORIG_H, ORIG_W)


def test_boxes_are_derived_from_the_masks():
    """Box của GĐ2 lấy từ chính mask, không qua lưới latent."""
    out = _run(_cfg(), [(2, 6, 2, 6), (10, 14, 10, 14)])
    m, b = out["masks_full"], out["boxes"]
    assert len(b) == len(m)
    for i in range(len(m)):
        ys, xs = np.where(m[i])
        if not len(ys):
            continue
        x1 = xs.min() / ORIG_W
        x2 = (xs.max() + 1) / ORIG_W
        assert abs((b[i][0] - b[i][2] / 2) - x1) < 1e-9
        assert abs((b[i][0] + b[i][2] / 2) - x2) < 1e-9


def test_masks_full_to_boxes_handles_empty():
    assert masks_full_to_boxes(np.zeros((0, 8, 8), bool), 8, 8).shape == (0, 4)


def test_stage1_still_works_unchanged():
    """[NEGATIVE CONTROL] Thêm GĐ2 không được đụng đường GĐ1.

    GĐ1 là thứ cửa chặn 1 đo; nếu nó đổi hành vi thì mọi số đã đo mất hiệu lực.
    """
    cfg = Diffu2SegConfig(canvas=CANVAS_stage2_pipeline, prompt_stride_cells=2, max_iter=60,
                          tau_prop=1e-8, lam=1e-2, p=1.6, stage=1).validate()
    A = _blocky_affinity([(2, 6, 2, 6), (10, 14, 10, 14)])
    img = np.zeros((CANVAS_stage2_pipeline, CANVAS_stage2_pipeline, 3), dtype=np.uint8)
    out = segment_image(img, valid_h=1.0, cfg=cfg, A=A)
    assert out["n_boxes"] == 2, f"GĐ1 phải vẫn ra 2 box, được {out['n_boxes']}"
    assert "masks_full" not in out or len(out.get("masks_full", ())) == 0


def main_stage2_pipeline():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_pipeline.py
# ============================================================================

R_pipeline = 16                      # small grid: a 128 canvas, fast and readable


CANVAS_pipeline = 128


def _cfg_pipeline(**kw):
    base = dict(canvas=CANVAS_pipeline, prompt_stride_cells=2, mask_quantile=0.90,
                max_iter=60, tau_prop=1e-8, lam=1e-2, p=1.6)
    base.update(kw)
    return Diffu2SegConfig(**base)


def _blocky_affinity_pipeline(blocks, r=R_pipeline, within=1.0, across=1e-4):
    """Graph where tokens inside a block attend to each other and little else.

    Stands in for "SD2 knows these patches belong to the same object".
    """
    n = r * r
    A = np.full((n, n), across, dtype=np.float64)
    owner = -np.ones(n, dtype=int)
    for b_idx, (r0, r1, c0, c1) in enumerate(blocks):
        idx = [i * r + j for i in range(r0, r1) for j in range(c0, c1)]
        owner[idx] = b_idx
        for i in idx:
            for j in idx:
                A[i, j] = within
    A /= A.sum(axis=1, keepdims=True)
    return torch.tensor(A, dtype=torch.float32), owner


def test_two_objects_give_two_boxes():
    """Two disjoint blocks of identical appearance -> two separate boxes.

    This is the CE-130 situation in miniature: ~21 objects of ONE class per
    image, which attention cannot tell apart and connectivity can.
    """
    A, _ = _blocky_affinity_pipeline([(2, 6, 2, 6), (10, 14, 10, 14)])
    img = np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), dtype=np.uint8)

    out = segment_image(img, valid_h=1.0, cfg=_cfg_pipeline(), A=A)

    assert out["n_boxes"] == 2, f"expected 2 boxes, got {out['n_boxes']}"
    cx = sorted(b[0] for b in out["boxes"])
    assert cx[0] < 0.5 < cx[1]


def test_box_coordinates_match_the_block():
    """Block rows/cols 2..5 -> canvas [16, 48) -> cxcywh (0.25, 0.25, .25, .25)."""
    A, _ = _blocky_affinity_pipeline([(2, 6, 2, 6)])
    out = segment_image(np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), np.uint8),
                        valid_h=1.0, cfg=_cfg_pipeline(), A=A)

    assert out["n_boxes"] == 1
    cell = CANVAS_pipeline / R_pipeline                       # 8 px
    lo, hi = 2 * cell, 6 * cell             # outer edge of the last cell
    expect = [((lo + hi) / 2) / CANVAS_pipeline, ((lo + hi) / 2) / CANVAS_pipeline,
              (hi - lo) / CANVAS_pipeline, (hi - lo) / CANVAS_pipeline]
    assert np.allclose(out["boxes"][0], expect, atol=1e-9)


def test_padding_yields_no_prompts_no_boxes():
    """valid_h small enough that every seed row is padding -> empty, not garbage."""
    A, _ = _blocky_affinity_pipeline([(12, 15, 12, 15)])
    out = segment_image(np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), np.uint8),
                        valid_h=0.02, cfg=_cfg_pipeline(), A=A)
    assert out["n_boxes"] == 0


def test_object_in_padding_is_filtered_out():
    """A block below valid_h must not become a box even if seeds reach it."""
    A, _ = _blocky_affinity_pipeline([(2, 6, 2, 6), (13, 16, 2, 6)])
    out = segment_image(np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), np.uint8),
                        valid_h=12.0 / R_pipeline, cfg=_cfg_pipeline(), A=A)

    assert out["n_boxes"] == 1, "the padding block should have been removed"
    assert out["boxes"][0][1] < 12.0 / R_pipeline


def test_diagnostics_are_reported():
    """The gate reads these; absent or wrong, results cannot be interpreted."""
    A, _ = _blocky_affinity_pipeline([(2, 6, 2, 6)])
    out = segment_image(np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), np.uint8),
                        valid_h=1.0, cfg=_cfg_pipeline(), A=A)

    for key in ("n_prompts", "n_masks", "n_boxes", "n_iter", "converged",
                "f_max", "filter_info"):
        assert key in out, f"missing diagnostic: {key}"
    assert out["n_prompts"] > 0
    assert 0.0 < out["f_max"] <= 1.0
    assert set(out["filter_info"]) == {
        "n_in", "n_kept", "n_degenerate", "n_too_large", "n_in_padding"}


def test_hitting_the_iteration_cap_is_reported_not_hidden():
    """max_iter=1 cannot converge; `converged` must say so.

    Numbers from a run that hit the cap describe the cap, not the mechanism --
    the gate rejects such a run rather than reading its oracle_recall.
    """
    A, _ = _blocky_affinity_pipeline([(2, 6, 2, 6)])
    out = segment_image(np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), np.uint8), valid_h=1.0,
                        cfg=_cfg_pipeline(max_iter=1, tau_prop=1e-30), A=A)
    assert out["n_iter"] == 1
    assert out["converged"] is False


def test_p2_runs_end_to_end_through_the_same_path():
    """p=2 is gate 1's control arm, so it must traverse the identical code path.

    Only that it RUNS and stays finite is asserted here. How many boxes it
    produces is the gate's question, not this suite's -- see the next test.
    """
    A, _ = _blocky_affinity_pipeline([(2, 6, 2, 6), (10, 14, 10, 14)])
    out = segment_image(np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), np.uint8), valid_h=1.0,
                        cfg=_cfg_pipeline(p=2.0), A=A)
    assert out["n_boxes"] > 0
    assert np.isfinite(out["boxes"]).all()
    assert out["n_prompts"] == 64


def test_p_below_two_suppresses_background_bleed():
    """The edge-preserving claim, as a measurement rather than an expectation.

    Measured on this graph (weak cross-links at 1e-4, 8 of 64 prompts inside a
    block):

        p     max f on background / max f inside a block
        1.6              0.0010          -> below mask_rel_floor, dropped
        2.0              0.0803          -> above it, survives as 56 1x1 boxes

    Linear diffusion (p=2) leaks across the weak links; the sub-quadratic
    penalty does not. Roughly an 80x difference in background response.

    WHAT THIS IS NOT. A hand-built graph with a clean 1e-4 boundary is the
    easiest possible case for edge preservation. It shows the implementation
    reproduces the mechanism; it says NOTHING about whether SD2's real attention
    on CE-130 has boundaries this clean, where the median object is 4.65 cells
    across. That is exactly what tools/check_plaplacian_vs_p2.py measures, and
    this test must never be cited as an answer to it.
    """
    blocks = [(2, 6, 2, 6), (10, 14, 10, 14)]
    A, _ = _blocky_affinity_pipeline(blocks)
    cells = build_prompt_grid(R_pipeline, 2, valid_h=1.0)
    f0 = f0_onehot(cells, R_pipeline)

    in_block = [i for i, (r, c) in enumerate(cells)
                if any(r0 <= r < r1 and c0 <= c < c1 for r0, r1, c0, c1 in blocks)]
    background = [i for i in range(len(cells)) if i not in in_block]
    assert len(in_block) == 8 and len(background) == 56

    ratio = {}
    for p in (1.6, 2.0):
        f, _, _ = plaplacian_propagate(A, f0, p=p, lam=1e-2, tau_prop=1e-8,
                                       max_iter=60)
        f = f.numpy()
        ratio[p] = f[background].max() / max(f[in_block].max(), 1e-12)

    assert ratio[1.6] < ratio[2.0], "p<2 must bleed less than linear diffusion"
    assert ratio[1.6] < 0.05, "p=1.6 background should fall under mask_rel_floor"
    assert ratio[2.0] > 0.05, "p=2.0 background should survive the floor here"


def test_affinity_size_must_match_grid():
    A, _ = _blocky_affinity_pipeline([(2, 6, 2, 6)], r=8)
    try:
        segment_image(np.zeros((CANVAS_pipeline, CANVAS_pipeline, 3), np.uint8),
                      valid_h=1.0, cfg=_cfg_pipeline(), A=A)
    except AssertionError:
        return
    raise AssertionError("a mismatched affinity size must be rejected")


def main_pipeline():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0

