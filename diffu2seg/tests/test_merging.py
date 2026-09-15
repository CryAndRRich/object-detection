#!/usr/bin/env python3
"""Algorithm 2 — gộp map + NMS. CPU, không cần GPU/SD/dữ liệu.

Khoá bốn tính chất mà nếu sai thì KHÔNG crash, chỉ ra số sai:

  1. Khai triển KL phải bằng định nghĩa ngây thơ. Dạng literal tốn K*K*N —
     ở K=529, N=19600 là 43 GB — nên bắt buộc khai triển, và khai triển sai
     vẫn cho một ma trận khoảng cách trông hợp lý.
  2. argmax qua các cụm phải cho PHÂN HOẠCH. Đây là chỗ GĐ2 khác GĐ1 về bản
     chất: không ngưỡng từng map, mỗi pixel thuộc đúng một cụm.
  3. Số cụm phải GIẢM khi h tăng. Nếu ngược lại thì dải height vô nghĩa và
     "6 mức granularity" chỉ là 6 lần chạy cùng một thứ.
  4. NMS tau_IoU=0.9 phải GIỮ mask lồng nhau. Cái ghế và mặt ghế đều là mục
     tiêu; NMS chặt sẽ xoá mất một nửa số mức granularity.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d2s.merging import (area_descending_nms, cluster_at_heights,  # noqa: E402
                         masks_from_clusters, normalise_maps,
                         symmetric_kl_matrix)


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


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
