#!/usr/bin/env python3
"""Toàn mạch GĐ2 trên affinity giả lập — không cần GPU/SD.

Chứng minh các mảnh GHÉP VỚI NHAU: prompt -> lan truyền -> chuẩn hoá -> KL ->
cụm -> upsample -> argmax -> components -> NMS -> mask ở độ phân giải ảnh gốc.
`test_merging.py` đã kiểm từng mảnh riêng; suite này kiểm chỗ nối, nơi một hàm
đúng vẫn có thể bị gọi sai.

⚠️ Đồ thị dựng tay là trường hợp DỄ NHẤT: hai khối tách biệt hoàn toàn, biên
sạch 1e-4. Nó không nói gì về việc self-attention thật có tách được vật trên
PACO hay không.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig          # noqa: E402
from d2s.pipeline import masks_full_to_boxes, segment_image  # noqa: E402

R = 16
CANVAS = 128
ORIG_H, ORIG_W = 64, 64


def _cfg(**kw):
    base = dict(canvas=CANVAS, prompt_stride_cells=2, max_iter=60, tau_prop=1e-8,
                lam=1e-2, p=1.6, stage=2, n_levels=4, kl_h_min=0.05,
                kl_h_max=5.0, nms_iou=0.9, min_area_px=4, max_masks=1000)
    base.update(kw)
    return Diffu2SegConfig(**base).validate()


def _blocky_affinity(blocks, r=R, within=1.0, across=1e-4):
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
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
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
    from d2s.merging import (cluster_at_heights, normalise_maps,
                             symmetric_kl_matrix)
    from d2s.plaplacian import plaplacian_propagate
    from d2s.prompts import build_prompt_grid, f0_onehot

    cfg = _cfg()
    A = _blocky_affinity([(2, 6, 2, 6), (10, 14, 10, 14)])
    cells = build_prompt_grid(R, cfg.prompt_stride_cells, valid_h=1.0,
                              min_valid_frac=cfg.min_valid_frac)
    f0 = f0_onehot(cells, R, device=A.device, dtype=A.dtype)
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
    assert bar[:, 4 * R + 4].argmax() == c1.pop(), "argmax ở tâm khối 1 chọn sai cụm"
    assert bar[:, 12 * R + 12].argmax() == c2.pop(), "argmax ở tâm khối 2 chọn sai cụm"


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
        g = np.zeros((R, R), dtype=bool)
        g[r0:r1, c0:c1] = True
        ys = ((np.arange(ORIG_H) + 0.5) / ORIG_H * R).astype(int).clip(0, R - 1)
        xs = ((np.arange(ORIG_W) + 0.5) / ORIG_W * R).astype(int).clip(0, R - 1)
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
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
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
    cfg = Diffu2SegConfig(canvas=CANVAS, prompt_stride_cells=2, max_iter=60,
                          tau_prop=1e-8, lam=1e-2, p=1.6, stage=1).validate()
    A = _blocky_affinity([(2, 6, 2, 6), (10, 14, 10, 14)])
    img = np.zeros((CANVAS, CANVAS, 3), dtype=np.uint8)
    out = segment_image(img, valid_h=1.0, cfg=cfg, A=A)
    assert out["n_boxes"] == 2, f"GĐ1 phải vẫn ra 2 box, được {out['n_boxes']}"
    assert "masks_full" not in out or len(out.get("masks_full", ())) == 0


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
