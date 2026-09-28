"""Decoder BoxDiT + criterion: mỗi test khoá một bất biến mà nếu vỡ thì mô hình VẪN CHẠY
(box vẫn trong [0,1], loss vẫn giảm) nhưng kết luận sai."""

import os
import sys

import numpy as np
import pytest
import torch

from ce_localization.models.criterion import SetCriterion
from ce_localization.models.dit_blocks import (                                    
    MIN_WH,
    BoxCoordEmbedder,
    DiTBlock,
    build_cross_mask,
    clamp_to_valid,
    update_box,
)

PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ce_localization")

# ============================================================================
# từ test_experiment_a.py
# ============================================================================

B, N, D, P, DIN = 2, 5, 32, 64, 24          # P=64 -> lưới 8x8, là số chính phương


def _block(**kw):
    torch.manual_seed(0)
    return DiTBlock(d_model=D, n_head=4, dim_feedforward=2 * D, dropout=0.0,
                    roi_dim=DIN, roi_k=3, **kw)


def _inputs(n=N):
    torch.manual_seed(1)
    x = torch.rand(B, n, 4) * 0.4 + 0.3                     # tránh sát biên
    h = torch.randn(B, n, D)
    r = torch.randn(B, n, D)
    mem = torch.randn(B, 10, D)
    praw = torch.randn(B, P, DIN)
    t = torch.randn(B, D)
    return x, h, r, mem, praw, t


def test_delta_zero_giu_nguyen_box():
    """`delta = 0` phải trả về ĐÚNG box vào. Đây là điều khiến `box_delta` zero-init
    làm mọi tầng thành ánh xạ đồng nhất ở bước 0 — nếu vỡ thì mô hình đã dịch box ngay
    trước khi học được gì, và không có gì báo."""
    x = torch.rand(3, 7, 4) * 0.5 + 0.25
    out = update_box(x, torch.zeros_like(x))
    assert torch.allclose(out, x, atol=1e-6)


def test_w_bang_0_van_dich_duoc_box():
    """Lỗ hổng mục 7.1b của kế hoạch: ở `t` lớn, `w` sau decode có đuôi chạm 0. Không
    clamp thì `cx + d_cx * w` đứng yên BẤT KỂ delta lớn cỡ nào, làm chuỗi cộng dồn vô
    nghĩa ở đúng những bước cần nó nhất."""
    x = torch.tensor([[[0.5, 0.5, 0.0, 0.0]]])
    out = update_box(x, torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]))
    assert out[0, 0, 0] > 0.5, "w=0 làm box đứng yên -> thiếu clamp MIN_WH"
    assert out[0, 0, 2] >= MIN_WH
    assert torch.isfinite(out).all()


def test_kich_thuoc_luon_duong():
    """Nhân `exp(d_w)` nên không bao giờ âm; clamp giữ trong [MIN_WH, 1]."""
    x = torch.rand(4, 6, 4).clamp(0.05, 0.9)
    out = update_box(x, torch.randn(4, 6, 4) * 5.0)          # delta rất lớn
    assert (out[..., 2:] >= MIN_WH).all() and (out[..., 2:] <= 1.0).all()
    assert torch.isfinite(out).all()


def test_mask_chan_dung_o_r_sang_h():
    """Chỉ ô `r -> h` bị chặn. Nếu chặn nhầm `h -> r` thì `delta` mất nguồn học chính
    và mô hình vẫn chạy, chỉ kém đi — không có gì báo."""
    m = build_cross_mask(N, torch.device("cpu"))
    assert m.shape == (2 * N, 2 * N)
    assert not m[:N, :N].any(), "h->h phải mở"
    assert not m[:N, N:].any(), "h->r phải mở (đường chính của delta)"
    assert m[N:, :N].all(), "r->h phải CHẶN"
    assert not m[N:, N:].any(), "r->r phải mở"


def test_clamp_keo_cy_ve_vung_anh():
    """Box do mô hình sinh có thể rơi vào vùng đệm ở đáy (valid_h trung vị chỉ 0,707).
    Lấy mẫu ở đó cho đặc trưng của một mảng phẳng, hoàn toàn vô nghĩa."""
    x = torch.tensor([[[0.5, 0.95, 0.1, 0.1], [0.5, 0.20, 0.1, 0.1]]])
    out = clamp_to_valid(x, torch.tensor([0.7]))
    assert out[0, 0, 1] == pytest.approx(0.7)       # bị kéo về
    assert out[0, 1, 1] == pytest.approx(0.20)      # vốn đã hợp lệ, giữ nguyên
    assert torch.equal(out[..., [0, 2, 3]], x[..., [0, 2, 3]])   # chỉ đụng cy


def test_clamp_khong_sua_tai_cho():
    x = torch.tensor([[[0.5, 0.95, 0.1, 0.1]]])
    before = x.clone()
    clamp_to_valid(x, torch.tensor([0.5]))
    assert torch.equal(x, before), "clamp_to_valid không được sửa tensor đầu vào"


def test_valid_h_none_khong_doi_gi():
    x = torch.rand(2, 3, 4)
    assert torch.equal(clamp_to_valid(x, None), x)


def test_tang_zero_init_tra_ve_dung_box_vao():
    """adaLN gate và `box_delta` đều zero-init, nên tầng KHÔNG được dịch box ở bước 0.
    Bất biến này là cách so sánh sạch: mọi thay đổi sau đều bắt đầu từ cùng một chỗ."""
    blk = _block().eval()
    x, h, r, mem, praw, t = _inputs()
    mask = build_cross_mask(N, x.device)
    with torch.no_grad():
        x2, _, _ = blk(x, h, r, mem, praw, t, mask)
    assert torch.allclose(x2, x, atol=1e-6)


def test_gradient_khong_chay_qua_toa_do_lay_mau():
    """PHÉP KIỂM QUAN TRỌNG NHẤT. `grid_sample` khả vi theo toạ độ lấy mẫu, nên nếu
    thiếu `.detach()` thì loss của score chảy ngược qua r -> roi -> x và mạng học DỊCH
    BOX TỚI CHỖ DỄ CHẤM ĐIỂM thay vì chỗ có vật.

    Vòng 1 đo được hậu quả khi vòng phản hồi này mở: độ ổn định nhãn sụp.
    """
    blk = _block()
    x, h, r, mem, praw, t = _inputs()
    x = x.clone().requires_grad_(True)
    mask = build_cross_mask(N, x.device)
    _, _, r_out = blk(x, h, r, mem, praw, t, mask)
    r_out.sum().backward()                      # loss CHỈ trên nhánh ảnh
    assert x.grad is None or float(x.grad.abs().max()) == 0.0, (
        "gradient chảy ngược từ r về toạ độ -> thiếu .detach() ở chỗ lấy mẫu RoI")


def test_box_van_hop_le_sau_nhieu_tang():
    """Cộng dồn qua nhiều tầng từ điểm xuất phát ngẫu nhiên là rủi ro đã biết
    (V-DETR/D-FINE cố ý dùng gốc cố định để tránh). Ít nhất box phải luôn hữu hạn và
    kích thước dương."""
    blk = _block()
    x, h, r, mem, praw, t = _inputs()
    mask = build_cross_mask(N, x.device)
    for _ in range(6):
        x, h, r = blk(x, h, r, mem, praw, t, mask)
    assert torch.isfinite(x).all()
    assert (x[..., 2:] >= MIN_WH).all()


def test_hoan_vi_box_thi_ket_qua_hoan_vi_theo():
    """Box là một TẬP, không phải chuỗi có thứ tự: không có pos_emb theo chỉ số, nên
    đảo thứ tự box phải cho cùng kết quả đã đảo. Nếu vỡ thì mô hình đang học "khe 0
    thường là GT thật" — đúng thứ nó không được học, vì lúc suy luận mọi khe đều từ
    randn."""
    blk = _block().eval()
    x, h, r, mem, praw, t = _inputs()
    mask = build_cross_mask(N, x.device)
    perm = torch.randperm(N)
    with torch.no_grad():
        a, _, _ = blk(x, h, r, mem, praw, t, mask)
        b, _, _ = blk(x[:, perm], h[:, perm], r[:, perm], mem, praw, t, mask)
    assert torch.allclose(a[:, perm], b, atol=1e-5)


def test_moi_tang_co_roi_rieng():
    """Mỗi tầng lấy mẫu lại tại toạ độ MỚI — đó là lý do tồn tại của thiết kế. Nếu các
    tầng dùng chung một `RoIFeatureSampler` thì chúng buộc phải diễn giải đặc trưng
    giống nhau dù phân phối toạ độ đã đổi."""
    blks = [_block() for _ in range(3)]
    ids = {id(b.roi) for b in blks}
    assert len(ids) == 3


def test_loss_la_TONG_khong_phai_trung_binh():
    """DiffusionDet/DETR/V-DETR đều cộng loss các tầng. Chia trung bình làm gradient tới
    mỗi tầng bị nhân 1/6 — mô hình vẫn train, chỉ chậm hơn 6 lần, và không có gì báo."""
    crit = SetCriterion(matcher_method="hungarian")
    torch.manual_seed(2)
    boxes = torch.rand(B, N, 4) * 0.3 + 0.35
    logits = torch.randn(B, N)
    targets = [torch.rand(3, 4) * 0.3 + 0.35 for _ in range(B)]

    one, _, _ = crit([(boxes, logits)], targets)
    three, st, _ = crit([(boxes, logits)] * 3, targets)
    assert float(three) == pytest.approx(3 * float(one), rel=1e-5)
    assert st["loss_mean"] == pytest.approx(float(one), rel=1e-5)
    assert st["n_layers"] == 3


def test_stats_co_duong_cong_theo_tang():
    """`*_per_layer` là chỉ số CHÍNH để đọc thí nghiệm: đường phẳng nghĩa là cộng dồn
    không mang lại gì."""
    crit = SetCriterion(matcher_method="hungarian")
    torch.manual_seed(3)
    layers = [(torch.rand(B, N, 4) * 0.3 + 0.35, torch.randn(B, N)) for _ in range(4)]
    targets = [torch.rand(2, 4) * 0.3 + 0.35 for _ in range(B)]
    _, st, _ = crit(layers, targets)
    assert len(st["iou_matched_per_layer"]) == 4
    assert st["loss_final"] == pytest.approx(st["loss_per_layer"][-1], rel=1e-5)


def test_criterion_tu_choi_dau_vao_sai_kieu():
    """Vòng 1 dispatch theo kiểu dữ liệu và bốn công cụ đã giải nén nhầm. Ở đây chữ ký
    chỉ có một dạng, và sai thì phải ném lỗi ngay chứ không âm thầm chạy."""
    crit = SetCriterion()
    with pytest.raises(TypeError):
        crit((torch.rand(B, N, 4), torch.randn(B, N)), [torch.rand(2, 4)])
    with pytest.raises(TypeError):
        crit([], [torch.rand(2, 4)])


def test_simota_chay_duoc():
    """SimOTA là matcher của EXPERIMENT A (DiffusionDet vốn dùng nó, không dùng
    Hungarian). Kiểm nó chạy và gán MỌI GT ít nhất một proposal."""
    crit = SetCriterion(matcher_method="simota")
    torch.manual_seed(4)
    boxes = torch.rand(1, 20, 4) * 0.3 + 0.35
    logits = torch.randn(1, 20)
    targets = [torch.rand(4, 4) * 0.3 + 0.35]
    loss, st, idx = crit([(boxes, logits)], targets)
    assert torch.isfinite(loss)
    assert len(set(idx[0][1].tolist())) == 4, "SimOTA phải gán mọi GT"


def test_simota_khong_vo_khi_nhieu_GT_hon_box():
    """LỖI THẬT tìm được khi chạy end-to-end lần đầu.

    Mỗi proposal nhận nhiều nhất 1 GT, nên khi `n_gt > N` thì về mặt toán học KHÔNG THỂ
    gán hết — vòng cứu của SimOTA quay vô vọng rồi `assert` nổ. Vòng 1 không gặp vì
    N=100 so với trung vị 20-30 GT; vòng 2 chạy N=30 với đúng các trung vị ấy, nên ảnh
    có `n_gt >= N` là CHUYỆN THƯỜNG (test có trung vị đúng 30).
    """
    from ce_localization.utils.matcher import simota_match
    torch.manual_seed(7)
    for n_gt in (5, 25, 30, 60):
        pred = torch.rand(30, 4) * 0.4 + 0.3
        gt = torch.rand(n_gt, 4) * 0.3 + 0.35
        pi, gi = simota_match(pred, gt, torch.randn(30), use_center_prior=True,
                              radius_ratio=2.5, top_k=10)
        assert len(pi) == len(gi)
        assert (torch.bincount(pi, minlength=30) <= 1).all(), \
            "một proposal bị gán cho nhiều GT"
        assert len(pi) <= 30


def test_criterion_chay_khi_GT_nhieu_hon_box():
    """Cùng tình huống, nhưng đi qua criterion — nơi nó thực sự nổ lần đầu."""
    crit = SetCriterion(matcher_method="simota", use_center_prior=True,
                            radius_ratio=2.5, top_k=10)
    torch.manual_seed(8)
    layers = [(torch.rand(2, 30, 4) * 0.3 + 0.35, torch.randn(2, 30))]
    targets = [torch.rand(12, 4) * 0.3 + 0.35, torch.rand(40, 4) * 0.3 + 0.35]
    loss, st, _ = crit(layers, targets)
    assert torch.isfinite(loss)
    assert st["n_matched"] > 0


def test_coord_embed_phan_biet_duoc_vi_tri():
    """Sin/cos cho mỗi vị trí một chữ ký riêng. Nếu hai box khác nhau cho cùng embedding
    thì attention không phân biệt được chúng."""
    emb = BoxCoordEmbedder(D)
    a = emb(torch.tensor([[[0.2, 0.5, 0.1, 0.1]]]))
    b = emb(torch.tensor([[[0.4, 0.5, 0.1, 0.1]]]))
    assert not torch.allclose(a, b, atol=1e-3)


def test_max_cond_len_theo_do_phan_giai_that():
    """`cond_pos_emb` phải đủ chỗ cho memory ở ĐỘ PHÂN GIẢI ĐANG DÙNG.

    Hằng 1152 đủ cho 512px (1024 patch + 1 text) nhưng KHÔNG đủ cho 1024px (4096 + 1):
    train trên cache 1024px sẽ ném lỗi ngay batch đầu."""
    from ce_localization.models.detector import BoxDiT

    for image_size, n_patch in [(512, 1024), (1024, 4096)]:
        dec = BoxDiT(64, 1, 2, 16, max_cond_len=n_patch + 128)
        assert dec.cond_pos_emb.shape[1] >= n_patch + 1, (image_size, n_patch)


def test_build_inputs_boc_t_rieng_cho_tung_anh():
    """DiffusionDet bốc `t` cho TỪNG ảnh. Bản cũ bốc một `t` cho cả batch."""
    from types import SimpleNamespace

    from ce_localization.models.detector import CELocDetector
    from ce_localization.utils.diffusion_math import cosine_alphas_cumprod

    fake = SimpleNamespace(alphas_cumprod=cosine_alphas_cumprod(1000), num_timesteps=1000,
                           snr_scale=2.0)
    gt = [torch.tensor([[0.5, 0.5, 0.1, 0.1]])] * 16
    g = torch.Generator().manual_seed(0)
    x_t, t, _ = CELocDetector.build_inputs(fake, gt, 30, [1.0] * 16, generator=g)
    assert x_t.shape == (16, 30, 4) and t.shape == (16,) and t.dtype == torch.long
    assert len(set(t.tolist())) > 8, t.tolist()          # 16 ảnh, gần như chắc khác nhau

