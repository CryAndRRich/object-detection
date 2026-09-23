"""Test cho EXPERIMENT A — mỗi test khoá một bất biến mà nếu vỡ thì mô hình VẪN CHẠY.

Đây là tiêu chí chọn test: box vẫn nằm trong [0,1], loss vẫn giảm, không assert nào kêu,
và kết luận thì sai. Vòng 1 mất nhiều giờ A30 cho đúng loại lỗi này.

Chạy: `python -m pytest tests/test_experiment_a.py -q`
(KHÔNG chạy `pytest tests/` trần — các file test vừa là script `main()` vừa có wrapper
`test_*`, và lệnh trần từng báo "no tests ran" mà vẫn exit 0.)
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.criterion import SetCriterion                            # noqa: E402
from models.dit_blocks import (                                      # noqa: E402
    MIN_WH,
    BoxCoordEmbedder,
    DiTBlock,
    build_cross_mask,
    clamp_to_valid,
    update_box,
)

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


# --------------------------------------------------------------- update_box

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


# ------------------------------------------------------------------- mask

def test_mask_chan_dung_o_r_sang_h():
    """Chỉ ô `r -> h` bị chặn. Nếu chặn nhầm `h -> r` thì `delta` mất nguồn học chính
    và mô hình vẫn chạy, chỉ kém đi — không có gì báo."""
    m = build_cross_mask(N, torch.device("cpu"))
    assert m.shape == (2 * N, 2 * N)
    assert not m[:N, :N].any(), "h->h phải mở"
    assert not m[:N, N:].any(), "h->r phải mở (đường chính của delta)"
    assert m[N:, :N].all(), "r->h phải CHẶN"
    assert not m[N:, N:].any(), "r->r phải mở"


# --------------------------------------------------------------- valid_h

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


# ------------------------------------------------------------- DiTBlock

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


# ------------------------------------------------------------- criterion

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
    from utils.matcher import simota_match
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


# ------------------------------------------------------------- embedder

def test_coord_embed_phan_biet_duoc_vi_tri():
    """Sin/cos cho mỗi vị trí một chữ ký riêng. Nếu hai box khác nhau cho cùng embedding
    thì attention không phân biệt được chúng."""
    emb = BoxCoordEmbedder(D)
    a = emb(torch.tensor([[[0.2, 0.5, 0.1, 0.1]]]))
    b = emb(torch.tensor([[[0.4, 0.5, 0.1, 0.1]]]))
    assert not torch.allclose(a, b, atol=1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# Cửa chặn gate_delta_direction — các phần thêm sau khi phát hiện 2 lỗi đo
# ---------------------------------------------------------------------------

def test_sample_roi_khop_bit_voi_forward_cua_sampler():
    """`sample_roi` tự viết lại đường đi của `RoIFeatureSampler.forward` để lấy được
    đặc trưng TRƯỚC lớp `out`. Nếu hai đường lệch nhau thì cột `cos2304` đo một thứ
    không tồn tại trong model thật."""
    import torch.nn as nn
    from models.roi_sampler import RoIFeatureSampler
    from tools.gate_delta_direction import sample_roi

    torch.manual_seed(0)
    s = RoIFeatureSampler(768, 256, 3, 0.0).eval()
    nn.init.xavier_uniform_(s.out.weight)          # như cửa chặn làm
    nn.init.zeros_(s.out.bias)
    praw = torch.randn(1, 1024, 768)
    box = torch.rand(7, 4) * 0.3 + 0.3

    feat, r = sample_roi(s, praw, box)
    with torch.no_grad():
        ref = s(praw, box.unsqueeze(0))[0]

    assert feat.shape == (7, 3 * 3 * 256)
    assert torch.allclose(r, ref, atol=1e-6), float((r - ref).abs().max())


def test_nhieu_khuech_tan_nuot_bien_d_o_t_lon():
    """Lý do `t=-1` phải có trong mặc định: ở `t` lớn, `d` không còn dấu vết.

    Đây là lỗi đo đã làm 4 hàng `t=999` của lần chạy đầu trùng khít nhau."""
    import numpy as np
    import yaml
    from tools.gate_delta_direction import perturb, true_delta, add_diffusion_noise
    from utils.diffusion_math import cosine_alphas_cumprod

    cfg = yaml.safe_load(open("config/experiment_a.yaml"))
    al = cosine_alphas_cumprod(cfg["diffusion"]["num_timesteps"]).float()
    snr = cfg["diffusion"]["snr_scale"]
    gt = torch.stack([torch.rand(4000) * 0.6 + 0.2, torch.rand(4000) * 0.6 + 0.2,
                      torch.full((4000,), 1.96 / 32), torch.full((4000,), 1.70 / 32)], -1)

    def tuong_quan(t, d):
        rng = np.random.default_rng(0)
        box = perturb(gt, d, rng)
        dirn = box[:, :2] - gt[:, :2]
        if t is not None:
            box = add_diffusion_noise(box, t, al, snr, rng)
        dt = true_delta(box, gt)
        return float(torch.nn.functional.cosine_similarity(dt[:, :2], -dirn, dim=-1).mean())

    # Không nhiễu: delta thật chỉ đường về, ngược đúng hướng đã dịch.
    assert tuong_quan(None, 1.0) > 0.99
    # t=999: alpha_bar=0, box là nhiễu thuần -> `d` bị xoá sạch.
    assert abs(tuong_quan(999, 1.0)) < 0.05
    assert abs(tuong_quan(999, 4.0) - tuong_quan(999, 0.5)) < 0.05


def test_lstsq_la_tran_khong_thap_hon_adamw():
    """Cột `lstsq` phải là TRẦN: nếu nó thấp hơn AdamW thì nó bị hỏng và không còn
    phân biệt được 'đặc trưng vô dụng' với 'train chưa đủ'."""
    from tools.gate_delta_direction import fit_and_score

    torch.manual_seed(0)
    K, D = 3000, 64
    W = torch.randn(D, 4) * 0.1
    r = torch.randn(K, D)
    d = r @ W + torch.randn(K, 4) * 0.3
    w = torch.full((K,), 2.0)
    cut = int(K * 0.7)
    out = fit_and_score(r[:cut], d[:cut], r[cut:], d[cut:], w[cut:],
                        400, 1e-3, torch.device("cpu"), 0)
    assert out["cosine_lstsq"] >= out["cosine"] - 0.02
    assert out["cosine_lstsq"] > 0.5        # quan hệ có thật thì phải bắt được


def test_nut_that_ngau_nhien_lam_mat_tin_hieu():
    """Bằng chứng cho lỗi đo thứ 2: chiếu ngẫu nhiên 2304->256 huỷ phần lớn tín hiệu,
    nên tiêu chí phải đọc trên cột TRƯỚC nút thắt."""
    import torch.nn as nn
    from tools.gate_delta_direction import fit_and_score

    torch.manual_seed(0)
    K, S = 3000, 2304
    d = torch.randn(K, S) @ (torch.randn(S, 4) * 0.05)
    feat = torch.randn(K, S)
    d = feat @ (torch.randn(S, 4) * 0.05) + torch.randn(K, 4) * 0.2
    w = torch.full((K,), 2.0)
    cut = int(K * 0.7)
    proj = nn.Linear(S, 256)
    nn.init.xavier_uniform_(proj.weight)
    nn.init.zeros_(proj.bias)
    with torch.no_grad():
        z = proj(feat)

    dev = torch.device("cpu")
    truoc = fit_and_score(feat[:cut], d[:cut], feat[cut:], d[cut:], w[cut:], 400, 1e-3, dev, 0)
    sau = fit_and_score(z[:cut], d[:cut], z[cut:], d[cut:], w[cut:], 400, 1e-3, dev, 0)
    assert truoc["cosine_lstsq"] > sau["cosine_lstsq"] + 0.2


# ---------------------------------------------------------------------------
# Chẩn đoán gate_delta_ablation — 7 giả thuyết vì sao cos2304 chỉ 0,266
# ---------------------------------------------------------------------------

def test_grid_points_scale_1_khop_box_grid_points_goc():
    """`grid_points(scale=1.0)` phải TRÙNG `box_grid_points` của model thật, nếu không
    thì cột 'nới 1.0x' không phải mốc so sánh hợp lệ cho các mức nới khác."""
    from models.roi_sampler import box_grid_points
    from tools.gate_delta_ablation import grid_points

    torch.manual_seed(0)
    box = torch.rand(11, 4) * 0.3 + 0.3
    for k in (1, 3, 5, 7):
        a = grid_points(box, k, 1.0)
        b = box_grid_points(box.unsqueeze(0), k)
        assert torch.allclose(a, b, atol=1e-6), (k, float((a - b).abs().max()))


def test_grid_points_noi_rong_vuot_ra_ngoai_box():
    """Giả thuyết 4 chỉ có nghĩa nếu `scale>1` thật sự lấy mẫu NGOÀI box."""
    from tools.gate_delta_ablation import grid_points

    box = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    trong = grid_points(box, 3, 1.0)[0, 0]
    ngoai = grid_points(box, 3, 2.0)[0, 0]
    # trong box: mọi điểm nằm trong [cx-w/2, cx+w/2] = [0.4, 0.6]
    assert float(trong[:, 0].min()) >= 0.4 - 1e-6
    assert float(trong[:, 0].max()) <= 0.6 + 1e-6
    # nới 2x: phải có điểm vượt ra ngoài
    assert float(ngoai[:, 0].min()) < 0.4
    assert float(ngoai[:, 0].max()) > 0.6


def test_sample_raw_bo_dung_nut_that_proj_point():
    """`sample_raw` phải trả CLIP THÔ k*k*768, và bằng đúng đặc trưng mà `proj_point`
    nhận làm đầu vào — nếu lệch thì giả thuyết 2 đo nhầm thứ."""
    import torch.nn as nn
    from models.roi_sampler import RoIFeatureSampler
    from tools.gate_delta_ablation import sample_raw
    from tools.gate_delta_direction import sample_roi

    torch.manual_seed(0)
    s = RoIFeatureSampler(768, 256, 3, 0.0).eval()
    nn.init.xavier_uniform_(s.out.weight); nn.init.zeros_(s.out.bias)
    praw = torch.randn(1, 1024, 768)
    box = torch.rand(6, 4) * 0.3 + 0.3

    raw = sample_raw(praw, box, 3, 1.0)
    assert raw.shape == (6, 9 * 768)
    # đưa raw qua proj_point phải ra đúng feat 2304-d của cửa chặn
    with torch.no_grad():
        lai = s.proj_point(raw.view(6, 9, 768)).flatten(-2)
    feat, _ = sample_roi(s, praw, box)
    assert torch.allclose(lai, feat, atol=1e-5), float((lai - feat).abs().max())


def test_knn_bat_duoc_quan_he_phi_tuyen_ma_linear_bo_lo():
    """k-NN phải thật sự là trần mạnh hơn tuyến tính, nếu không thì kết luận
    'đặc trưng không chứa thông tin' dựa trên nó là vô giá trị."""
    from tools.gate_delta_ablation import probe_knn, probe_linear, _cos

    torch.manual_seed(0)
    K, D = 4000, 32
    X = torch.randn(K, D)
    # Phi tuyến thuần trên CẢ HAI kênh tâm. `.abs()` không dùng được ở đây: nó luôn
    # dương nên Linear đoán trúng bằng một hằng số, che mất việc nó không học được gì.
    d = torch.stack([X[:, 0] * X[:, 1], X[:, 2] * X[:, 3],
                     torch.zeros(K), torch.zeros(K)], dim=-1)
    cut = int(K * 0.7)
    dev = torch.device("cpu")
    lin = _cos(probe_linear(X[:cut], d[:cut], X[cut:], d[cut:], dev, 0), d[cut:])
    knn = _cos(probe_knn(X[:cut], d[:cut], X[cut:], dev, 10), d[cut:])
    assert knn > lin + 0.2, (knn, lin)


def test_probe_mlp_hoc_duoc_quan_he_phi_tuyen():
    """MLP + early stop phải hội tụ trên quan hệ phi tuyến có thật."""
    from tools.gate_delta_ablation import probe_mlp, _cos

    torch.manual_seed(0)
    K, D = 5000, 32
    X = torch.randn(K, D)
    d = torch.stack([X[:, 0] * X[:, 1], X[:, 2] * X[:, 3],
                     torch.zeros(K), torch.zeros(K)], dim=-1)
    cut = int(K * 0.7)
    pred = probe_mlp(X[:cut], d[:cut], X[cut:], d[cut:], torch.device("cpu"), 0,
                     hidden=256, layers=2, epochs=4000)
    assert _cos(pred, d[cut:]) > 0.5
