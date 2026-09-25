"""Test cho EXPERIMENT A — mỗi test khoá một bất biến mà nếu vỡ thì mô hình VẪN CHẠY.

Đây là tiêu chí chọn test: box vẫn nằm trong [0,1], loss vẫn giảm, không assert nào kêu,
và kết luận thì sai. Vòng 1 mất nhiều giờ A30 cho đúng loại lỗi này.

Chạy: `python -m pytest tests/test_experiment_a.py -q`
(KHÔNG chạy `pytest tests/` trần — các file test vừa là script `main()` vừa có wrapper
`test_*`, và lệnh trần từng báo "no tests ran" mà vẫn exit 0.)
"""

import os
import sys

import numpy as np
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


# ---------------------------------------------------------------------------
# gate_grid_resolution — ngoại suy độ phân giải lưới
# ---------------------------------------------------------------------------

def test_sample_at_grid_giu_nguyen_khi_khong_ha_luoi():
    """`g_new == g` phải là no-op, khớp `sample_raw`. Nếu lệch thì hàng 'lưới 32'
    không so được với chẩn đoán trước đó."""
    from tools.gate_delta_ablation import sample_raw
    from tools.gate_grid_resolution import sample_at_grid

    torch.manual_seed(0)
    praw = torch.randn(1, 1024, 768)
    box = torch.rand(9, 4) * 0.3 + 0.3
    a = sample_at_grid(praw, box, 3, 1.0, 32)
    b = sample_raw(praw, box, 3, 1.0)
    assert torch.allclose(a, b, atol=1e-6), float((a - b).abs().max())


def test_ha_luoi_lam_mat_chi_tiet_khong_gian():
    """Hạ lưới phải THỰC SỰ làm mất chi tiết: hai điểm cách nhau dưới một ô của lưới
    mới phải cho cùng giá trị. Nếu không thì phép ngoại suy đo một thứ giả."""
    from tools.gate_grid_resolution import sample_at_grid

    torch.manual_seed(0)
    praw = torch.randn(1, 1024, 768)
    # hai box lệch nhau 1 ô của lưới 32 (=1/32), nhưng cùng một ô của lưới 8
    b1 = torch.tensor([[0.5, 0.5, 0.02, 0.02]])
    b2 = torch.tensor([[0.5 + 1.0 / 32, 0.5, 0.02, 0.02]])
    v32 = (sample_at_grid(praw, b1, 1, 1.0, 32) - sample_at_grid(praw, b2, 1, 1.0, 32))
    v8 = (sample_at_grid(praw, b1, 1, 1.0, 8) - sample_at_grid(praw, b2, 1, 1.0, 8))
    # ở lưới 32 hai vị trí khác nhau rõ; ở lưới 8 gần như trùng
    assert float(v32.abs().mean()) > float(v8.abs().mean()) * 2


def test_avg_pool_khong_phai_lay_thua():
    """Hạ lưới bằng average-pool giữ trung bình vùng. Kiểm bằng feature map hằng số
    theo khối: pool phải trả đúng giá trị khối đó."""
    from tools.gate_grid_resolution import sample_at_grid

    # feature map: nửa trái = 1, nửa phải = 3 -> pool 32->8 vẫn giữ 1 và 3
    fm = torch.ones(1, 1, 32, 32)
    fm[:, :, :, 16:] = 3.0
    praw = fm.reshape(1, 1, 1024).transpose(1, 2)          # [1,1024,1]
    trai = sample_at_grid(praw, torch.tensor([[0.25, 0.5, 0.02, 0.02]]), 1, 1.0, 8)
    phai = sample_at_grid(praw, torch.tensor([[0.75, 0.5, 0.02, 0.02]]), 1, 1.0, 8)
    assert abs(float(trai) - 1.0) < 1e-4, float(trai)
    assert abs(float(phai) - 3.0) < 1e-4, float(phai)


def test_set_grid_lam_d_cells_dung_don_vi_o_o_moi_do_phan_giai():
    """`perturb(d=1)` phải dịch đúng MỘT ô của lưới ĐANG dùng, ở mọi độ phân giải.

    Không có `set_grid`, `GRID=32` cố định sẽ làm `d=1 ô` trên cache 1024px dịch thật ra
    2 ô của lưới 64 — bảng so sánh hai độ dịch khác nhau, không assert nào bắt.
    (Cạm bẫy 1: sai âm thầm.)
    """
    import numpy as np
    import tools.gate_delta_direction as gdd

    cu = gdd.GRID
    try:
        box = torch.tensor([[0.5, 0.5, 0.06, 0.05]])
        for n_token, g in [(1024, 32), (4096, 64), (256, 16)]:
            assert gdd.set_grid(n_token) == g
            out = gdd.perturb(box, 1.0, np.random.default_rng(0))
            dich_chuan_hoa = float((out[0, :2] - box[0, :2]).norm())
            assert abs(dich_chuan_hoa * g - 1.0) < 1e-4, (g, dich_chuan_hoa * g)
    finally:
        gdd.GRID = cu


def test_set_grid_tu_choi_so_token_khong_vuong():
    import pytest

    import tools.gate_delta_direction as gdd
    cu = gdd.GRID
    try:
        with pytest.raises(AssertionError):
            gdd.set_grid(1000)
    finally:
        gdd.GRID = cu


def test_probe_mlp_on_dinh_tren_dac_trung_thang_lech():
    """Hồi quy cho lỗi @1024: `mlp 3 lớp` cho 0,077 và `k=7 nới 2,0x` cho 0,030 —
    THẤP HƠN CẢ mức sàn k=1. Nguyên nhân là đặc trưng thang lệch + lr cố định làm
    phân kỳ. Sau khi chuẩn hoá + quét lr, mạng sâu KHÔNG được tệ hơn mạng nông."""
    from tools.gate_delta_ablation import probe_mlp, _cos

    torch.manual_seed(0)
    K, D = 3000, 512
    X = torch.randn(K, D)
    # thang rất lệch giữa các chiều, như đặc trưng CLIP thô
    X = X * torch.logspace(-2, 3, D).unsqueeze(0)
    W = torch.randn(D, 4) * 0.01
    d = X @ W + torch.randn(K, 4) * 0.3
    cut = int(K * 0.7)
    dev = torch.device("cpu")

    nong = _cos(probe_mlp(X[:cut], d[:cut], X[cut:], d[cut:], dev, 0,
                          hidden=256, layers=2, epochs=800), d[cut:])
    sau = _cos(probe_mlp(X[:cut], d[:cut], X[cut:], d[cut:], dev, 0,
                         hidden=256, layers=3, epochs=800), d[cut:])
    assert nong > 0.3, nong
    assert sau > nong - 0.15, (sau, nong)   # sâu hơn không được sụp đổ


def test_max_cond_len_theo_do_phan_giai_that():
    """`cond_pos_emb` phải đủ chỗ cho memory ở ĐỘ PHÂN GIẢI ĐANG DÙNG.

    Hằng 1152 đủ cho 512px (1024 patch + 1 text) nhưng KHÔNG đủ cho 1024px (4096 + 1):
    train trên cache 1024px sẽ ném lỗi ngay batch đầu."""
    from models.detector import BoxDiT

    for image_size, n_patch in [(512, 1024), (1024, 4096)]:
        dec = BoxDiT(64, 1, 2, 16, max_cond_len=n_patch + 128)
        assert dec.cond_pos_emb.shape[1] >= n_patch + 1, (image_size, n_patch)


# ---------------------------------------------------------------------------
# Checkpoint: last.pt / best.pt mỗi epoch + resume
# ---------------------------------------------------------------------------

def _toy_train_state(seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Linear(4, 2)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2)
    return net, opt


def _one_step(net, opt, gen):
    x = torch.randn(8, 4, generator=gen)
    opt.zero_grad()
    net(x).pow(2).mean().backward()
    opt.step()


def test_checkpoint_save_ghi_last_va_chi_chep_best_khi_cai_thien(tmp_path):
    from utils.checkpoint import CheckpointManager

    m = CheckpointManager(str(tmp_path))
    m.save({"epoch": 0, "v": 1}, is_best=True)
    m.save({"epoch": 1, "v": 2}, is_best=False)
    assert m.load_last()["epoch"] == 1
    best = torch.load(m.best_path, weights_only=False)
    assert best["epoch"] == 0                      # best KHÔNG bị ghi đè bởi epoch tệ hơn
    assert not list(tmp_path.glob("*.tmp"))        # không để lại file tạm


def test_checkpoint_ghi_nguyen_tu_giu_file_cu_khi_ghi_hong(tmp_path, monkeypatch):
    """Bị ngắt giữa lúc ghi thì last.pt CŨ phải còn nguyên — đó là lý do tồn tại."""
    import pytest

    from utils.checkpoint import CheckpointManager

    m = CheckpointManager(str(tmp_path))
    m.save({"epoch": 5}, is_best=False)

    def hong(obj, path):
        with open(path, "wb") as f:
            f.write(b"dang ghi do")
        raise KeyboardInterrupt                    # giả lập bị kill giữa chừng
    monkeypatch.setattr(torch, "save", hong)
    with pytest.raises(KeyboardInterrupt):
        m.save({"epoch": 6}, is_best=False)
    monkeypatch.undo()
    assert m.load_last()["epoch"] == 5


def test_resume_cho_ket_qua_trung_khit_train_lien_tuc(tmp_path):
    """Train 4 bước liền == train 2 bước, lưu, nạp vào model MỚI, train tiếp 2 bước.
    Kiểm cả optimizer (moment của AdamW) lẫn RNG — thiếu cái nào cũng lệch."""
    from utils.checkpoint import CheckpointManager, rng_state, set_rng_state

    net_a, opt_a = _toy_train_state()
    gen_a = torch.Generator().manual_seed(1)
    for _ in range(4):
        _one_step(net_a, opt_a, gen_a)

    net_b, opt_b = _toy_train_state()
    gen_b = torch.Generator().manual_seed(1)
    for _ in range(2):
        _one_step(net_b, opt_b, gen_b)
    m = CheckpointManager(str(tmp_path))
    m.save({"model": net_b.state_dict(), "optimizer": opt_b.state_dict(),
            "rng": rng_state(gen_b), "epoch": 1}, is_best=False)

    net_c, opt_c = _toy_train_state(seed=123)      # khởi tạo KHÁC, phải bị ghi đè hết
    gen_c = torch.Generator().manual_seed(999)
    st = m.load_last()
    net_c.load_state_dict(st["model"])
    opt_c.load_state_dict(st["optimizer"])
    set_rng_state(st["rng"], gen_c)
    for _ in range(2):
        _one_step(net_c, opt_c, gen_c)

    for pa, pc in zip(net_a.parameters(), net_c.parameters()):
        assert torch.equal(pa, pc)


def test_config_mismatch_chan_doi_kien_truc_cho_phep_doi_batch():
    from utils.checkpoint import CheckpointManager

    cu = {"model": {"n_layer": 6}, "diffusion": {}, "matcher": {},
          "data": {"image_size": 1024, "num_workers": 8}, "training": {"batch_size": 2}}
    doi_batch = {**cu, "training": {"batch_size": 6},
                 "data": {"image_size": 1024, "num_workers": 4}}
    doi_anh = {**cu, "data": {"image_size": 512, "num_workers": 8}}

    assert CheckpointManager.config_mismatch(cu, doi_batch) == ([], ["training"])
    assert CheckpointManager.config_mismatch(cu, doi_anh)[0] == ["data"]


# ---------------------------------------------------------------------------
# Rà pipeline 2026-09-23: t mỗi ảnh, lật ảnh, grad monitor
# ---------------------------------------------------------------------------

def test_build_inputs_boc_t_rieng_cho_tung_anh():
    """DiffusionDet bốc `t` cho TỪNG ảnh. Bản cũ bốc một `t` cho cả batch."""
    from types import SimpleNamespace

    from models.detector import CELocDetector
    from utils.diffusion_math import cosine_alphas_cumprod

    fake = SimpleNamespace(alphas_cumprod=cosine_alphas_cumprod(1000), num_timesteps=1000,
                           snr_scale=2.0)
    gt = [torch.tensor([[0.5, 0.5, 0.1, 0.1]])] * 16
    g = torch.Generator().manual_seed(0)
    x_t, t, _ = CELocDetector.build_inputs(fake, gt, 30, [1.0] * 16, generator=g)
    assert x_t.shape == (16, 30, 4) and t.shape == (16,) and t.dtype == torch.long
    assert len(set(t.tolist())) > 8, t.tolist()          # 16 ảnh, gần như chắc khác nhau


def test_train_truyen_flip_prob_tu_config():
    """Hồi quy: train.py từng gọi build_dataset(cfg, "train") KHÔNG có flip_prob, nên
    augmentation tắt âm thầm dù config ghi 0.5."""
    src = open("train.py").read()
    call = src[src.index('ds_tr = build_dataset('):]
    call = call[:call.index(")\n")]
    assert "flip_prob=" in call and 'cfg["data"]' in call, call


class _CoinDS(torch.utils.data.Dataset):
    """Dataset tung đồng xu bằng Generator RIÊNG — giống CE130Detection."""

    def __init__(self):
        import numpy as np
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return 64

    def __getitem__(self, i):
        return float(self.rng.random())


class _Wrap(torch.utils.data.Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i]


def test_seed_worker_lam_cac_worker_tung_dong_xu_khac_nhau():
    """Không seed lại thì 2 worker fork cùng trạng thái RNG và trả CÙNG dãy số."""
    from train import seed_worker

    def draws(init):
        ld = torch.utils.data.DataLoader(_Wrap(_CoinDS()), batch_size=8, num_workers=2,
                                         worker_init_fn=init)
        b = [x.tolist() for x in ld]
        return b[0], b[1]                     # batch 0 từ worker 0, batch 1 từ worker 1

    w0, w1 = draws(None)
    assert w0 == w1                           # lỗi được tái hiện: hai worker trùng dãy
    w0, w1 = draws(seed_worker)
    assert w0 != w1


def test_grad_monitor_nhom_va_ti_phan():
    from utils.grad_monitor import GradMonitor, group_of

    assert group_of("decoder.layers.0.box_delta.weight") == "box_delta[0]"
    assert group_of("decoder.layers.5.roi.proj_point.weight") == "roi.proj_point"
    assert group_of("decoder.layers.2.roi.out.bias") == "roi.out"
    assert group_of("decoder.layers.3.cross_attn.in_proj_weight") == "cross_attn"
    assert group_of("encoder.proj_patch.weight") == "proj_patch"
    assert group_of("decoder.score_head.4.weight") == "score_head"
    assert group_of("decoder.cond_pos_emb") == "embed/khác"

    net = torch.nn.Module()
    net.encoder = torch.nn.Module()
    net.encoder.proj_patch = torch.nn.Linear(4, 4)
    net.encoder.proj_text = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)
    (net.encoder.proj_patch(x * 100).sum() + net.encoder.proj_text(x).sum()).backward()
    mon = GradMonitor(net, every=1)
    mon.maybe_record(0)
    s = mon.summary()
    assert list(s)[0] == "proj_patch"          # đầu vào to x100 -> gradient áp đảo
    sh = GradMonitor.share(s)
    assert abs(sum(sh.values()) - 1.0) < 1e-6
    assert mon.summary() == {}                 # summary xoá mẫu để đo epoch sau


# ---------------------------------------------------------------------------
# eval.py chạy TRỌN LUỒNG — trước 2026-09-25 không test nào làm vậy và lần eval đầu
# tiên vỡ ngay (torch.argsort trên numpy, cls=None bị index, evaluate nhận dict).
# ---------------------------------------------------------------------------

class _FakeSampler(torch.nn.Module):
    """ddim_sample trả về đúng box/score định sẵn cho từng ảnh."""

    def __init__(self, per_image):
        super().__init__()
        self.per_image = per_image                     # list[(boxes [N,4], logits [N])]
        self.i = 0

    def ddim_sample(self, n, valid_h=None, generator=None, return_all_layers=False,
                    **kw):
        B = kw["patch_raw"].shape[0]
        items = self.per_image[self.i:self.i + B]
        self.i += B
        boxes = torch.stack([b for b, _ in items])
        logits = torch.stack([l for _, l in items])
        # 2 "tầng": tầng đầu lệch hẳn, tầng cuối là dự đoán thật
        return [(boxes + 0.3, logits), (boxes, logits)]


class _FakeLoader:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = [None] * sum(len(b["boxes"]) for b in batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _batch(gts, ids):
    B = len(gts)
    return {"boxes": [torch.tensor(g, dtype=torch.float32) for g in gts],
            "labels": [torch.zeros(len(g), dtype=torch.long) for g in gts],
            "text": ["x"] * B, "valid_h": [1.0] * B, "image_id": ids,
            "patch_raw": torch.zeros(B, 4, 8), "text_raw": torch.zeros(B, 1, 8)}


def test_eval_tron_luong_du_doan_hoan_hao():
    import eval as ev

    gt0 = [[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]]
    gt1 = [[0.5, 0.3, 0.2, 0.1]]
    far = [0.9, 0.9, 0.05, 0.05]
    per = [(torch.tensor(gt0 + [far, far]), torch.tensor([5.0, 5.0, -5.0, -5.0])),
           (torch.tensor(gt1 + [far, far, far]), torch.tensor([5.0, -5.0, -5.0, -5.0]))]
    loader = _FakeLoader([_batch([gt0, gt1], ["a", "b"])])
    recs, rec_layer = ev.predict(_FakeSampler(per), loader, 4, torch.device("cpu"),
                                 top_k=100, nms_thr=0.5)
    res = ev.score_records(recs)

    assert res["AP50"] == pytest.approx(1.0)
    assert res["oracle_recall"] == pytest.approx(1.0)
    assert res["score_AUC"] == pytest.approx(1.0)
    assert rec_layer[-1] == pytest.approx(1.0) and rec_layer[0] < 1.0
    # NMS gộp các box `far` trùng nhau: 4 box -> còn 3 ở ảnh 0 (2 GT + 1 far)
    assert len(recs[0]["keep"]) == 3


def test_eval_ap_tinh_tay():
    """2 GT; dự đoán theo score: TP 0.9, FP 0.8, TP 0.7.
    recall [.5 .5 1], precision [1 .5 .667] -> AP = .5*1 + .5*.667 = 0,8333."""
    from utils.metrics_np import evaluate

    gt = np.array([[0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]])
    boxes = np.array([[0.0, 0.0, 0.1, 0.1], [0.8, 0.8, 0.9, 0.9], [0.5, 0.5, 0.6, 0.6]])
    r = evaluate([(boxes, np.array([0.9, 0.8, 0.7]), gt)], 0.5)
    assert r["AP"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert r["recall"] == pytest.approx(1.0)


def test_postprocess_top_k_truoc_roi_nms():
    import eval as ev

    b = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]])
    s = np.array([0.9, 0.8, 0.1])
    assert ev.postprocess(b, s, top_k=2).tolist() == [0, 1]
    assert ev.postprocess(b, s, top_k=2, nms_thr=0.5).tolist() == [0]   # trùng bị gộp
    assert ev.postprocess(b, s, top_k=3, nms_thr=0.5).tolist() == [0, 2]


def test_scores_and_classes_tra_numpy():
    import eval as ev

    s, c = ev.scores_and_classes(torch.tensor([0.0, 2.0]))
    assert isinstance(s, np.ndarray) and c is None
    s, c = ev.scores_and_classes(torch.tensor([[0.0, 3.0], [1.0, -1.0]]))
    assert c.tolist() == [1, 0]


def test_oracle_score_tach_loi_xep_hang_khoi_loi_box():
    """Box đúng nhưng score đảo ngược: AP thật thấp, TRẦN phải về 1.
    Box sai hoàn toàn: trần cũng 0 — sửa score vô ích."""
    import eval as ev

    gt = np.array([[0.2, 0.2, 0.1, 0.1], [0.6, 0.6, 0.1, 0.1]])
    far = [0.9, 0.9, 0.05, 0.05]
    boxes = np.array(gt.tolist() + [far, far])
    bad_sc = np.array([0.1, 0.2, 0.9, 0.8])            # box sai lại điểm cao
    rec = {"image_id": "a", "boxes": boxes, "scores": bad_sc, "classes": None,
           "gt": gt, "keep": ev.postprocess(boxes, bad_sc, 100, None)}

    that = ev.score_records([rec])
    tran = ev.score_records(ev.with_oracle_scores([rec], 100, None))
    # FP, FP, TP, TP -> precision đơn điệu 0,5 ở cả hai mức recall -> AP = 0,5
    assert that["AP50"] == pytest.approx(0.5)
    assert tran["AP50"] == pytest.approx(1.0)

    orc = ev.with_oracle_scores([rec], 100, None)[0]
    assert np.array_equal(orc["boxes"], boxes)          # box giữ nguyên từng bit
    assert orc["scores"][:2] == pytest.approx([1.0, 1.0]) and orc["scores"][2] == 0.0

    sai = {**rec, "boxes": np.array([far] * 4)}
    assert ev.score_records(ev.with_oracle_scores([sai], 100, None))["AP50"] == 0.0
