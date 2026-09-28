"""Giai đoạn 1: affinity từ attention, Markov, p-Laplacian, prompt, nguồn model, SDPA."""

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

from diffuse2seg.config.base import Diffu2SegConfig
from diffuse2seg.d2s.affinity import blend_timesteps, change_temperature, to_affinity
from diffuse2seg.d2s.markov import matrix_ipf, markov_map_from_prompt
from diffuse2seg.d2s.plaplacian import compute_g, plaplacian_propagate
from diffuse2seg.d2s.prompts import build_prompt_grid, cells_to_canvas_xy, f0_onehot

try:                                    # cần diffusers — máy local có thể chưa cài
    from diffuse2seg.d2s.attention import AttnProcessor2_0Wrapper
except ImportError:
    AttnProcessor2_0Wrapper = None
needs_diffusers = pytest.mark.skipif(AttnProcessor2_0Wrapper is None, reason="cần diffusers")

PROJECT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "diffuse2seg")

# ============================================================================
# từ test_affinity.py
# ============================================================================

H = W = 6


N = H * W


def _fake_attn(seed=0):
    """(h, w, h, w) with the last two axes summing to 1, like SD2's output."""
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(H, W, H, W, generator=g, dtype=torch.float32) + 1e-4
    return a / a.reshape(H, W, -1).sum(dim=2)[:, :, None, None]


def test_reshape_gives_row_stochastic_matrix():
    """The property everything downstream assumes.

    Normalising the last two axes of (h,w,h,w) is the same as making each row of
    the flattened (N,N) sum to 1 -- so the tensor is already a transition matrix
    and needs no extra normalisation to be a graph.
    """
    A = to_affinity(_fake_attn(), tau_att=None)
    assert A.shape == (N, N)
    assert torch.allclose(A.sum(dim=1), torch.ones(N), atol=1e-6)
    assert (A >= 0).all()


def test_row_stochastic_after_temperature():
    """Temperature re-runs a softmax, so rows must be renormalised, and are."""
    A = to_affinity(_fake_attn(1), tau_att=0.55)
    assert torch.allclose(A.sum(dim=1), torch.ones(N), atol=1e-6)


def test_row_stochastic_after_symmetrize():
    A = to_affinity(_fake_attn(2), tau_att=0.55, symmetrize=True)
    assert torch.allclose(A.sum(dim=1), torch.ones(N), atol=1e-6)


def test_temperature_below_one_sharpens():
    """tau < 1 concentrates mass: the largest entry of a row grows."""
    attn = _fake_attn(3)
    plain = to_affinity(attn, tau_att=None)
    sharp = to_affinity(attn, tau_att=0.55)
    assert sharp.max(dim=1).values.mean() > plain.max(dim=1).values.mean()


def test_temperature_above_one_flattens():
    attn = _fake_attn(4)
    plain = to_affinity(attn, tau_att=None)
    flat = to_affinity(attn, tau_att=2.0)
    assert flat.max(dim=1).values.mean() < plain.max(dim=1).values.mean()


def test_zero_entries_do_not_produce_nan():
    """SD2 attention contains exact zeros and log(0) = -inf.

    NEGATIVE CONTROL for the clamp inside change_temperature.
    """
    x = torch.zeros(3, 4)
    x[:, 0] = 1.0
    out = change_temperature(x, 0.55)
    assert torch.isfinite(out).all()
    assert torch.allclose(out.sum(dim=1), torch.ones(3), atol=1e-6)


def test_blend_is_a_weighted_mean():
    a, b = _fake_attn(5), _fake_attn(6)
    out = blend_timesteps([a, b], [0.85, 0.15])
    assert torch.allclose(out.reshape(H, W, -1).sum(dim=2), torch.ones(H, W), atol=1e-6)

    same = blend_timesteps([a], [1.0])
    assert torch.allclose(same, a, atol=1e-6)


def test_blend_rejects_weights_that_do_not_sum_to_one():
    try:
        blend_timesteps([_fake_attn(7), _fake_attn(8)], [0.5, 0.2])
    except AssertionError:
        return
    raise AssertionError("weights not summing to 1 must be rejected")


def test_output_is_fp32():
    """p-Laplacian rejects fp16: g**(p-2) has a negative exponent."""
    assert to_affinity(_fake_attn(9).half(), tau_att=0.55).dtype == torch.float32


def test_rejects_non_square_tensor():
    try:
        to_affinity(torch.rand(4, 5, 4, 4))
    except AssertionError:
        return
    raise AssertionError("mismatched (h,w) pairs must be rejected")


def main():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_markov.py
# ============================================================================

torch = pytest.importorskip("torch")


# Ma trận 2x2 đã tính TAY ở phiên thiết kế; mọi số dưới đây là kết quả tay.
A_2x2 = torch.tensor([
    [0.720242, 0.015886, 0.015886, 0.247986],
    [0.015886, 0.645866, 0.322262, 0.015886],
    [0.015886, 0.322262, 0.645866, 0.015886],
    [0.247986, 0.015886, 0.015886, 0.720242]], dtype=torch.float64)


def test_markov_map_matches_hand_computation():
    """Lưới 2x2: ô 0 và ô 3 CÙNG VẬT (chéo nhau), ô 1,2 là nền (kề ô 0)."""
    m, _ = markov_map_from_prompt(A_2x2, 0, tau=0.3, max_iterations=1000)
    assert abs(float(m[0]) - 0.0) < 1e-9
    assert abs(float(m[3]) - 0.9367) < 1e-3, "ô cùng vật phải đến ở ~0.94"
    assert abs(float(m[1]) - 9.4472) < 1e-3, "nền phải đến ở ~9.45"
    assert abs(float(m[1]) - float(m[2])) < 1e-12, "hai nền đối xứng"


def test_semantic_beats_spatial():
    """[NEGATIVE CONTROL] ô CHÉO cùng vật phải đến TRƯỚC ô KỀ là nền.

    Đây là toàn bộ điểm của phương pháp: A là đồ thị NGỮ NGHĨA, không phải
    lưới không gian. Nếu test này đỏ, mô hình "lan sang ô kề" đã lẻn vào.
    """
    m, _ = markov_map_from_prompt(A_2x2, 0, tau=0.3, max_iterations=1000)
    assert float(m[3]) < float(m[1]), "ô chéo cùng vật phải đến trước nền kề"


def test_ipf_makes_doubly_stochastic():
    """IPF là thứ làm p_inf = uniform cho MỌI ảnh (paper §3.3)."""
    g = torch.Generator().manual_seed(0)
    A = torch.rand(64, 64, generator=g, dtype=torch.float64) ** 3
    A = A / A.sum(dim=1, keepdim=True)          # right-stochastic
    assert (A.sum(0) - 1).abs().max() > 1e-3, "cột LẼ RA chưa bằng 1"

    M = matrix_ipf(A, iterations=200)
    assert (M.sum(1) - 1).abs().max() < 1e-9
    assert (M.sum(0) - 1).abs().max() < 1e-6, "IPF 200 vòng phải doubly stochastic"


def test_ipf_15_iterations_is_not_enough():
    """[NEGATIVE CONTROL] mặc định 15 của M2N2 KHÔNG đủ — họ luôn truyền 200.

    ⚠️ Chỉ thấy được trên ma trận ĐÃ QUA NHIỆT ĐỘ — tức đúng thứ đường chạy
    thật đưa vào IPF. Trên ma trận ngẫu nhiên "dễ", IPF hội tụ ngay ở vòng 15
    (đo: 4,4e-16 vs 2,2e-16) và test sẽ xanh giả. Đây là lý do phải dựng đầu
    vào giống thật thay vì lấy rand() cho tiện.
    """
    g = torch.Generator().manual_seed(0)
    A = torch.softmax(torch.randn(128, 128, generator=g, dtype=torch.float64) * 2.0,
                      dim=-1)
    A = torch.softmax(torch.log(A.clamp_min(1e-12)) / 0.65, dim=-1)   # T = 0.65

    d15 = (matrix_ipf(A, 15).sum(0) - 1).abs().max()
    d200 = (matrix_ipf(A, 200).sum(0) - 1).abs().max()
    assert d15 > 1e-4, f"15 vòng phải còn lệch đáng kể, đo {d15:.2e}"
    assert d200 < 1e-9, f"200 vòng phải hội tụ, đo {d200:.2e}"


def test_without_max_division_nothing_crosses_threshold():
    """[NEGATIVE CONTROL] vì sao Eq.6 phải chia cho max.

    A doubly stochastic => p_t -> uniform 1/N. Với N=64 thì uniform = 0.0156,
    không bao giờ vượt tau=0.3. Bỏ phép chia là thuật toán chết câm lặng:
    không lỗi, không NaN, chỉ là KHÔNG ô nào vượt ngưỡng.
    """
    g = torch.Generator().manual_seed(1)
    A = torch.rand(64, 64, generator=g, dtype=torch.float64) ** 3
    A = matrix_ipf(A / A.sum(dim=1, keepdim=True), 200)

    p = torch.zeros(64, dtype=torch.float64); p[0] = 1.0
    for _ in range(200):
        p = p @ A                      # chuỗi THUẦN, không chia max
    assert p.max() < 0.3, "chuỗi thuần không bao giờ vượt tau"
    assert abs(float(p.sum()) - 1.0) < 1e-9, "nhưng vẫn là phân bố xác suất"

    m, _ = markov_map_from_prompt(A, 0, tau=0.3, max_iterations=200)
    assert (m < 200).sum() > 0, "bản CÓ chia max thì có ô vượt ngưỡng"


def test_snapshots_are_returned_at_requested_steps():
    """Mỗi snapshot là (p_t, m_t): trạng thái chuỗi VÀ Markov-map từng phần."""
    m, snaps = markov_map_from_prompt(A_2x2, 0, tau=0.3, max_iterations=50,
                                      snapshot_steps=[0, 1, 5])
    assert set(snaps) == {0, 1, 5}
    p0, m0 = snaps[0]
    assert abs(float(p0[0]) - 1.0) < 1e-12
    for t, (p, _) in snaps.items():
        assert abs(float(p.max()) - 1.0) < 1e-9, "đã chia max nên đỉnh = 1"


def test_partial_markov_map_keeps_unreached_at_max():
    """Ô CHƯA tới phải giữ max_iterations, KHÔNG mang giá trị bước hiện tại.

    ⚠️ Đây là chỗ đã vẽ ra hình hỏng (2026-09-18). Bản đầu gán (i+1) cho ô
    chưa tới; ở t nhỏ thì giá trị đó (1..50) rơi TRONG dải màu của vùng đã
    tới, nên nền bị tô gần TRẮNG và nuốt hết vật. Giữ max_iterations thì nền
    luôn nằm ngoài dải -> luôn đen, và mọi panel dùng chung được một thang màu.

    Đây là tính chất HIỂN THỊ, nhưng nó quyết định hình có đọc được hay không,
    nên phải có test — nhìn hình thì cái nào cũng "hợp lý".
    """
    m, snaps = markov_map_from_prompt(A_2x2, 0, tau=0.3, max_iterations=50,
                                      snapshot_steps=[1, 2, 5, 20])
    ts = sorted(snaps)

    # ô đã tới: ĐÓNG BĂNG ngay khi vượt tau
    for t in ts:
        _, m_t = snaps[t]
        assert abs(float(m_t[0]) - 0.0) < 1e-12, "seed luôn 0"
        assert abs(float(m_t[3]) - 0.9367) < 1e-3, "ô cùng vật chốt từ t=1"

    # ô nền (đến ở ~9.45): giữ max_iterations cho tới khi thực sự vượt
    for t in (1, 2, 5):
        _, m_t = snaps[t]
        assert float(m_t[1]) == 50.0, \
            f"t={t}: ô chưa tới phải là max_iterations, đo {float(m_t[1])}"
    _, m20 = snaps[20]
    assert abs(float(m20[1]) - float(m[1])) < 1e-9, \
        "sau khi vượt tau thì bằng m thật"

    # ĐƠN ĐIỆU GIẢM: vùng đã tới chỉ lớn dần, không bao giờ co lại
    for a, b in zip(ts, ts[1:]):
        _, ma = snaps[a]
        _, mb = snaps[b]
        assert bool((mb <= ma + 1e-12).all()), \
            f"m_t phải giảm dần (vùng trắng lan ra), vỡ giữa t={a} và t={b}"



# ============================================================================
# từ test_plaplacian.py
# ============================================================================

TOL = 1e-12          # fp64: the expansion is exact algebra, not an approximation


def _random_affinity(n, seed=0, dtype=torch.float64):
    """Row-stochastic (N, N), like softmax(QK^T) coming out of SD2."""
    g = torch.Generator().manual_seed(seed)
    A = torch.rand(n, n, generator=g, dtype=dtype) + 1e-3
    return A / A.sum(dim=1, keepdim=True)


def _random_f(k, n, seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(k, n, generator=g, dtype=dtype)


def _naive_g(A, f):
    """g_i = sqrt(sum_j A_ij (f_j - f_i)^2), written out literally."""
    K, N = f.shape
    out = torch.zeros(K, N, dtype=f.dtype)
    for k in range(K):
        for i in range(N):
            s = 0.0
            for j in range(N):
                s += A[i, j].item() * (f[k, j].item() - f[k, i].item()) ** 2
            out[k, i] = s ** 0.5
    return out


def test_g_expansion_matches_naive():
    """(A f^2) - 2 f (A f) + f^2 (A 1) == the triple loop."""
    A, f = _random_affinity(12), _random_f(3, 12)
    fast = compute_g(A, f, eps=0.0)
    slow = _naive_g(A, f)
    assert (fast - slow).abs().max().item() < TOL


def test_gamma_expansion_matches_naive():
    """The SECOND expansion -- the one that hides inside a plain-looking average.

    sum_j gamma_ij f_j  and  sum_j gamma_ij, with gamma_ij = A_ij (gp_i + gp_j).
    Easy to miss because the update reads like a weighted mean; just as fatal
    in memory terms, and just as silent when wrong.
    """
    A, f = _random_affinity(10, seed=4), _random_f(2, 10, seed=5)
    p, lam = 1.6, 1e-5

    g = compute_g(A, f, eps=1e-8)
    gp = g.pow(p - 2.0)
    row_sum = A.sum(dim=1)

    fast_num = lam * f + gp * (f @ A.T) + (gp * f) @ A.T
    fast_den = lam + gp * row_sum + gp @ A.T

    K, N = f.shape
    slow_num = torch.zeros(K, N, dtype=f.dtype)
    slow_den = torch.zeros(K, N, dtype=f.dtype)
    for k in range(K):
        for i in range(N):
            acc_n, acc_d = 0.0, 0.0
            for j in range(N):
                gamma = A[i, j].item() * (gp[k, i].item() + gp[k, j].item())
                acc_n += gamma * f[k, j].item()
                acc_d += gamma
            slow_num[k, i] = lam * f[k, i].item() + acc_n
            slow_den[k, i] = lam + acc_d

    assert (fast_num - slow_num).abs().max().item() < TOL
    assert (fast_den - slow_den).abs().max().item() < TOL


def test_p2_collapses_to_linear_diffusion():
    """At p=2, gp = g^0 = 1 and gamma = 2A, so one step is a one-liner.

    NEGATIVE CONTROL: writing `gp = g.pow(p)` instead of `g.pow(p - 2)` -- an
    easy slip -- breaks this, because g^2 != 1.
    """
    A, f0 = _random_affinity(9, seed=7, dtype=torch.float32), None
    f0 = torch.zeros(2, 9, dtype=torch.float32)
    f0[0, 0] = 1.0
    f0[1, 5] = 1.0
    lam = 1e-3

    f, _, _ = plaplacian_propagate(A, f0, p=2.0, lam=lam, tau_prop=0.0, max_iter=1)
    row_sum = A.sum(dim=1)
    expected = (lam * f0 + 2.0 * (f0 @ A.T)) / (lam + 2.0 * row_sum)
    assert (f - expected).abs().max().item() < 1e-6


def test_flat_field_does_not_produce_nan():
    """g == 0 everywhere; with p-2 = -0.4 the unclamped form is inf -> NaN.

    NEGATIVE CONTROL included: with g_eps=0 the result MUST be non-finite. If
    this half ever starts passing, the clamp has been silently neutralised and
    the guard above is no longer proving anything.
    """
    N = 8
    A = _random_affinity(N, seed=11, dtype=torch.float32)
    f_flat = torch.full((1, N), 0.3, dtype=torch.float32)

    ok = compute_g(A, f_flat, eps=1e-8)
    assert torch.isfinite(ok).all() and (ok > 0).all()
    assert torch.isfinite(ok.pow(1.6 - 2.0)).all()

    bad = compute_g(A, f_flat, eps=0.0)
    assert not torch.isfinite(bad.pow(1.6 - 2.0)).all(), \
        "clamp guard is no longer being tested"


def test_propagate_stays_finite_on_flat_seed():
    N = 16
    A = _random_affinity(N, seed=13, dtype=torch.float32)
    f0 = torch.zeros(3, N, dtype=torch.float32)
    f0[:, 0] = 1.0
    f, n_iter, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e-5,
                                        tau_prop=1e-4, max_iter=50)
    assert torch.isfinite(f).all()
    assert 1 <= n_iter <= 50


def test_residual_decreases_and_converges():
    A = _random_affinity(20, seed=17, dtype=torch.float32)
    f0 = torch.zeros(2, 20, dtype=torch.float32)
    f0[0, 3] = 1.0
    f0[1, 11] = 1.0
    f, n_iter, res = plaplacian_propagate(A, f0, p=1.6, lam=1e-3, tau_prop=1e-10,
                                          max_iter=300, return_history=True)
    assert n_iter < 300, "should converge well inside the cap on a dense graph"
    assert res[-1] <= res[0]
    assert res[-1] <= 1e-10


def test_large_lam_pins_solution_to_seed():
    """lam -> large means the anchor dominates: f ~ f0. Checks lam's position."""
    N = 12
    A = _random_affinity(N, seed=19, dtype=torch.float32)
    f0 = torch.zeros(1, N, dtype=torch.float32)
    f0[0, 4] = 1.0
    f, _, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e6, tau_prop=1e-12, max_iter=50)
    assert (f - f0).abs().max().item() < 1e-3


def test_output_is_bounded_by_seed_range():
    """A row-stochastic, f0 in [0,1] -> f stays in [0,1] (maximum principle)."""
    N = 14
    A = _random_affinity(N, seed=23, dtype=torch.float32)
    f0 = torch.zeros(4, N, dtype=torch.float32)
    for k in range(4):
        f0[k, k * 3] = 1.0
    f, _, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e-4, tau_prop=1e-8, max_iter=100)
    assert f.min().item() >= -1e-6
    assert f.max().item() <= 1.0 + 1e-6


def test_seed_keeps_the_maximum():
    """The anchored cell must stay the strongest -- otherwise thresholding a map
    would not even select the object the prompt was placed on."""
    N = 24
    A = _random_affinity(N, seed=29, dtype=torch.float32)
    seed_idx = 7
    f0 = torch.zeros(1, N, dtype=torch.float32)
    f0[0, seed_idx] = 1.0
    f, _, _ = plaplacian_propagate(A, f0, p=1.6, lam=1e-2, tau_prop=1e-10, max_iter=200)
    assert int(f[0].argmax().item()) == seed_idx


def test_rejects_fp16():
    """fp16 has eps ~6e-8; g**(-0.4) on such values overflows."""
    N = 8
    A = _random_affinity(N, dtype=torch.float32).half()
    f0 = torch.zeros(1, N, dtype=torch.float16)
    f0[0, 0] = 1.0
    try:
        plaplacian_propagate(A, f0, p=1.6, lam=1e-5, tau_prop=1e-4, max_iter=5)
    except AssertionError:
        return
    raise AssertionError("fp16 input must be rejected")


def main_plaplacian():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_prompts.py
# ============================================================================

R = 64                 # latent grid for a 512 canvas


CANVAS = 512


CELL_PX = CANVAS / R   # 8 px


def test_f0_is_exactly_onehot():
    """Each row sums to exactly 1.0 -- lam*f0 must mean the same for every prompt.

    Guards against drifting back to M2N2's bilinear-and-normalise-by-max seed,
    whose rows sum to 1.0-4.0 and would make the anchor strength per-prompt.
    """
    cells = build_prompt_grid(R, 3)
    f0 = f0_onehot(cells, R)

    assert f0.shape == (len(cells), R * R)
    assert torch.all(f0.sum(dim=1) == 1.0)
    assert torch.all((f0 == 0.0) | (f0 == 1.0)), "seeds must be hard one-hot"
    assert int((f0 > 0).sum().item()) == len(cells)


def test_f0_lands_on_the_requested_cell():
    cells = np.array([[0, 0], [5, 9], [R - 1, R - 1]])
    f0 = f0_onehot(cells, R)
    for k, (r, c) in enumerate(cells):
        assert f0[k, r * R + c].item() == 1.0


def test_padding_cells_are_dropped():
    """valid_h = 0.71 (aspect 1.41, the CE-130 median) -> nothing below row 45."""
    valid_h = 384.0 / 541.0                     # ~0.71, a real CE-130 shape
    cells = build_prompt_grid(R, 3, valid_h=valid_h)
    assert len(cells) > 0
    assert cells[:, 0].max() < valid_h * R, "a prompt was seeded in the padding"


def test_square_image_keeps_full_grid():
    """valid_h = 1.0 (a 384x384 image) -> no rows removed."""
    full = build_prompt_grid(R, 3, valid_h=1.0)
    cropped = build_prompt_grid(R, 3, valid_h=384.0 / 541.0)
    assert len(full) > len(cropped)
    n_cols = len(np.unique(full[:, 1]))
    assert len(full) == len(np.unique(full[:, 0])) * n_cols


def test_extreme_aspect_still_returns_prompts():
    """W/H = 4.99 is the measured CE-130 maximum: valid_h ~ 0.20, still usable."""
    cells = build_prompt_grid(R, 3, valid_h=384.0 / 1918.0)
    assert len(cells) > 0


def _coverage(box_cells, stride):
    """Fraction of synthetic boxes of side `box_cells` containing >=1 prompt."""
    cells = build_prompt_grid(R, stride, valid_h=1.0)
    xy = cells_to_canvas_xy(cells, R, CANVAS)
    side = box_cells * CELL_PX

    rng = np.random.default_rng(0)
    hits = 0
    n = 400
    for _ in range(n):
        cx, cy = rng.uniform(side / 2, CANVAS - side / 2, size=2)
        inside = ((np.abs(xy[:, 0] - cx) <= side / 2) &
                  (np.abs(xy[:, 1] - cy) <= side / 2))
        hits += int(inside.any())
    return hits / n


def test_grid_coverage_matches_measured_numbers():
    """Median CE-130 box short side is 4.65 cells; stride 3 measured 96.6 %."""
    assert _coverage(4.65, stride=3) >= 0.90


def test_stride6_misses_more_than_stride3():
    """Why we deviate from the paper.

    At the median SMALLEST box per image (2.24 cells), the paper's stride of 6
    steps over objects that stride 3 catches.
    """
    small = 2.24
    assert _coverage(small, stride=6) < _coverage(small, stride=3)


def test_cells_to_canvas_xy_is_cell_centre():
    xy = cells_to_canvas_xy(np.array([[0, 0], [1, 2]]), R, CANVAS)
    assert np.allclose(xy[0], [4.0, 4.0])          # centre of the first cell
    assert np.allclose(xy[1], [2 * 8 + 4, 1 * 8 + 4])


def test_valid_w_drops_right_hand_padding():
    """A PORTRAIT image pads on the RIGHT, and those columns must lose seeds.

    CE-130 never exercises this (every image is 384 tall and at least that
    wide, so W >= H and padding is always at the bottom), which is exactly why
    it needs a test: the bug would only ever appear on COCO, where 427x640 is
    common, and it would appear as one enormous mask rather than as a crash.
    """
    full = build_prompt_grid(R, 3, valid_h=1.0, valid_w=1.0)
    # 586x640 -> valid_w = 0.915; the rightmost ~8.5 % of columns are grey.
    narrow = build_prompt_grid(R, 3, valid_h=1.0, valid_w=0.915)
    assert len(narrow) < len(full), "right-hand padding must remove seeds"
    assert narrow[:, 1].max() + 1.0 <= 0.915 * R + 1e-9, \
        "a surviving seed still sits in the padding"
    # And rows must be untouched: only the width is padded here.
    assert set(narrow[:, 0].tolist()) == set(full[:, 0].tolist())


def test_valid_w_default_leaves_ce130_unchanged():
    """[NEGATIVE CONTROL] Adding valid_w must not move a single CE-130 seed.

    The parameter defaults to 1.0 precisely so that every number already
    measured on CE-130 (96.6 % coverage at stride 3) still describes the code
    that runs. If this fails, the COCO change silently altered CE-130 results.
    """
    for vh in (1.0, 0.9412, 0.6652, 0.2):
        a = build_prompt_grid(R, 3, valid_h=vh)
        b = build_prompt_grid(R, 3, valid_h=vh, valid_w=1.0)
        assert np.array_equal(a, b), f"valid_w=1.0 changed the grid at valid_h={vh}"


def test_stride_must_fit_the_grid():
    try:
        build_prompt_grid(R, R + 1)
    except AssertionError:
        return
    raise AssertionError("an out-of-range stride must be rejected")


def main_prompts():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_model_source.py
# ============================================================================

def test_falls_back_to_hub_id_when_dir_missing():
    cfg = Diffu2SegConfig(local_model_dir="/khong/ton/tai/o/dau/ca")
    assert cfg.model_source == cfg.hf_model_id


def test_uses_local_dir_when_it_has_model_index():
    with tempfile.TemporaryDirectory() as d:
        with open(os.path.join(d, "model_index.json"), "w") as f:
            json.dump({"_class_name": "StableDiffusionPipeline"}, f)
        cfg = Diffu2SegConfig(local_model_dir=d)
        assert cfg.model_source == d


def test_empty_dir_is_not_accepted():
    """Thư mục rỗng do giải nén hỏng phải bị từ chối NGAY.

    Chỉ kiểm os.path.isdir là chưa đủ: một thư mục rỗng sẽ lọt qua rồi mới chết
    ở from_pretrained với thông báo khó hiểu, sau khi đã tốn thời gian khởi động.
    model_index.json là file from_pretrained đọc đầu tiên nên nó là điều kiện
    đúng để kiểm.
    """
    with tempfile.TemporaryDirectory() as d:
        cfg = Diffu2SegConfig(local_model_dir=d)
        assert cfg.model_source == cfg.hf_model_id, \
            "thư mục rỗng phải rơi về hub id, không được dùng"


def test_relative_path_resolves_against_project_root():
    """Đường dẫn mặc định là tương đối so với diffuse2seg/, không phải cwd.

    Nếu phân giải theo cwd thì chạy tool từ thư mục khác sẽ im lặng không thấy
    model rồi quay ra gọi mạng — đúng thứ đang hỏng trên server.
    """
    cfg = Diffu2SegConfig()
    assert cfg.local_model_dir.startswith("..")

    root = PROJECT
    with tempfile.TemporaryDirectory() as tmp:
        old = os.getcwd()
        try:
            os.chdir(tmp)                    # cwd khác hẳn project root
            src = cfg.model_source
        finally:
            os.chdir(old)

    expected = os.path.normpath(os.path.join(root, cfg.local_model_dir))
    assert src in (expected, cfg.hf_model_id)
    assert not src.startswith(tmp), "không được phân giải theo cwd"


def test_default_points_into_weights_dir():
    """Khớp quy ước weights/ của dự án (zip thủ công, không scp/rsync)."""
    cfg = Diffu2SegConfig()
    assert "weights" in cfg.local_model_dir
    assert "stable-diffusion" in cfg.local_model_dir


def test_to_dict_records_the_actual_source():
    """Log phải ghi nguồn THẬT đã dùng, không chỉ tên repo.

    Nếu chỉ log hf_model_id thì đọc log cũ sẽ không biết lần chạy đó lấy model
    từ đâu — mà đó đúng là câu hỏi đã tốn thời gian với detectron2 của D.1.
    """
    d = Diffu2SegConfig().to_dict()
    assert "model_source" in d
    assert "local_model_dir" in d and "hf_model_id" in d


def main_model_source():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
    return 0



# ============================================================================
# từ test_sdpa_chunked.py
# ============================================================================

class _DummyProcessor:
    """Đủ để dựng wrapper mà không cần diffusers.

    AttnProcessor2_0Wrapper.__init__ làm `self.__dict__ = other.__dict__.copy()`,
    nên `other` chỉ cần là object có __dict__. scaled_dot_product_attention chỉ
    đọc self.path và self.wrapper_callback_func — cả hai do wrapper tự đặt.
    """

    pass


def _wrapper(callback):
    return AttnProcessor2_0Wrapper(_DummyProcessor(), path=".t", callback_func=callback)


class _Collector:
    """Bắt chước phần cộng dồn của aggregator, không cần SD."""

    def __init__(self, weight=0.15):
        self.weight = weight
        self.current_merged_tensor = None
        self._raw_acc = None
        self._raw_heads = 0
        self._raw_weight = 0.0

    def callback(self, path, x):
        # Phải giống hệt collect_attention_tensors_callback, kể cả copy=True:
        # `.float()` trên tensor fp32 trả về VIEW, khiến bộ đệm alias vào x.
        B, H = x.shape[0], x.shape[1]
        for b in range(B):
            for h in range(H):
                if self._raw_acc is None:
                    self._raw_acc = x[b, h].to(torch.float32, copy=True)
                else:
                    self._raw_acc.add_(x[b, h])
                self._raw_heads += 1
                self._raw_weight = self.weight
        return x

    def _finish_layer(self):
        if self._raw_acc is None:
            return
        acc = self._raw_acc.div_(float(self._raw_heads))
        if self.current_merged_tensor is None:
            self.current_merged_tensor = acc.mul_(self._raw_weight)
        else:
            self.current_merged_tensor.add_(acc.mul_(self._raw_weight))
        self._raw_acc = None
        self._raw_heads = 0


def _make(B=1, H=8, N=32, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
    k = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
    v = torch.randn(B, H, N, D, generator=g, dtype=torch.float32)
    return q, k, v


@needs_diffusers
def test_chunked_matches_single_pass():
    """Hai nhánh của CHÍNH scaled_dot_product_attention phải trùng khít.

    Cả hai lần đều gọi hàm thật; chỉ khác CHUNK_THRESHOLD_ELEMS để ép nhánh.
    Không chép lại vòng lặp ở đây — bản chép lại vẫn xanh khi code thật sai.
    """
    q, k, v = _make()

    col_s = _Collector()
    w_s = _wrapper(col_s.callback)
    w_s.CHUNK_THRESHOLD_ELEMS = float("inf")        # ép nhánh một-lần
    out_s = w_s.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    merged_s = col_s.current_merged_tensor

    col_b = _Collector()
    w_b = _wrapper(col_b.callback)
    w_b.CHUNK_THRESHOLD_ELEMS = 0                   # ép nhánh từng-head
    out_b = w_b.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    merged_b = col_b.current_merged_tensor

    assert torch.allclose(out_s, out_b, atol=1e-6), \
        f"output lệch: {(out_s - out_b).abs().max()}"
    assert merged_s is not None and merged_b is not None, \
        "callback không chạy — test sẽ xanh giả nếu bỏ qua assert này"
    assert torch.allclose(merged_s, merged_b, atol=1e-6), \
        f"merged lệch: {(merged_s - merged_b).abs().max()}"


@needs_diffusers
def test_chunked_branch_actually_taken():
    """[NEGATIVE CONTROL] xác nhận hai nhánh THẬT SỰ khác nhau khi chạy.

    Nếu ngưỡng bị bỏ qua và cả hai lần đều chạy cùng một nhánh, test trên vẫn
    xanh mà chẳng chứng minh gì. Đếm số lần callback được gọi: nhánh một-lần
    gọi 1 lần, nhánh từng-head gọi H lần.
    """
    q, k, v = _make()
    H = q.shape[1]

    calls = []

    class _Counter(_Collector):
        def callback(self, path, x):
            calls.append(x.shape)
            return super().callback(path, x)

    c1 = _Counter()
    w1 = _wrapper(c1.callback)
    w1.CHUNK_THRESHOLD_ELEMS = float("inf")
    w1.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    n_single = len(calls)

    calls.clear()
    c2 = _Counter()
    w2 = _wrapper(c2.callback)
    w2.CHUNK_THRESHOLD_ELEMS = 0
    w2.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    n_chunked = len(calls)

    assert n_single == 1, f"nhánh một-lần phải gọi callback 1 lần, đo {n_single}"
    assert n_chunked == H, f"nhánh từng-head phải gọi {H} lần, đo {n_chunked}"


@needs_diffusers
def test_callback_does_not_mutate_input():
    """[NEGATIVE CONTROL] callback không được sửa x tại chỗ.

    SDPA dùng LẠI chính tensor đó cho `attn_weight @ value` sau khi callback
    trả về. Nếu bộ đệm alias vào x (ví dụ `.float()` trên tensor đã fp32 trả
    về view), các head sau ghi đè lên head 0 và output sai — nhưng chỉ ở head
    0, nên nhìn qua rất giống nhiễu số học.

    Đo 2026-09-16: lệch 4.97, và đường chạy thật (fp16) KHÔNG lộ vì đổi dtype
    thì `.float()` có sao chép. Test này ép fp32 để bắt.
    """
    q, k, v = _make()
    col = _Collector()
    w = _wrapper(col.callback)
    w.CHUNK_THRESHOLD_ELEMS = float("inf")

    sf = 1.0 / (q.shape[-1] ** 0.5)
    attn = torch.softmax(q @ k.transpose(-2, -1) * sf, dim=-1)
    before = attn.clone()

    col.callback(".t", attn)

    assert torch.equal(attn, before), \
        f"callback đã sửa x tại chỗ: lệch tối đa {(attn - before).abs().max()}"


@needs_diffusers
def test_per_head_division_would_be_8x_wrong():
    """[NEGATIVE CONTROL] chia B*H của từng lần gọi -> gấp H lần.

    Đây chính là lỗi đã mắc. Nếu ai đó 'đơn giản hoá' _finish_layer về lại
    phép chia trong callback, test này phải đỏ.
    """
    q, k, v = _make()
    H = q.shape[1]
    col = _Collector()
    sf = 1.0 / (q.shape[-1] ** 0.5)

    wrong = None
    for b in range(q.shape[0]):
        for h in range(H):
            w = torch.softmax(q[b, h] @ k[b, h].transpose(-2, -1) * sf, dim=-1)
            piece = (w / 1.0) * col.weight        # chia cho B*H == 1
            wrong = piece if wrong is None else wrong + piece
            col.callback(".t", w[None, None])
    col._finish_layer()
    right = col.current_merged_tensor

    ratio = wrong.abs().max() / right.abs().max()
    assert abs(float(ratio) - H) < 0.01, f"kỳ vọng gấp {H}, đo {ratio}"

