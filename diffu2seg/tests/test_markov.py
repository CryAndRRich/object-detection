"""M2N2 Markov-map — kiểm công thức, KHÔNG cần GPU/SD.

⚠️ Đây là code MINH HOẠ, không nằm trên đường chạy Diffuse2Seg. Nhưng nó vẫn
cần test: nếu công thức sai thì hình vẽ ra sẽ dạy sai cơ chế, mà hình thì
nhìn nào cũng "hợp lý".

Run:  python -m pytest tests/test_markov.py -q
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from d2s.markov import matrix_ipf, markov_map_from_prompt  # noqa: E402


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


def test_partial_markov_map_freezes_then_grows():
    """m_t KHÔNG đơn điệu giảm — nó TĂNG ở ô chưa tới, và ĐÓNG BĂNG ở ô đã tới.

    ⚠️ Trực giác "m_t giảm dần về m" là SAI, và tôi đã suýt ghi vào test.
    Đo trên A_2x2: ô nền đi 1 -> 2 -> 5 -> 9.447 (TĂNG, vì ô chưa vượt tau
    mang giá trị bước hiện tại), còn ô cùng vật chốt ở 0.937 ngay từ t=1 và
    không đổi nữa.

    Đây chính là thứ làm hình "lan dần": vùng TRẮNG (giá trị thấp, đã tới)
    lớn dần qua các panel, phần còn lại tối dần đi.
    """
    m, snaps = markov_map_from_prompt(A_2x2, 0, tau=0.3, max_iterations=50,
                                      snapshot_steps=[1, 2, 5, 20])
    ts = sorted(snaps)

    # ô đã tới thì ĐÓNG BĂNG
    for t in ts:
        _, m_t = snaps[t]
        assert abs(float(m_t[0]) - 0.0) < 1e-12, "seed luôn 0"
        assert abs(float(m_t[3]) - 0.9367) < 1e-3, "ô cùng vật chốt từ t=1"

    # ô chưa tới thì TĂNG theo bước, rồi chốt khi vượt tau
    vals = [float(snaps[t][1][1]) for t in ts]
    assert vals == sorted(vals), f"ô nền phải tăng dần, đo {vals}"
    assert abs(vals[-1] - float(m[1])) < 1e-9, "cuối cùng phải bằng m thật"
