"""M2N2's Markov-map — VIẾT LẠI TỪ CÔNG THỨC, chỉ để MINH HOẠ cơ chế.

⚠️ KHÔNG nằm trên đường chạy Diffuse2Seg. Diffuse2Seg thay toàn bộ cơ chế này
bằng p-Laplacian (xem d2s/plaplacian.py). File này tồn tại để vẽ ra được
"chuỗi Markov lan truyền thế nào từ một prompt point", phục vụ đọc hiểu.

NGUỒN: Karmann & Urfalioglu, "Repurposing Stable Diffusion Attention for
Training-Free Unsupervised Interactive Segmentation", CVPR 2025 (M2N2), §3.3.
Viết lại từ các phương trình trong paper, KHÔNG copy code refs/repos/m2n2/.

                        BA BƯỚC, THEO ĐÚNG PAPER

1. NHIỆT ĐỘ (Eq. 5) — làm nhọn/phẳng A:
       A_new[k,l] = softmax_l( log(A[k,l]) / T )
   T < 1 làm nhọn => chuỗi cần NHIỀU bước hơn để về uniform => `m` có độ phân
   giải thời gian cao hơn. Paper: "we are able to control the rate of
   convergence by changing the entropy ... by modifying the temperature T".

2. IPF (Sinkhorn) — biến A thành DOUBLY STOCHASTIC.
   ⚠️ KHÔNG PHẢI trang trí. Paper §3.3: phân bố dừng p_inf "depends on the
   attention matrix A and is therefore DIFFERENT FOR EACH IMAGE. In order to be
   IMAGE AGNOSTIC, we remove the per-image bias in p_inf by applying IPF."
   Sau IPF, p_inf = uniform 1/N với MỌI ảnh, nên ngưỡng tau mới có cùng ý
   nghĩa giữa các ảnh. Đo được (N=256): không IPF, p_inf lệch uniform tới
   118,6 %; có IPF, lệch 0,0000 %.

3. THỜI GIAN ĐẾN (Eq. 6) — cái được TRẢ VỀ không phải p_t mà là:
       m[k] = min { t : p_t[k] / max(p_t) > tau }
   Phép chia cho max nằm TRONG định nghĩa của paper ("relative probability
   threshold"), không phải thủ thuật implement.

   ⚠️ VÌ SAO PHẢI CHIA: A doubly stochastic => p_t hội tụ về uniform = 1/N.
   Ở N=19600 thì uniform = 5e-5, KHÔNG BAO GIỜ vượt tau=0,3. Chuỗi thuần sẽ
   chạy hết max_iter mà không ô nào vượt ngưỡng. Chia cho max biến tau thành
   ngưỡng TƯƠNG ĐỐI, đo ĐỘ TƯƠNG PHẢN thay vì xác suất tuyệt đối.
   Đo được (N=128, 60 bước): chuỗi thuần 0/128 ô vượt; bản chia-max 128/128.

   Phép chia là VÔ HƯỚNG nên không đổi thứ hạng trong p_t — hai quỹ đạo lệch
   nhau 6,7e-16. Thông tin y hệt chuỗi thuần, chỉ khác thang đo.
"""

import numpy as np
import torch

__all__ = ["matrix_ipf", "markov_map_from_prompt"]


def matrix_ipf(A: torch.Tensor, iterations: int = 200) -> torch.Tensor:
    """Iterative proportional fitting -> doubly stochastic.

    Mỗi vòng: chuẩn hoá CỘT rồi chuẩn hoá HÀNG. Phép cuối là hàng nên hàng
    chính xác bằng 1; cột hội tụ dần về 1.

    ⚠️ Mặc định của M2N2 là 15 nhưng CẢ HAI call-site của họ truyền 200 — 15 là
    chưa đủ (đo N=256: cột vẫn lệch 1,6e-2 ở vòng 15, còn 2,2e-16 ở vòng 200).
    """
    for _ in range(iterations):
        A = A / A.sum(dim=0, keepdim=True).clamp_min(1e-30)
        A = A / A.sum(dim=1, keepdim=True).clamp_min(1e-30)
    return A


def markov_map_from_prompt(A, seed_index, tau=0.3, max_iterations=1000,
                           linear_interpolation=True, snapshot_steps=None):
    """Eq. 6: thời gian đến ngưỡng cho từng ô, từ MỘT prompt.

    A: (N, N) doubly stochastic. seed_index: chỉ số ô prompt trong [0, N).
    Trả (m, snapshots) với m: (N,) float — bước đầu tiên mỗi ô vượt tau;
    ô không bao giờ vượt nhận max_iterations.
    snapshots: {bước: (p_t đã chia max, m_t)} cho các bước trong
    snapshot_steps. `m_t` là Markov-map NẾU dừng ở bước đó — ô chưa vượt tau
    nhận giá trị bước hiện tại. Đây là thứ paper vẽ (Hình 3), không phải `p_t`.

    Nội suy tuyến tính (mặc định trong code M2N2) làm m liên tục:
        m = i + 1 - (p_t[k] - tau) / (delta + 1e-6)
    với delta = (1 - tau) ở i == 0, ngược lại p_t - p_{t-1}.
    ⚠️ Nhánh i == 0 là đặc biệt: dùng p_t - p_{t-1} ở đó (prev = 0) sẽ cho số
    khác (0,871 thay vì 0,937 trên ví dụ 2x2), rất dễ sót.
    """
    N = A.shape[0]
    p = torch.zeros(N, dtype=A.dtype, device=A.device)
    p[seed_index] = 1.0

    m = torch.full((N,), float(max_iterations), dtype=A.dtype, device=A.device)
    not_yet = torch.ones(N, dtype=torch.bool, device=A.device)
    passed = p > tau
    m[passed] = 0.0
    not_yet[passed] = False

    want = set(snapshot_steps or [])
    snaps = {0: (p.clone(), m.clone())} if 0 in want else {}

    for i in range(max_iterations):
        prev = p
        p = p @ A
        p = p / p.max().clamp_min(1e-30)

        now = p > tau
        upd = not_yet & now
        if upd.any():
            if linear_interpolation:
                delta = torch.full_like(p, 1.0 - tau) if i == 0 else (p - prev)
                grad = 1.0 - (p - tau) / (delta + 1e-6)
                m[upd] = (float(i) + grad)[upd]
            else:
                m[upd] = float(i + 1)
            not_yet[upd] = False

        if i + 1 in want:
            # Ảnh chụp gồm CẢ HAI:
            #   p   = trạng thái chuỗi (xác suất, đã chia max)
            #   m_t = Markov-map NẾU DỪNG Ở ĐÂY — ô chưa vượt tau nhận (i+1)
            # Cái thứ hai mới là thứ paper vẽ, và là thứ "lan dần ra" theo
            # từng bước. `p` thì luôn có đỉnh = 1 ở seed nên nhìn ít thay đổi.
            m_t = m.clone()
            m_t[not_yet] = float(i + 1)
            snaps[i + 1] = (p.clone(), m_t)
        if not not_yet.any() and not want:
            break

    return m, snaps
