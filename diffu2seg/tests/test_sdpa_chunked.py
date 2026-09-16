"""SDPA theo từng head phải cho kết quả GIỐNG HỆT đường một-lần.

Đường tiết kiệm bộ nhớ (một head mỗi lúc) chỉ được bật khi tensor lớn, nên nó
KHÔNG bao giờ chạy trong test nhỏ nếu ta không ép. Vì thế test ở đây gọi thẳng
scaled_dot_product_attention với ngưỡng bị hạ xuống.

Vì sao file này tồn tại: bản đầu tiên của đường chunked có một lỗi ÂM THẦM —
callback chia cho `B*H` của riêng lần gọi, mà gọi từng head thì B*H == 1, nên
attention tích luỹ gấp 8 lần. Không assert nào bắt được: shape đúng, không
NaN, mask vẫn sinh ra, chỉ là sai thang. Đo được ratio đúng 8.00 (2026-09-16).

Run:  python -m pytest tests/test_sdpa_chunked.py -q
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from d2s.attention import AttnProcessor2_0Wrapper  # noqa: E402


class _Collector:
    """Bắt chước phần cộng dồn của aggregator, không cần SD."""

    def __init__(self, weight=0.15):
        self.weight = weight
        self.current_merged_tensor = None
        self._raw_acc = None
        self._raw_heads = 0
        self._raw_weight = 0.0

    def callback(self, path, x):
        B, H = x.shape[0], x.shape[1]
        for b in range(B):
            for h in range(H):
                head = x[b, h].float()
                if self._raw_acc is None:
                    self._raw_acc = head
                else:
                    self._raw_acc.add_(head)
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


def test_chunked_matches_single_pass():
    """Hai nhánh phải trùng khít; và tổng merged không được lệch thang."""
    q, k, v = _make()

    # nhánh 'small' (mặc định với N=32)
    col_s = _Collector()
    w_s = AttnProcessor2_0Wrapper(None, path=".t", callback_func=col_s.callback)
    out_s = w_s.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    merged_s = col_s.current_merged_tensor

    # nhánh 'big': mô phỏng đúng vòng lặp head của bản chunked
    col_b = _Collector()
    B, H = q.shape[0], q.shape[1]
    out_b = torch.empty(B, H, q.shape[-2], v.shape[-1], dtype=q.dtype)
    sf = 1.0 / (q.shape[-1] ** 0.5)
    for b in range(B):
        for h in range(H):
            w = q[b, h] @ k[b, h].transpose(-2, -1)
            w.mul_(sf)
            torch.softmax(w, dim=-1, out=w)
            w = col_b.callback(".t", w[None, None])[0, 0]
            out_b[b, h] = w @ v[b, h]
    col_b._finish_layer()
    merged_b = col_b.current_merged_tensor

    assert torch.allclose(out_s, out_b, atol=1e-6), \
        f"output lệch: {(out_s - out_b).abs().max()}"
    assert merged_s is not None and merged_b is not None
    assert torch.allclose(merged_s, merged_b, atol=1e-6), \
        f"merged lệch: {(merged_s - merged_b).abs().max()}"


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


if __name__ == "__main__":
    test_chunked_matches_single_pass()
    test_per_head_division_would_be_8x_wrong()
    print("OK")
