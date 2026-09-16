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
    test_chunked_branch_actually_taken()
    test_callback_does_not_mutate_input()
    test_per_head_division_would_be_8x_wrong()
    print("OK")
