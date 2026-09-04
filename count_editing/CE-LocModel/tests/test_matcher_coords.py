"""Test không gian toạ độ của utils/matcher.py — neo vào PIXEL THẬT.

Vì sao tồn tại: `_cxcywh_to_xyxy` đã có lỗi trộn hai hệ toạ độ (tâm ở [-1,1],
extent ở [0,1]) suốt từ lúc viết cho tới 2026-09-04. Lỗi đó KHÔNG bị bắt bởi bộ
test cũ, vì test cũ chỉ kiểm các bất biến NỘI TẠI của hệ chuẩn hoá:

    - IoU của một box với chính nó = 1,0        -> vẫn đúng khi cả hai cùng sai
    - box lồng nhau cho tỉ lệ diện tích đúng    -> vẫn đúng khi CÙNG TÂM
    - round-trip normalize -> denormalize       -> không đi qua _cxcywh_to_xyxy

Lỗi chỉ lộ ra khi so với PIXEL THẬT và khi hai box LỆCH TÂM. Đó là điều bộ test
này làm, và nó phát hiện được lỗi cũ (đã kiểm: bản cũ FAIL, bản mới PASS).

Triệu chứng thực tế: box GT vẽ lên ảnh chỉ rộng bằng NỬA vật thể.

Chạy: python tests/test_matcher_coords.py
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.matcher import _cxcywh_to_xyxy, box_iou_normalized  # noqa: E402

T = 512.0


def norm(cx, cy, w, h, target=T):
    """Đúng công thức _normalize_bbox của CE130DetectionDataset / CocoDetectionDataset."""
    return torch.tensor([(cx / target) * 2 - 1, (cy / target) * 2 - 1,
                         (w / target) * 2 - 1, (h / target) * 2 - 1])


def px_xyxy(cx, cy, w, h):
    return torch.tensor([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=torch.float)


def iou_px(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def to_pixels(b_xyxy, W=T, H=T):
    """Giống hệt tools/visualize_predictions.py."""
    b = (b_xyxy + 1.0) / 2.0
    return torch.stack([b[..., 0] * W, b[..., 1] * H, b[..., 2] * W, b[..., 3] * H], dim=-1)


def test_corners_match_real_pixels():
    """Góc box sau _cxcywh_to_xyxy, quy về pixel, phải trùng pixel THẬT.

    Đây là test bắt được lỗi cũ: bản cũ cho box hẹp đúng 2x."""
    for cx, cy, w, h in [(100, 75, 50, 40), (256, 256, 100, 100), (300, 200, 150, 90),
                         (50, 400, 30, 25), (256, 256, 500, 500)]:
        got = to_pixels(_cxcywh_to_xyxy(norm(cx, cy, w, h).unsqueeze(0)))[0]
        exp = px_xyxy(cx, cy, w, h)
        err = (got - exp).abs().max().item()
        assert err < 1e-3, f"box ({cx},{cy},{w},{h}): {got.tolist()} != {exp.tolist()}"
    print("OK góc box quy về pixel khớp pixel thật (5 box, sai số < 1e-3 px)")


def test_iou_matches_pixel_iou_when_centres_differ():
    """IoU trong hệ chuẩn hoá phải bằng IoU tính bằng pixel — KỂ CẢ khi lệch tâm.

    Lỗi cũ đúng ở trường hợp cùng tâm và sai ở trường hợp lệch tâm, nên phải
    kiểm cả hai."""
    cases = [
        ((100, 75, 50, 40), (110, 80, 55, 45)),      # lệch tâm — bắt lỗi cũ
        ((100, 100, 80, 80), (160, 100, 80, 80)),    # lệch tâm nhiều
        ((256, 256, 100, 100), (256, 256, 100, 100)),  # cùng tâm, trùng khít
        ((256, 256, 100, 100), (256, 256, 200, 200)),  # cùng tâm, lồng nhau
        ((300, 200, 150, 90), (305, 205, 140, 95)),
        ((80, 80, 40, 40), (400, 400, 40, 40)),      # rời hẳn nhau -> 0
    ]
    for A, B in cases:
        a = _cxcywh_to_xyxy(norm(*A).unsqueeze(0))
        b = _cxcywh_to_xyxy(norm(*B).unsqueeze(0))
        got = box_iou_normalized(a, b)[0, 0].item()
        exp = iou_px(px_xyxy(*A).tolist(), px_xyxy(*B).tolist())
        assert abs(got - exp) < 1e-5, f"{A} vs {B}: IoU {got:.6f} != {exp:.6f}"
    print("OK IoU chuẩn hoá == IoU pixel trên 6 cặp (gồm 2 cặp LỆCH TÂM)")


def test_area_preserved():
    """Diện tích box quy về pixel phải bằng w*h thật."""
    for cx, cy, w, h in [(100, 75, 50, 40), (300, 300, 200, 120)]:
        c = to_pixels(_cxcywh_to_xyxy(norm(cx, cy, w, h).unsqueeze(0)))[0]
        area = (c[2] - c[0]) * (c[3] - c[1])
        assert abs(area.item() - w * h) < 1.0, (area.item(), w * h)
    print("OK diện tích box giữ nguyên khi quy về pixel")


def test_self_iou_is_one():
    """Bất biến cũ vẫn phải đúng (không hồi quy) — nhưng nhớ rằng CHỈ MÌNH NÓ
    thì không đủ để bắt lỗi trộn hệ toạ độ."""
    b = _cxcywh_to_xyxy(torch.stack([norm(100, 75, 50, 40), norm(300, 200, 150, 90)]))
    d = torch.diag(box_iou_normalized(b, b))
    assert torch.allclose(d, torch.ones(2), atol=1e-6), d
    print("OK IoU của box với chính nó = 1,0 (bất biến cũ, không hồi quy)")


def test_degenerate_clamped():
    """norm_w < -1 nghĩa là chiều rộng âm — phải bị chặn về 0, không cho x2<x1."""
    b = _cxcywh_to_xyxy(torch.tensor([[0.0, 0.0, -3.0, -3.0]]))
    assert b[0, 2] >= b[0, 0] and b[0, 3] >= b[0, 1], b
    print("OK box suy biến (norm_w < -1) bị clamp, không đảo góc")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)}/{len(fns)} test PASS")
