"""Hình học / tiền xử lý của hai baseline: trans <-> box CSANet, letterbox ảnh vật, luật chọn box + box cảnh + patch + resize BootPlace."""

import numpy as np
import torch
from PIL import Image

from baseline.methods.bootplace import patch_canvas, prep_images, rank_queries, resize_size, scene_boxes
from baseline.methods.csanet import fg_inputs, get_trans_label, trans_to_box


def test_csanet_trans_box_roundtrip_and_letterbox():
    rng = np.random.default_rng(0)
    for _ in range(50):
        W, H = (int(v) for v in rng.integers(60, 400, 2))
        fw, fh = (int(v) for v in rng.integers(5, 50, 2))
        b = trans_to_box([rng.uniform(0.05, 0.9), rng.uniform(0, 1), rng.uniform(0, 1)], (W, H), (fw, fh))
        assert 0 <= b[0] < b[2] <= W and 0 <= b[1] < b[3] <= H
        if b[2] < W and b[3] < H:                                     # không chạm mép: giữ tỉ lệ cạnh ảnh vật (sai số làm tròn)
            assert abs((b[2] - b[0]) / (b[3] - b[1]) - fw / fh) < 0.15 * fw / fh + 2 / (b[3] - b[1])
        x, y, w, h = int(rng.integers(0, W // 2)), int(rng.integers(0, H // 2)), int(rng.integers(4, W // 2)), int(rng.integers(4, H // 2))
        back = trans_to_box(get_trans_label((W, H), (w, h), (x, y, w, h)), (W, H), (w, h))   # box GT -> trans -> box ≈ box GT
        assert np.abs(np.asarray(back) - [x, y, x + w, y + h]).max() <= 1.5
    f, m, fb = fg_inputs(Image.new("RGB", (20, 40), (9, 9, 9)), (100, 50), 32)
    assert f.shape == (3, 32, 32) and m.shape == (1, 32, 32)
    x, y, w, h = fb.int().tolist()
    assert (m[0, y:y + h, x:x + w] == 255).all() and m.sum() == 255 * w * h and (f[:, y:y + h, x:x + w] == 9).all()


def test_bootplace_rank_scene_patch_resize():
    torch.manual_seed(0)
    logits, clip = torch.randn(40, 9), torch.randn(40) * 3
    idx = rank_queries(logits, clip, 30)
    top10 = set(torch.topk(logits.softmax(-1)[:, 0], 10).indices.tolist())
    assert set(idx[:10].tolist()) == top10 and len(set(idx.tolist())) == 30
    assert idx[0] == min(top10, key=lambda i: clip.softmax(0)[i])                  # box chính = khớp patch THẤP nhất trong top-10
    s = scene_boxes([[0, 0, 10, 20], [5, 5, 15, 9], [0, 0, 1, 1]], (100, 50), 2)
    torch.testing.assert_close(s, torch.tensor([[0.05, 0.2, 0.1, 0.4], [0.1, 0.14, 0.1, 0.08]]))   # 2 box to nhất, giữ thứ tự
    p = patch_canvas([torch.full((1, 64, 64, 3), 255, dtype=torch.uint8), torch.zeros((0, 64, 64, 3), dtype=torch.uint8)], "cpu")
    assert p.shape == (2, 4, 6400, 64)
    torch.testing.assert_close(p[0, 3, :64], torch.full((64, 64), 2.0))           # alpha 1 -> (1 − 0) / 0,5
    torch.testing.assert_close(p[0, 3, 64:], torch.zeros(6336, 64))
    assert resize_size((640, 480), 800, 960) == (720, 960) and resize_size((300, 900), 800, 1333) == (1332, 444)
    x, mask = prep_images([torch.zeros(3, 30, 40, dtype=torch.uint8), torch.zeros(3, 50, 20, dtype=torch.uint8)], [(60, 80), (70, 28)], "cpu")
    assert x.shape == (2, 3, 70, 80) and not mask[0, :60, :80].any() and mask[0, 60:].all() and mask[1, :, 28:].all()
