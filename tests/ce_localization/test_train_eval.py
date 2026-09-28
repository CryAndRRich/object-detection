"""Train (checkpoint/resume, grad monitor, chọn checkpoint) và eval chạy TRỌN LUỒNG
(predict -> score_records, AP tính tay, trần oracle-score)."""

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
    from ce_localization.utils.checkpoint import CheckpointManager

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

    from ce_localization.utils.checkpoint import CheckpointManager

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
    from ce_localization.utils.checkpoint import CheckpointManager, rng_state, set_rng_state

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
    from ce_localization.utils.checkpoint import CheckpointManager

    cu = {"model": {"n_layer": 6}, "diffusion": {}, "matcher": {},
          "data": {"image_size": 1024, "num_workers": 8}, "training": {"batch_size": 2}}
    doi_batch = {**cu, "training": {"batch_size": 6},
                 "data": {"image_size": 1024, "num_workers": 4}}
    doi_anh = {**cu, "data": {"image_size": 512, "num_workers": 8}}

    assert CheckpointManager.config_mismatch(cu, doi_batch) == ([], ["training"])
    assert CheckpointManager.config_mismatch(cu, doi_anh)[0] == ["data"]


def test_grad_monitor_nhom_va_ti_phan():
    from ce_localization.utils.grad_monitor import GradMonitor, group_of

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
    from ce_localization import eval as ev

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
    from ce_localization.utils.metrics_np import evaluate

    gt = np.array([[0.0, 0.0, 0.1, 0.1], [0.5, 0.5, 0.6, 0.6]])
    boxes = np.array([[0.0, 0.0, 0.1, 0.1], [0.8, 0.8, 0.9, 0.9], [0.5, 0.5, 0.6, 0.6]])
    r = evaluate([(boxes, np.array([0.9, 0.8, 0.7]), gt)], 0.5)
    assert r["AP"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert r["recall"] == pytest.approx(1.0)


def test_postprocess_top_k_truoc_roi_nms():
    from ce_localization import eval as ev

    b = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]])
    s = np.array([0.9, 0.8, 0.1])
    assert ev.postprocess(b, s, top_k=2).tolist() == [0, 1]
    assert ev.postprocess(b, s, top_k=2, nms_thr=0.5).tolist() == [0]   # trùng bị gộp
    assert ev.postprocess(b, s, top_k=3, nms_thr=0.5).tolist() == [0, 2]


def test_oracle_score_tach_loi_xep_hang_khoi_loi_box():
    """Box đúng nhưng score đảo ngược: AP thật thấp, TRẦN phải về 1.
    Box sai hoàn toàn: trần cũng 0 — sửa score vô ích."""
    from ce_localization import eval as ev

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


def test_select_metric():
    from ce_localization.train import select_metric
    assert select_metric({}) == "oracle_recall"
    assert select_metric({"eval": {"select_metric": "score_AUC"}}) == "score_AUC"
    with pytest.raises(ValueError):
        select_metric({"eval": {"select_metric": "iou_matched"}})   # cạm bẫy 3

