"""Đánh giá ALPHA — docs/EXPERIMENT_ALPHA.md mục 6.2 / 6.3.

Suy luận thật: DDIM từ nhiễu thuần (KHÔNG dùng chỉ số trong log train — cạm bẫy 9). Mọi thứ sau
model là numpy, chấm bằng `ce_localization/utils/metrics_np.py` qua `eval.score_records`:
  oracle_recall / score_AUC / mean_bestIoU trên TOÀN BỘ box trước NMS ; AP trên top-k sau NMS.
Box được chuẩn hoá theo vùng ảnh thật (whwh) trước khi chấm; IoU không đổi qua phép co giãn
theo trục nên chuẩn hoá không làm đổi số.

Thêm: `oracle_recall` theo từng stage, tách theo cỡ GT (theo ô P5 = 32 px), và chẩn đoán
attention của cross-attn (khối lượng trên token t / text / SpatialSoftmax, và lift trên ô trong
box GT ở ALPHA2).
"""

import time

import numpy as np
import torch

from ce_localization.alpha.data import to_device
from ce_localization.alpha.diffusion import prepare_train_boxes
from ce_localization.eval import postprocess, score_records
from ce_localization.utils.box_ops_np import box_iou as box_iou_np
from ce_localization.utils.box_ops_np import cxcywh_to_xyxy as c2x_np
from ce_localization.utils.box_ops_np import xyxy_to_cxcywh as x2c_np
from ce_localization.utils.log import fmt_time
from ce_localization.utils.metrics_np import nms_class_agnostic, oracle_hits

__all__ = ["SIZE_BINS", "DENSITY_BINS", "POSTPROCESS", "postprocess_nms_first", "predict", "score",
           "size_recall", "density_recall", "attention_diagnostics"]

# cạnh sqrt(w·h) của GT, pixel QUY VỀ CANVAS 512 (để so được giữa canvas 512 và 1024):
# < 1 ô P5 của canvas 512 (32 px) | 1–4 ô | > 4 ô
SIZE_BINS = ((0, 32, "<1 ô P5"), (32, 128, "1-4 ô P5"), (128, 1e9, ">4 ô P5"))


def _norm_cxcywh(boxes_abs, whwh):
    return x2c_np(boxes_abs / whwh)


@torch.no_grad()
def predict(model, loader, text_table, num_proposals, steps=1, top_k=100, nms_thr=0.5,
            renewal=True, seed=0, log_every=0, log=print):
    """-> (records, oracle_recall theo stage). Record: box cxcywh chuẩn hoá (TOÀN BỘ), score,
    keep sau top-k/NMS, gt cxcywh chuẩn hoá, gt cỡ pixel."""
    model.eval()
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev.type).manual_seed(seed)
    records, stage_hit, n_gt = [], None, 0
    t0, n_img = time.time(), len(loader.dataset)
    for bi, batch in enumerate(loader):
        batch = to_device(batch, dev)
        text = text_table(batch["text"], dev)
        out = model.sample(batch["images"], text, batch["valid_hw"], batch["whwh"],
                           num_proposals, steps=steps, renewal=renewal, generator=gen)
        whwh = batch["whwh"].cpu().numpy()
        boxes = out["boxes"].float().cpu().numpy()
        scores = out["scores"].float().cpu().numpy()
        stage = out["stage_boxes"].float().cpu().numpy()                  # [S,B,N,4]
        to512 = 512.0 / batch["images"].shape[-1]
        if stage_hit is None:
            stage_hit = np.zeros(stage.shape[0])
        for i in range(len(batch["image_id"])):
            gt_abs = batch["boxes"][i].float().cpu().numpy()
            gt = _norm_cxcywh(gt_abs, whwh[i])
            n_gt += len(gt)
            for s in range(stage.shape[0]):
                stage_hit[s] += oracle_hits(_norm_cxcywh(stage[s, i], whwh[i]), gt)[0]
            b = _norm_cxcywh(boxes[i], whwh[i])
            records.append({"image_id": batch["image_id"][i], "boxes": b, "scores": scores[i],
                            "keep": postprocess(b, scores[i], top_k, nms_thr), "gt": gt,
                            "gt_size_px": to512 * np.sqrt(np.clip(gt_abs[:, 2] - gt_abs[:, 0], 0, None)
                                                          * np.clip(gt_abs[:, 3] - gt_abs[:, 1], 0, None))})
        done = len(records)
        if log_every and (bi % log_every == 0 or done == n_img):
            el = time.time() - t0
            log(f"    [eval {done:5d}/{n_img}] {el / done * 1000:.0f} ms/ảnh | {fmt_time(el)} | "
                f"còn ~{fmt_time(el / done * (n_img - done))}")
    return records, (stage_hit / max(n_gt, 1)).tolist()


# số GT trong ảnh (kiểm giả thuyết: vật nhỏ nằm trong ảnh dày, 200 proposal không phủ hết)
DENSITY_BINS = ((0, 30, "<=30 vật"), (31, 100, "31-100 vật"), (101, 10 ** 9, ">100 vật"))


def postprocess_nms_first(boxes_cxcywh, scores, top_k, nms_thr=None):
    """Thứ tự của DiffusionDet + COCO: NMS trên TOÀN BỘ box trước, rồi giữ top-k theo score.
    (`eval.postprocess` làm ngược lại: top-k trước rồi NMS — với matcher một-nhiều như SimOTA,
    top-k toàn bản trùng, NMS xong mỗi ảnh chỉ còn ~25 box; docs mục 12.3.)"""
    scores = np.asarray(scores)
    if nms_thr is None or not len(scores):
        return np.argsort(-scores, kind="stable")[:top_k]
    return nms_class_agnostic(c2x_np(boxes_cxcywh), scores, nms_thr)[:top_k]


POSTPROCESS = {"topk_first": postprocess, "nms_first": postprocess_nms_first}


def rekeep(records, top_k, nms_thr, order="topk_first"):
    fn = POSTPROCESS[order]
    return [{**r, "keep": fn(r["boxes"], r["scores"], top_k, nms_thr)} for r in records]


def oracle_score_records(records, top_k, nms_thr, order="topk_first"):
    """Score = IoU thật lớn nhất với GT (box giữ nguyên), rồi hậu xử lý theo CÙNG thứ tự."""
    fn = POSTPROCESS[order]
    out = []
    for r in records:
        if len(r["gt"]) and len(r["boxes"]):
            sc = box_iou_np(c2x_np(r["boxes"]), c2x_np(r["gt"]))[0].max(axis=1)
        else:
            sc = np.zeros(len(r["boxes"]))
        out.append({**r, "scores": sc, "keep": fn(r["boxes"], sc, top_k, nms_thr)})
    return out


def density_recall(records, iou_thr=0.5):
    """Theo số GT của ảnh: oracle_recall (mọi box) và recall của box GIỮ LẠI sau hậu xử lý
    (có box giữ lại nào phủ GT ở IoU >= thr, không nhìn score)."""
    acc = {name: [0, 0, 0, 0] for _, _, name in DENSITY_BINS}           # hit_all, hit_kept, n_gt, n_img
    for r in records:
        n = len(r["gt"])
        name = next(nm for lo, hi, nm in DENSITY_BINS if lo <= n <= hi)
        a = acc[name]
        a[2] += n
        a[3] += 1
        if not n or not len(r["boxes"]):
            continue
        iou = box_iou_np(c2x_np(r["boxes"]), c2x_np(r["gt"]))[0]
        a[0] += int((iou.max(0) >= iou_thr).sum())
        kept = iou[r["keep"]] if len(r["keep"]) else np.zeros((0, n))
        a[1] += int((kept.max(0) >= iou_thr).sum()) if len(kept) else 0
    return {name: {"oracle_recall": h / max(g, 1), "kept_recall": k / max(g, 1), "n_gt": g, "n_img": m}
            for name, (h, k, g, m) in acc.items()}


def size_recall(records, iou_thr=0.5):
    """oracle_recall tách theo cỡ GT (SIZE_BINS)."""
    hit = {name: [0, 0] for _, _, name in SIZE_BINS}
    for r in records:
        if not len(r["gt"]):
            continue
        best = box_iou_np(c2x_np(r["boxes"]), c2x_np(r["gt"]))[0].max(0) if len(r["boxes"]) \
            else np.zeros(len(r["gt"]))
        for lo, hi, name in SIZE_BINS:
            m = (r["gt_size_px"] >= lo) & (r["gt_size_px"] < hi)
            hit[name][0] += int((best[m] >= iou_thr).sum())
            hit[name][1] += int(m.sum())
    return {name: {"oracle_recall": h / max(n, 1), "n_gt": n} for name, (h, n) in hit.items()}


def score(records, stage_recall, top_k=100, nms_thr=0.5, oracle=True, order="topk_first"):
    """Mọi chỉ số của một lượt eval -> dict (float thuần, ghi JSON được).
    `order`: "topk_first" (quy ước cũ, `eval.postprocess`) | "nms_first" (như DiffusionDet)."""
    records = rekeep(records, top_k, nms_thr, order)
    res = score_records(records)
    res["postprocess"] = order
    res["kept_per_image"] = float(np.mean([len(r["keep"]) for r in records])) if records else 0.0
    res["oracle_recall_per_stage"] = stage_recall
    res["size_recall"] = size_recall(records)
    res["density_recall"] = density_recall(records)
    if oracle:
        res["oracle_score"] = score_records(oracle_score_records(records, top_k, nms_thr, order))
    return res


# ----------------------------------------------------------------------- chẩn đoán attention

class _AttnCatcher:
    """Bắt trọng số cross-attn [B,N,M] (trung bình head) của từng stage. `TransformerDecoderLayer`
    gọi `multihead_attn(..., need_weights=False)` nên phải bọc lại để ép need_weights=True."""

    def __init__(self, head):
        self.store, self._orig = [], []
        for st in head.stages:
            mha = st.decoder.multihead_attn
            orig = mha.forward

            def fwd(*a, _orig=orig, **k):
                k["need_weights"], k["average_attn_weights"] = True, True
                out, w = _orig(*a, **k)
                self.store.append(w.detach().float())
                return out, w

            self._orig.append((mha, orig))
            mha.forward = fwd

    def close(self):
        for mha, orig in self._orig:
            mha.forward = orig


@torch.no_grad()
def attention_diagnostics(model, loader, text_table, num_proposals, ts=(99, 499, 999),
                          max_batches=20, seed=0):
    """Khối lượng attention theo stage và t, trên box NHIỄU TỪ GT (như lúc train, t cố định).

    -> {t: {"time": [S], "text": [S], "ss": [S] (ALPHA1), "grid": [S] (ALPHA2),
            "grid_lift_in_gt": [S] (ALPHA2)}}
    lift = khối lượng trên ô có tâm nằm trong box GT / tỉ lệ ô thật nằm trong box GT (1 = ngẫu nhiên).
    """
    model.eval()
    dev = next(model.parameters()).device
    kind = model.memory.kind
    gen = torch.Generator(device=dev.type).manual_seed(seed)
    out = {}
    for t in ts:
        acc = {}
        catcher = _AttnCatcher(model.head)
        try:
            for bi, batch in enumerate(loader):
                if bi >= max_batches:
                    break
                batch = to_device(batch, dev)
                text = text_table(batch["text"], dev)
                boxes, tb = prepare_train_boxes(batch["boxes"], batch["whwh"], num_proposals,
                                                model.alphas_cumprod, model.snr_scale, gen, t=t)
                catcher.store.clear()
                model(batch["images"], text, batch["valid_hw"], boxes, tb)
                gin = None
                if kind == "grid":
                    H = W = batch["images"].shape[-1] // 32
                    cxs, cys, valid = model.memory.grid_geometry(batch["valid_hw"], H, W)  # [B,K]
                    gin = []
                    for gt, cx, cy in zip(batch["boxes"], cxs, cys):
                        inside = ((cx[None] > gt[:, :1]) & (cx[None] < gt[:, 2:3])
                                  & (cy[None] > gt[:, 1:2]) & (cy[None] < gt[:, 3:4]))
                        gin.append(inside.any(0))
                    gin = torch.stack(gin) & valid                                       # [B,K]
                for s, w in enumerate(catcher.store):                                     # w [B,N,M]
                    m = w.mean(1)                                                         # [B,M]
                    rec = acc.setdefault(s, {"time": [], "text": [], "ss": [], "grid": [], "lift": []})
                    rec["time"].append(m[:, 0])
                    rec["text"].append(m[:, 1])
                    if kind == "spatial_softmax":
                        rec["ss"].append(m[:, 2])
                    if kind == "grid":
                        g = m[:, 2:]
                        rec["grid"].append(g.sum(1))
                        frac = gin.float().sum(1) / valid.float().sum(1).clamp(min=1)
                        mass_in = (g * gin.float()).sum(1) / g.sum(1).clamp(min=1e-12)
                        ok = frac > 0
                        rec["lift"].append((mass_in[ok] / frac[ok]))
        finally:
            catcher.close()
        S = len(acc)
        res = {}
        for key, name in (("time", "time"), ("text", "text"), ("ss", "ss"), ("grid", "grid"),
                          ("lift", "grid_lift_in_gt")):
            vals = [torch.cat(acc[s][key]).mean().item() if acc[s][key] else None for s in range(S)]
            if any(v is not None for v in vals):
                res[name] = vals
        out[int(t)] = res
    return out
