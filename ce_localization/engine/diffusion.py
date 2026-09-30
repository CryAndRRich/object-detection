"""Khuếch tán trên box — giống hệt DiffusionDet (`refs/repos/DiffusionDet/diffusiondet/detector.py`).

Train (`prepare_diffusion_concat`, :370-405), MỖI ẢNH:
  t ~ U[0, T) ; GT chuẩn hoá cxcywh theo `images_whwh` (vùng ảnh thật)
  ít GT hơn N: thêm placeholder randn/6 + 0.5, w/h kẹp >= 1e-4 (KHÔNG dùng FIX của vòng 1)
  nhiều GT hơn N: lấy ngẫu nhiên N ; không có GT: một box giả [0.5, 0.5, 1, 1]
  x_start = (box·2 − 1)·snr -> q_sample -> kẹp [−snr, snr] -> về [0,1] -> xyxy -> × whwh
Suy luận (`ddim_sample`, :187-274): x_T ~ N(0, I) (KHÔNG nhân snr), mỗi bước dự đoán x0 từ stage
cuối, tính lại ε từ x0 đã kẹp, cập nhật DDIM với eta = 1.
  - 1 bước (chính, `SAMPLE_STEP=1`): renewal/ensemble không có tác dụng.
  - nhiều bước: box renewal (score > 0.5, bơm thêm nhiễu cho đủ N) + ensemble các bước rồi NMS
    ở hậu xử lý. Khác bản gốc MỘT chỗ, có chủ đích: bản gốc bỏ sót kết quả của BƯỚC CUỐI khỏi
    ensemble (`continue` ở :222 chạy trước khi thêm vào ensemble) — ở đây bước cuối CÓ vào.
"""

import torch

from ce_localization.utils.box_ops import cxcywh_to_xyxy, xyxy_to_cxcywh
from ce_localization.utils.diffusion_math import (cosine_alphas_cumprod, ddim_time_pairs,
                                                  predict_noise_from_start, q_sample)

__all__ = ["cosine_alphas_cumprod", "boxes_from_diffusion", "diffusion_from_boxes",
           "prepare_train_boxes", "ddim_sample"]


def boxes_from_diffusion(x, whwh, snr):
    """x [..., 4] không gian khuếch tán -> xyxy tuyệt đối (kẹp trước khi đổi, như bản gốc)."""
    b = (x.clamp(-snr, snr) / snr + 1) / 2
    return cxcywh_to_xyxy(b) * whwh


def diffusion_from_boxes(boxes_abs, whwh, snr):
    """xyxy tuyệt đối -> không gian khuếch tán, kẹp [−snr, snr] (như `model_predictions`)."""
    b = xyxy_to_cxcywh(boxes_abs / whwh)
    return ((b * 2 - 1) * snr).clamp(-snr, snr)


def _x_start_one(gt_xyxy, whwh1, num_proposals, snr, generator):
    dev = whwh1.device
    gt = xyxy_to_cxcywh(gt_xyxy / whwh1) if len(gt_xyxy) else \
        torch.tensor([[0.5, 0.5, 1.0, 1.0]], device=dev)
    m = gt.shape[0]
    if m < num_proposals:
        ph = torch.randn(num_proposals - m, 4, device=dev, generator=generator) / 6.0 + 0.5
        ph[:, 2:] = ph[:, 2:].clamp(min=1e-4)
        x0 = torch.cat([gt, ph], dim=0)
    elif m > num_proposals:
        idx = torch.randperm(m, device=dev, generator=generator)[:num_proposals]
        x0 = gt[idx.sort().values]
    else:
        x0 = gt
    return (x0 * 2.0 - 1.0) * snr


def prepare_train_boxes(gt_list, whwh, num_proposals, alphas_cumprod, snr=2.0, generator=None,
                        t=None):
    """GT (list [M_i,4] xyxy tuyệt đối) -> (box nhiễu [B,N,4] xyxy tuyệt đối, t [B] long).

    `t` truyền vào (chẩn đoán) thì dùng chung cho mọi ảnh; không thì mỗi ảnh một t ngẫu nhiên.
    """
    dev = whwh.device
    B, T = len(gt_list), alphas_cumprod.shape[0]
    if t is None:
        ts = torch.randint(0, T, (B,), device=dev, generator=generator)
    else:
        ts = torch.full((B,), int(t), device=dev, dtype=torch.long)
    out = []
    for i, gt in enumerate(gt_list):
        x_start = _x_start_one(gt.to(dev).float(), whwh[i], num_proposals, snr, generator)
        noise = torch.randn(num_proposals, 4, device=dev, generator=generator)
        x = q_sample(x_start, int(ts[i]), noise, alphas_cumprod.to(dev))
        out.append(boxes_from_diffusion(x, whwh[i], snr))
    return torch.stack(out), ts.long()


@torch.no_grad()
def ddim_sample(head_fn, B, num_proposals, whwh, alphas_cumprod, snr=2.0, steps=1, eta=1.0,
                renewal=True, generator=None):
    """head_fn(boxes_abs [B,N,4], t [B]) -> (logits [S,B,N,1], boxes [S,B,N,4]).

    -> dict:
      boxes [B,K,4] xyxy tuyệt đối, scores [B,K]  — 1 bước: K = N (đầu ra stage cuối);
                                                     nhiều bước: gộp ensemble mọi bước
      stage_boxes [S,B,N,4]                        — các stage ở bước CUỐI (đo recall/stage)
    Nhiều bước + renewal chỉ chạy với B = 1, như bản gốc (số box giữ lại khác nhau mỗi ảnh).
    """
    dev = whwh.device
    ac = alphas_cumprod.to(dev)
    pairs = ddim_time_pairs(ac.shape[0], steps)
    if steps > 1 and renewal and B != 1:
        raise ValueError("ddim_sample nhiều bước có renewal cần batch 1 (như DiffusionDet)")
    img = torch.randn(B, num_proposals, 4, device=dev, generator=generator)
    ens_boxes, ens_scores = [], []
    logits = boxes = None
    for t, t_next in pairs:
        tb = torch.full((B,), t, device=dev, dtype=torch.long)
        logits, boxes = head_fn(boxes_from_diffusion(img, whwh[:, None, :], snr), tb)
        x_start = diffusion_from_boxes(boxes[-1], whwh[:, None, :], snr)
        pred_noise = predict_noise_from_start(img, t, x_start, ac)
        score = logits[-1].squeeze(-1).sigmoid()
        if steps > 1:
            ens_boxes.append(boxes[-1])
            ens_scores.append(score)
        if t_next < 0:
            break
        if renewal and steps > 1:
            keep = score[0] > 0.5
            pred_noise, x_start, img = pred_noise[:, keep], x_start[:, keep], img[:, keep]
        a, a_next = ac[t], ac[t_next]
        sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).sqrt()
        c = (1 - a_next - sigma ** 2).sqrt()
        noise = torch.randn(img.shape, device=dev, generator=generator)
        img = (x_start * a_next.sqrt() + c * pred_noise + sigma * noise).float()
        if renewal and steps > 1:
            refill = torch.randn(1, num_proposals - img.shape[1], 4, device=dev, generator=generator)
            img = torch.cat([img, refill], dim=1)
    if steps > 1:
        return {"boxes": torch.cat(ens_boxes, dim=1), "scores": torch.cat(ens_scores, dim=1),
                "stage_boxes": boxes}
    return {"boxes": boxes[-1], "scores": logits[-1].squeeze(-1).sigmoid(), "stage_boxes": boxes}
