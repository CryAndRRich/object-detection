"""Đặc trưng hình học tương đối giữa box đang refine và các VẬT ĐANG CÓ trong ảnh (`model.geo`, GAMMA3 —
docs/EXPERIMENT_GAMMA.md mục 14). Tính lại ở ĐẦU MỖI STAGE từ box stage đó nhận vào; FiLM vào `pro` trước cross-attn
(`DynamicStage`). Box vật = GT (`objects` của `CE130AddDataset`: ảnh inpaint đã bỏ các vật bị xoá tới lượt t), không nhiễu.

Box b (cx, cy, w, h), vật i (cxᵢ, cyᵢ, wᵢ, hᵢ), xyxy pixel cùng hệ:
    rel_i = (slog((cxᵢ − cx)/w), slog((cyᵢ − cy)/h), log wᵢ/w, log hᵢ/h, IoU(b, i), IoA(b, i) = giao / diện tích b)
    slog(x) = sign(x)·log(1 + |x|)  (nén khoảng cách — box nhỏ thì Δ/w rất lớn)
Gộp thành vector GEO_DIM = 34:
    [max IoU, log(1 + Σ IoA)]                                              2
    rel_i ⊕ cờ có-vật của GEO_K = 4 vật gần nhất (khoảng cách tâm chia (w, h))   4 × 7
    trung vị log wᵢ/w, log hᵢ/h trên mọi vật                                2
    log(1 + số vật có |Δx/w| < 2 và |Δy/h| < 2)                            1
    log(1 + số vật)                                                         1
Không có vật nào (hoặc ít hơn 4) -> phần thiếu = 0, cờ = 0.
"""

import torch

__all__ = ["GEO_K", "GEO_DIM", "pad_objects", "geo_features"]

GEO_K = 4
REL_DIM = 6
GEO_DIM = 2 + GEO_K * (REL_DIM + 1) + 2 + 1 + 1
LOG_CLAMP = 5.0
NEAR = 2.0


def pad_objects(objects, dev, min_m=GEO_K):
    """list B tensor [Mᵢ,4] (xyxy pixel) -> (objs [B,M,4] float32, mask [B,M] bool), M = max(max Mᵢ, min_m)."""
    M = max([len(o) for o in objects] + [min_m])
    objs = torch.zeros(len(objects), M, 4)
    mask = torch.zeros(len(objects), M, dtype=torch.bool)
    for i, o in enumerate(objects):
        o = torch.as_tensor(o, dtype=torch.float32).reshape(-1, 4)
        objs[i, :len(o)] = o
        mask[i, :len(o)] = True
    return objs.to(dev, non_blocking=True), mask.to(dev, non_blocking=True)


def _slog(x):
    return x.sign() * x.abs().log1p()


def geo_features(boxes, objs, mask, k=GEO_K):
    """boxes [N,4] xyxy pixel, objs [N,M,4], mask [N,M] (M >= k) -> [N, GEO_DIM] float32. Không grad về box (box đã detach)."""
    boxes, objs = boxes.float(), objs.float()
    x1, y1, x2, y2 = boxes.unbind(-1)
    w, h = (x2 - x1).clamp(min=1.0), (y2 - y1).clamp(min=1.0)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ox1, oy1, ox2, oy2 = objs.unbind(-1)
    ow, oh = (ox2 - ox1).clamp(min=1.0), (oy2 - oy1).clamp(min=1.0)
    dx = ((ox1 + ox2) / 2 - cx[:, None]) / w[:, None]
    dy = ((oy1 + oy2) / 2 - cy[:, None]) / h[:, None]
    lw = (ow / w[:, None]).log().clamp(-LOG_CLAMP, LOG_CLAMP)
    lh = (oh / h[:, None]).log().clamp(-LOG_CLAMP, LOG_CLAMP)
    iw = (torch.minimum(x2[:, None], ox2) - torch.maximum(x1[:, None], ox1)).clamp(min=0)
    ih = (torch.minimum(y2[:, None], oy2) - torch.maximum(y1[:, None], oy1)).clamp(min=0)
    inter = iw * ih
    area = (w * h)[:, None]
    iou = inter / (area + ow * oh - inter)
    ioa = inter / area
    m = mask.float()
    iou, ioa = iou * m, ioa * m
    rel = torch.stack([_slog(dx), _slog(dy), lw, lh, iou, ioa], -1) * m[..., None]          # [N,M,6]
    dist = (dx ** 2 + dy ** 2).masked_fill(~mask, float("inf"))
    near_d, near_i = dist.topk(k, dim=1, largest=False)                                      # [N,k]
    near_ok = near_d.isfinite()
    near = rel.gather(1, near_i[..., None].expand(-1, -1, REL_DIM)) * near_ok[..., None]
    near = torch.cat([near, near_ok[..., None].float()], -1).flatten(1)                      # [N, k·7]
    med = torch.stack([lw, lh], -1).masked_fill(~mask[..., None], float("nan")).nanmedian(dim=1).values
    med = torch.nan_to_num(med, nan=0.0)
    n_near = ((dx.abs() < NEAR) & (dy.abs() < NEAR) & mask).sum(1).float()
    return torch.cat([iou.max(1).values[:, None], ioa.sum(1).log1p()[:, None], near, med,
                      n_near.log1p()[:, None], mask.sum(1).float().log1p()[:, None]], -1)
