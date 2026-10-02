"""Attention từ box đang refine tới box các VẬT ĐANG CÓ, kiểu Relation-DETR (`model.relation`, GAMMA3.1 —
docs/EXPERIMENT_GAMMA.md mục 15). Chép theo `refs/repos/Relation-DETR/models/bricks/relation_transformer.py`
(`box_rel_encoding` :481-490, `PositionRelationEmbedding` :493-530, self-attn của decoder :446-460, `ref_point_head` :290) và
`position_encoding.py` (`get_sine_pos_embed` :115-138, `get_dim_t` :102-105); paper mục 4.1–4.2, công thức (2)–(6).

Relation-DETR: các query nhìn NHAU, điểm attention = q·k/√d + Rel(b_i, b_j). Ở đây (bài add, một box / mẫu) box nhìn VẬT:
    feature vật i = TB 7×7 RoIAlign(P2..P5, box vật i)                        (một lần mỗi lượt refine, dùng chung mọi stage)
    chọn `k_near` vật có tâm gần box nhất (Relation-DETR nhìn tất cả — giới hạn vì có ảnh > 1.000 vật)
    Rel(b, i) = ReLU(Linear(64 -> n_head)(sine(e(b, i))))   e = [log(|Δx|/w + 1), log(|Δy|/h + 1), log(w/wᵢ), log(h/hᵢ)]
                (sine 16 / số, nhiệt độ 10000, scale 100; e + sine tính dưới no_grad như bài)
    q = pro + PE(b), k = feature vật + PE(vật), v = feature vật;  PE = MLP(512 -> 256 -> 256)(sine(cx, cy, w, h / vùng thật))
    điểm = q·k/√d + Rel ; pro = LN(pro + Σ softmax(điểm) v)
Khác bài: vật là box GT cố định (feature RoI, không cập nhật qua tầng), giới hạn k_near, bỏ `query_scale` (Conditional DETR),
không có vật nào thì bước này cộng 0. `rel_bias=False` = bỏ Rel (phép thử `_norel`).
"""

import torch
import torch.nn as nn

from ce_localization.utils.box_ops import xyxy_to_cxcywh

__all__ = ["sine_embed", "box_rel_encoding", "PositionRelationEmbedding", "ObjectRelation"]


def sine_embed(x, num_pos_feats, temperature=10000.0, scale=1.0):
    """`get_sine_pos_embed` của Relation-DETR (exchange_xy=False): x [..., n] -> [..., n·num_pos_feats], sin / cos xen kẽ."""
    dim_t = temperature ** (torch.arange(num_pos_feats // 2, dtype=torch.float32, device=x.device) * 2 / num_pos_feats)
    pos = x.float().unsqueeze(-1) * scale / dim_t
    return torch.stack((pos.sin(), pos.cos()), dim=-1).flatten(-2).flatten(-2)


def box_rel_encoding(src, tgt, eps=1e-5):
    """`box_rel_encoding` của Relation-DETR: src [..., 4], tgt [..., 4] cxcywh (cùng hình, đã broadcast) -> [..., 4]."""
    xy1, wh1 = src.split([2, 2], -1)
    xy2, wh2 = tgt.split([2, 2], -1)
    dxy = torch.log((xy1 - xy2).abs() / (wh1 + eps) + 1.0)
    dwh = torch.log((wh1 + eps) / (wh2 + eps))
    return torch.cat([dxy, dwh], -1)


class PositionRelationEmbedding(nn.Module):
    """Rel = ReLU(Linear(sine(e))) — Conv 1×1 + ReLU (Conv2dNormActivation không norm, có bias) của bài viết bằng Linear."""

    def __init__(self, embed_dim=16, num_heads=8, temperature=10000.0, scale=100.0):
        super().__init__()
        self.embed_dim, self.temperature, self.scale = embed_dim, temperature, scale
        self.pos_proj = nn.Linear(embed_dim * 4, num_heads)

    def forward(self, src, tgt):
        """src, tgt [..., 4] cxcywh -> [..., n_head] >= 0."""
        with torch.no_grad():
            e = sine_embed(box_rel_encoding(src, tgt), self.embed_dim, self.temperature, self.scale)
        return torch.relu(self.pos_proj(e))


class ObjectRelation(nn.Module):
    """Phần dùng chung 6 stage (như bài: một `position_relation_embedding` + một `ref_point_head` cho cả decoder):
    feature vật, chọn vật gần, Rel, PE. Attention + LayerNorm nằm trong từng stage (`DynamicStage.rel_attn`)."""

    def __init__(self, d_model=256, n_head=8, k_near=32, rel_embed_dim=16):
        super().__init__()
        self.k_near, self.n_head = k_near, n_head
        self.rel_embed = PositionRelationEmbedding(rel_embed_dim, n_head)
        self.ref_point_head = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.d_model = d_model

    def pos(self, boxes_xyxy, whwh):
        """PE vị trí tuyệt đối (ref_point_head ∘ sine 128 / số, scale 2π) của box chuẩn hoá theo vùng ảnh thật."""
        return self.ref_point_head(sine_embed(xyxy_to_cxcywh(boxes_xyxy) / whwh, self.d_model // 2, scale=2 * torch.pi))

    def object_features(self, feats, pooler, rel):
        """-> rel thêm `feat` [B, M, d] (TB RoI 7×7, chỉ ô có vật; ô đệm = 0) và `pos` [B, M, d]."""
        objs, mask = rel["objs"], rel["mask"]
        B, M = mask.shape
        bi, mi = mask.nonzero(as_tuple=True)
        f = pooler.pool(feats, bi, objs[bi, mi]).flatten(2).mean(-1)                      # [R, d]
        feat = f.new_zeros(B, M, f.shape[-1]).index_put((bi, mi), f)
        return {**rel, "feat": feat, "pos": self.pos(objs, rel["whwh_img"][:, None])}

    def gather(self, boxes, rel):
        """boxes [N,4] xyxy (box của stage, N = B·K) -> (key [N,k,d], value [N,k,d], bias [N,k,n_head], ok [N,k]) của k vật gần nhất."""
        img = rel["img"]
        objs, mask = rel["objs"][img], rel["mask"][img]                                    # [N, M, 4], [N, M]
        c = xyxy_to_cxcywh(boxes.float())
        oc = xyxy_to_cxcywh(objs)
        dist = ((oc[..., :2] - c[:, None, :2]) ** 2).sum(-1).masked_fill(~mask, float("inf"))
        k = min(self.k_near, dist.shape[1])
        d, idx = dist.topk(k, dim=1, largest=False)
        ok = d.isfinite()
        value = rel["feat"][img[:, None], idx]                                            # [N, k, d]
        key = value + rel["pos"][img[:, None], idx]
        bias = self.rel_embed(c[:, None].expand(-1, k, -1), oc.gather(1, idx[..., None].expand(-1, -1, 4)))
        return key, value, bias, ok


def relation_attend(attn, norm, pro, boxes, whwh, shared, gathered, use_bias=True):
    """Một stage: pro [N,d] -> LN(pro + MHA(q = pro + PE(box), k, v, mask = Rel | −inf ở ô không có vật)). Box không có vật nào: cộng 0."""
    key, value, bias, ok = gathered
    N, k = ok.shape
    q = (pro + shared.pos(boxes, whwh))[:, None]
    mask = (bias if use_bias else torch.zeros_like(bias)).permute(0, 2, 1)                 # [N, n_head, k]
    none = ~ok.any(1)
    ok = ok | (none[:, None] & (torch.arange(k, device=ok.device) == 0))                 # tránh softmax toàn −inf
    mask = mask.masked_fill(~ok[:, None, :], float("-inf")).reshape(N * shared.n_head, 1, k)
    a = attn(q, key, value, attn_mask=mask, need_weights=False)[0][:, 0]
    return norm(pro + a * (~none).float()[:, None])
