"""ProposeRefine (`model.arch: propose_refine`, GAMMA2 / GAMMA2.1 — docs/EXPERIMENT_GAMMA.md mục 13): CE-Loc (đề xuất box)
-> model refine 6 stage kiểu DiffusionDet, DÙNG CHUNG backbone ảnh.

    ảnh [B,4,T,T] (kiểu bài: RGB /255 + density .convert("L")) -> ResNet18 4 kênh của CE-Loc (`proposer.vision`, MỘT lượt)
        ├── C5 -> SpatialSoftmax mask phần đệm -> kp [B, 1024] ─┬─ CE-Loc: Linear(1024,128) ⊕ text -> U-Net 1D (đề xuất box)
        │                                                       └─ refine: Linear(1024, 256) -> token vis
        └── C2..C5 -> FPN (mới, 256 kênh) -> P2..P5 (RoIAlign của refine)
    memory refine = MLP([t ; text ; vis] + pos)   (`MemoryEncoder` của ALPHA1, vis lấy từ kp chung)
    head refine   = `DynamicRefineHead` (models/dynamic_head.py)

Khuếch tán của refine như DiffusionDet: lịch cosine T = 1000, box cxcywh / (nw, nh) VÙNG ẢNH THẬT · 2 − 1, nhân snr 2, kẹp
[−snr, snr]; w, h của box nhiễu >= MIN_WH (box w = 0 thì apply_deltas kẹt suy biến — như GAMMA1). CE-Loc giữ khuếch tán của
bài (tuyến tính, box chia canvas). Train refine: x0 = lỗ đích, `k` bản nhiễu / ảnh, t ~ U[0, T), loss Σ_stage 5·L1 + 2·(1 − GIoU).
  `freeze_proposer: true`  (GAMMA2)  CE-Loc (ResNet18 + SpatialSoftmax + Linear + text + U-Net) đóng băng, eval mode (BN dùng
                           thống kê chạy) — chỉ train FPN + memory + head refine.
  `freeze_proposer: false` (GAMMA2.1) train chung: loss = loss refine + `proposer_weight` · ε-MSE của CE-Loc trên cùng backbone.
Suy luận (`sample_variants`): CE-Loc sinh n box (DDPM / mock) -> mỗi biến thể: `refine_t` None = box CE-Loc nguyên; số t* = box
CE-Loc -> không gian refine -> cộng nhiễu tới t* -> DDIM (eta 1) `refine_steps` bước về 0; "noise" = refine tự chạy từ nhiễu thuần
(t = T − 1, không nhìn box CE-Loc). Đầu ra chuẩn hoá theo VÙNG ẢNH THẬT (`box_norm = "valid"`).
`geo` (GAMMA3, mục 14): head nhận thêm box các vật đang có (`objects`, GT, không nhiễu) — FiLM theo đặc trưng hình học tương
đối ở đầu mỗi stage (models/geo.py, models/dynamic_head.py). Biến thể suy luận `{"geo": False}` = tắt nhánh geo (phép thử
mô hình có dùng box vật không).
`relation` (GAMMA3.1, mục 15): bước đầu mỗi stage = attention box -> feature RoI của các vật gần nhất, điểm cộng Rel hình học kiểu
Relation-DETR (models/relation.py). Biến thể `{"geo": False}` = bỏ bước đó, `{"rel_bias": False}` = giữ attention, bỏ Rel.
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.ops import FeaturePyramidNetwork

from ce_localization.engine.diffusion import diffusion_from_boxes
from ce_localization.models.box_policy import BoxPolicy, boxes_to_unit, unit_to_boxes
from ce_localization.models.box_refiner import MIN_WH, paired_giou
from ce_localization.models.dynamic_head import DynamicRefineHead
from ce_localization.models.geo import pad_objects
from ce_localization.models.memory import MemoryEncoder
from ce_localization.utils.box_ops import cxcywh_to_xyxy
from ce_localization.utils.diffusion_math import cosine_alphas_cumprod, ddim_time_pairs, predict_noise_from_start

__all__ = ["ProposeRefine", "refine_noisy_boxes", "R18_CHANNELS"]

R18_CHANNELS = (64, 128, 256, 512)


def refine_noisy_boxes(x, whwh, snr):
    """x [...,4] không gian khuếch tán của refine -> xyxy pixel: kẹp [−snr, snr], w, h >= MIN_WH (tỉ lệ vùng thật)."""
    b = (x.clamp(-snr, snr) / snr + 1) / 2
    b = torch.cat([b[..., :2], b[..., 2:].clamp(min=MIN_WH)], dim=-1)
    return cxcywh_to_xyxy(b) * whwh


class ProposeRefine(nn.Module):
    def __init__(self, proposer_kw, d_model=256, n_stage=6, dim_feedforward=2048, nhead=8, dropout=0.0, dim_dynamic=64,
                 num_dynamic=2, text_dim=512, num_timesteps=1000, snr_scale=2.0, l1_weight=5.0, giou_weight=2.0,
                 freeze_proposer=True, proposer_weight=1.0, geo=False, geo_hidden=256, relation=False, relation_k=32,
                 relation_embed=16):
        super().__init__()
        self.proposer = BoxPolicy(**proposer_kw)
        if self.proposer.vision_kind != "r18_paper":
            raise ValueError("ProposeRefine dùng chung ResNet18 của CE-Loc: proposer cần vision r18_paper")
        if self.proposer.use_refiner or not self.proposer.use_condition:
            raise ValueError("ProposeRefine chưa hỗ trợ proposer có refiner / use_condition: false (DELTA)")
        self.fpn = FeaturePyramidNetwork(list(R18_CHANNELS), d_model)
        self.memory = MemoryEncoder("spatial_softmax", d_model, text_dim, feat_channels=R18_CHANNELS[-1], feat_stride=32)
        self.geo, self.relation = bool(geo), bool(relation)
        self.head = DynamicRefineHead(n_stage, d_model, dim_feedforward, nhead, dropout, dim_dynamic, num_dynamic,
                                      geo_hidden=geo_hidden if self.geo else None,
                                      relation=dict(k_near=relation_k, rel_embed_dim=relation_embed) if self.relation else None)
        self.num_timesteps, self.snr = num_timesteps, snr_scale
        self.l1_weight, self.giou_weight, self.proposer_weight = l1_weight, giou_weight, proposer_weight
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(num_timesteps).float(), persistent=False)
        self.box_norm = "valid"
        self.freeze_proposer = freeze_proposer
        if freeze_proposer:
            for p in self.proposer.parameters():
                p.requires_grad_(False)
        self.track_attn = False
        self._attn, self._attn_key = {}, 0          # attention cộng dồn THEO biến thể refine (khoá = chỉ số biến thể)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_proposer:                    # CE-Loc đóng băng: BN luôn dùng thống kê chạy của pha 1
            self.proposer.eval()
        return self

    # ------------------------------------------------------------------ phần dùng chung
    def encode(self, images, valid_hw):
        """-> (P2..P5 của refine, kp SpatialSoftmax [B, 1024]). Backbone CE-Loc không grad khi đóng băng."""
        vis_enc = self.proposer.vision
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_proposer):
            c = vis_enc.features(images)
            kp = vis_enc.keypoints_flat(c[-1], valid_hw)
        p = self.fpn(OrderedDict((f"p{i + 2}", f) for i, f in enumerate(c)))
        return list(p.values()), kp

    def vis_token(self, kp):
        return self.memory.ss_proj(kp)[:, None] + self.memory.cond_pos_emb[:, 2:3]

    @property
    def needs_objects(self):
        """Train / suy luận cần `objects` (box các vật đang có) — GAMMA3 / 3.1 (refine), GAMMA4.1 (CE-Loc `obj_attn`)."""
        return self.geo or self.relation or self.proposer.obj_attn

    def proposer_obj(self, objects, valid_hw, canvas, cond, generator=None, cap=False):
        """Token box vật cho CE-Loc đề xuất (GAMMA4.1: proposer `obj_attn`, như `BoxPolicy.sample`) | None."""
        if not self.proposer.obj_attn:
            return None
        return self.proposer.object_tokens(objects, valid_hw, canvas, cond, generator, cap)

    def refine(self, feats, vis, text_raw, t, boxes, need_weights=False, geo=None, rel=None):
        """boxes [B,K,4] xyxy pixel, t [B·K], geo (vật [B·K,M,4], mask) | None, rel (`_rel`) | None
        -> (box mọi stage [S, B·K, 4], attention [S, B·K, 3] | None).
        Memory + head LUÔN fp32 kể cả khi train AMP (`training.amp`): head kiểu DiffusionDet vỡ với fp16 (CLAUDE.md);
        AMP chỉ áp cho backbone + FPN + U-Net 1D."""
        K = boxes.shape[1]
        with torch.autocast(device_type=boxes.device.type, enabled=False):
            feats = [f.float() for f in feats]
            mem, _ = self.memory(t, text_raw.float().repeat_interleave(K, 0), vis.float().repeat_interleave(K, 0))
            return self.head(feats, boxes.float(), t, mem, need_weights, geo, rel)

    def _geo(self, objects, K, dev):
        """objects: list B tensor [Mᵢ,4] xyxy pixel canvas -> (vật [B·K,M,4], mask [B·K,M]) | None khi không có nhánh geo."""
        if not self.geo:
            return None
        if objects is None:
            raise ValueError("model.geo cần `objects` (box các vật đang có) khi train / suy luận")
        objs, mask = pad_objects(objects, dev)
        return objs.repeat_interleave(K, 0), mask.repeat_interleave(K, 0)

    def _rel(self, objects, K, whwh, dev):
        """-> dict vật cho `relation` (models/relation.py) | None. whwh [B,4] vùng thật của từng ảnh."""
        if not self.relation:
            return None
        if objects is None:
            raise ValueError("model.relation cần `objects` (box các vật đang có) khi train / suy luận")
        objs, mask = pad_objects(objects, dev, min_m=1)
        B = len(objects)
        return {"objs": objs, "mask": mask, "img": torch.arange(B, device=dev).repeat_interleave(K),
                "whwh": whwh.float().repeat_interleave(K, 0), "whwh_img": whwh.float()}

    @staticmethod
    def _whwh(valid_hw):
        return torch.stack([valid_hw[:, 1], valid_hw[:, 0], valid_hw[:, 1], valid_hw[:, 0]], 1).float()

    # ------------------------------------------------------------------ train
    def forward(self, images, text_raw, valid_hw, target, whwh, k=1, generator=None, objects=None):
        """target [B,4] xyxy pixel canvas (lỗ đích), whwh [B,4] vùng thật, objects (geo) list B [Mᵢ,4] xyxy pixel canvas
        -> (loss, {loss, loss_per_stage, loss_eps})."""
        feats, kp = self.encode(images, valid_hw)
        B, dev = images.shape[0], images.device
        loss_eps = images.new_zeros(())
        if not self.freeze_proposer and self.proposer_weight > 0:          # GAMMA2.1: giữ CE-Loc khớp backbone chung
            canvas = torch.full_like(whwh, float(images.shape[-1]))
            cond = self.proposer.cond_from_keypoints(kp, text_raw)
            loss_eps = self.proposer.eps_loss(cond, boxes_to_unit(target, canvas), 1, generator,
                                              self.proposer_obj(objects, valid_hw, images.shape[-1], cond, generator, cap=True))
        wk = whwh.repeat_interleave(k, 0)
        gt = target.repeat_interleave(k, 0)
        x0 = diffusion_from_boxes(gt, wk, self.snr)
        t = torch.randint(0, self.num_timesteps, (B * k,), device=dev, generator=generator)
        noise = torch.randn(x0.shape, device=dev, generator=generator)
        ab = self.alphas_cumprod[t].unsqueeze(-1)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * noise
        preds, _ = self.refine(feats, self.vis_token(kp), text_raw, t, refine_noisy_boxes(xt, wk, self.snr).view(B, k, 4),
                               geo=self._geo(objects, k, dev), rel=self._rel(objects, k, whwh, dev))
        per = torch.stack([self.l1_weight * ((p - gt) / wk).abs().sum(-1).mean()
                           + self.giou_weight * (1 - paired_giou(p, gt)).mean() for p in preds])
        loss = per.sum() + self.proposer_weight * loss_eps
        st = {"loss": loss.detach(), "loss_per_stage": per.detach()}
        if not self.freeze_proposer and self.proposer_weight > 0:              # GAMMA2 đóng băng: không có ε-MSE để báo
            st["loss_eps"] = loss_eps.detach()
        return loss, st

    # ------------------------------------------------------------------ suy luận
    @torch.no_grad()
    def _refine_from(self, feats, vis, text_raw, x, wk, B, K, t_start, steps, eta, generator, geo=None, rel=None):
        """DDIM của refine từ x (không gian khuếch tán, [B·K,4]) ở t_start -> box xyxy pixel [B·K,4]."""
        ac = self.alphas_cumprod
        dev = x.device
        out = None
        for t, t_next in ddim_time_pairs(t_start + 1, min(steps, t_start + 1)):   # t* nhỏ: không lặp lại cùng một t
            tb = torch.full((B * K,), t, device=dev, dtype=torch.long)
            preds, attn = self.refine(feats, vis, text_raw, tb, refine_noisy_boxes(x, wk, self.snr).view(B, K, 4),
                                      self.track_attn, geo, rel)
            if attn is not None:
                s, n = self._attn.get(self._attn_key, (0.0, 0))
                self._attn[self._attn_key] = (s + attn.float().sum(1), n + attn.shape[1])
            out = preds[-1]
            if t_next < 0:
                break
            x0 = diffusion_from_boxes(out, wk, self.snr)
            eps = predict_noise_from_start(x, t, x0, ac)
            a, a_next = ac[t], ac[t_next]
            sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).sqrt()
            c = (1 - a_next - sigma ** 2).sqrt()
            x = x0 * a_next.sqrt() + c * eps + sigma * torch.randn(x.shape, device=dev, generator=generator)
        return out

    @torch.no_grad()
    def sample_variants(self, images, text_raw, valid_hw, n_samples, generator=None, variants=({"refine_t": None},),
                        proposer_sampler="ddpm", eta=1.0, objects=None):
        """CE-Loc sinh n_samples box MỘT lần, mỗi biến thể refine dùng lại -> list [B, n, 4] chuẩn hoá theo vùng thật.
        Biến thể: {"refine_t": None | int t* | "noise", "refine_steps": S (mặc định 1), "geo": True | False (tắt nhánh geo /
        relation), "rel_bias": True | False (relation: bỏ Rel hình học)}."""
        feats, kp = self.encode(images, valid_hw)
        B, K, dev = images.shape[0], n_samples, images.device
        whwh = self._whwh(valid_hw)
        wk = whwh.repeat_interleave(K, 0)
        canvas = torch.full((4,), float(images.shape[-1]), device=dev)
        cond = self.proposer.cond_from_keypoints(kp, text_raw)
        u_ce = self.proposer.sample_from_cond(cond, K, generator, proposer_sampler,
                                              obj=self.proposer_obj(objects, valid_hw, images.shape[-1], cond))
        box_ce = unit_to_boxes(u_ce.reshape(-1, 4), canvas)                 # [B·K,4] xyxy pixel canvas
        vis = self.vis_token(kp)
        geo_all = self._geo(objects, K, dev)
        rel_all = self._rel(objects, K, whwh, dev)
        outs = []
        for vi, v in enumerate(variants):
            self._attn_key = vi
            rt, S = v.get("refine_t"), v.get("refine_steps", 1)
            geo = geo_all if v.get("geo", True) else None
            rel = (None if rel_all is None or not v.get("geo", True) else
                   {**rel_all, "bias": bool(v.get("rel_bias", True))})
            if rt is None:
                box = box_ce
            elif rt == "noise":
                x = torch.randn((B * K, 4), device=dev, generator=generator)
                box = self._refine_from(feats, vis, text_raw, x, wk, B, K, self.num_timesteps - 1, S, eta, generator, geo, rel)
            else:
                rt = int(rt)
                ab = self.alphas_cumprod[rt]
                x0 = diffusion_from_boxes(box_ce, wk, self.snr)
                x = ab.sqrt() * x0 + (1 - ab).sqrt() * torch.randn(x0.shape, device=dev, generator=generator)
                box = self._refine_from(feats, vis, text_raw, x, wk, B, K, rt, S, eta, generator, geo, rel)
            outs.append(boxes_to_unit(box, wk).view(B, K, 4))
        return outs

    @torch.no_grad()
    def sample(self, images, text_raw, valid_hw, n_samples, generator=None, refine_t=None, refine_steps=1,
               proposer_sampler="ddpm", eta=1.0, objects=None):
        return self.sample_variants(images, text_raw, valid_hw, n_samples, generator,
                                    [{"refine_t": refine_t, "refine_steps": refine_steps}], proposer_sampler, eta,
                                    objects)[0]

    def pop_attn(self, key=0):
        """-> attention TB lên [t ; text ; vis] mỗi stage của biến thể `key` (None nếu biến thể đó không chạy refine)."""
        if key not in self._attn:
            return None
        s, n = self._attn.pop(key)
        return [dict(zip(("t", "text", "vis"), row)) for row in (s / max(n, 1)).tolist()]
