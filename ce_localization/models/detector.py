"""CE-Loc detector: CLIP đóng băng + decoder `BoxDiT` + khuếch tán trên box.

    ảnh -> CLIP ViT-B/16 frozen -+-> patch_raw [B,P,768] ---------------> RoI mỗi tầng
                                 +-> Linear(768->256) -+
    text -> CLIP text frozen  -----> Linear(512->256) -+--> memory [B,1+P,256]

    x_T ~ N(0,I) [B,N,4]  -> DDIM `sampling_steps` bước, mỗi bước chạy 6 DiTBlock
                          -> N box + N score

Đầu ra của decoder là TOẠ ĐỘ trực tiếp (x0), không phải epsilon — như DiffusionDet
(`objective='pred_x0'`): set matching và GIoU cần toạ độ, và 1/sqrt(alpha_bar) tới
20.291x ở t=999 sẽ khuếch đại sai số nếu suy x0 từ epsilon.
"""

import torch
import torch.nn as nn

from ce_localization.models.clip_encoder import CLIPConditionEncoder
from ce_localization.models.dit_blocks import (
    BoxCoordEmbedder,
    DiTBlock,
    TimestepConditioner,
    build_cross_mask,
    clamp_to_valid,
)
from ce_localization.utils.box_ops import decode_diffusion, encode_diffusion
from ce_localization.utils.diffusion_math import (
    cosine_alphas_cumprod,
    ddim_time_pairs,
    predict_noise_from_start,
    prepare_diffusion_concat,
)

__all__ = ["BoxDiT", "CELocDetector", "build_model"]


def _check_generator(generator, dev):
    """`torch.randn(device=X, generator=g)` đòi `g.device == X`; kiểm sớm để báo rõ."""
    if generator is not None and generator.device.type != torch.device(dev).type:
        raise ValueError(
            f"generator nằm trên {generator.device} còn model trên {dev}; "
            f"tạo generator bằng torch.Generator(device='{torch.device(dev).type}')")


class BoxDiT(nn.Module):
    """N box + điều kiện ảnh/text -> danh sách `(boxes [B,N,4], logits [B,N])`, MỘT cặp
    cho MỖI tầng (cũ trước). Loss đặt ở mọi tầng và matcher chạy lại từng tầng — chuẩn
    của 8/8 bài trong khảo sát (DETR đo +8,2 AP giữa tầng 1 và 6).
    """

    def __init__(self, d_model=256, n_layer=6, n_head=8, coord_dim=64,
                 dim_feedforward=None, dropout=0.1, roi_dim=768, roi_k=3,
                 max_cond_len=1152):
        super().__init__()
        self.d_model = d_model

        self.box_embed = BoxCoordEmbedder(d_model, coord_dim)
        self.time_cond = TimestepConditioner(d_model)

        # Memory CÓ thứ tự (patch i luôn là cùng một vùng) nên có pos_emb. Token box thì
        # KHÔNG: thứ tự do placeholder sinh ngẫu nhiên và matcher hoán vị, vị trí của box
        # đến từ sin/cos trên TOẠ ĐỘ. `max_cond_len` phải >= 1 + số patch.
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, d_model))
        nn.init.trunc_normal_(self.cond_pos_emb, std=0.02)

        self.type_emb = nn.Parameter(torch.zeros(2, d_model))      # 0: h, 1: r
        nn.init.trunc_normal_(self.type_emb, std=0.02)

        self.layers = nn.ModuleList([
            DiTBlock(d_model, n_head, dim_feedforward, dropout, roi_dim, roi_k)
            for _ in range(n_layer)
        ])

        # Score head RIÊNG từng tầng, đọc `r` (nội dung), không đọc `h` (hình học).
        # KHÔNG dùng cat([h, r]): mở lại đường score -> toạ độ mà mask đang chặn.
        self.score_head = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(n_layer)])
        self.norm_r = nn.LayerNorm(d_model)

    def forward(self, boxes_norm, timesteps, memory, patch_raw, valid_h=None):
        """
        boxes_norm : [B,N,4] cxcywh trong [0,1]
        timesteps  : [B] long — một giá trị cho mỗi ảnh
        memory     : [B,M,d_model]
        patch_raw  : [B,P,768]
        valid_h    : [B] hoặc None
        """
        if memory.shape[1] > self.cond_pos_emb.shape[1]:
            raise ValueError(f"memory có {memory.shape[1]} token nhưng cond_pos_emb chỉ "
                             f"giữ {self.cond_pos_emb.shape[1]}; tăng max_cond_len")
        memory = memory + self.cond_pos_emb[:, : memory.shape[1]]

        x = boxes_norm
        t_emb = self.time_cond(timesteps)
        h = self.box_embed(x) + self.type_emb[0]
        # Luồng ảnh khởi đầu rỗng; tầng đầu nạp RoI qua gate (một chỗ gọi RoI duy nhất).
        r = torch.zeros_like(h) + self.type_emb[1]

        mask = build_cross_mask(x.shape[1], x.device)
        outs = []
        for layer, head in zip(self.layers, self.score_head):
            x, h, r = layer(x, h, r, memory, patch_raw, t_emb, mask, valid_h)
            outs.append((x, head(self.norm_r(r)).squeeze(-1)))
        return outs


class CELocDetector(nn.Module):
    """CLIP frozen + `BoxDiT` + lịch khuếch tán cosine."""

    def __init__(self, clip_name="openai/clip-vit-base-patch16", d_model=256,
                 n_layer=6, n_head=8, image_size=512, num_timesteps=1000,
                 snr_scale=2.0, sampling_steps=4, dropout=0.1, freeze_clip=True,
                 roi_k=3, coord_dim=64):
        super().__init__()
        self.encoder = CLIPConditionEncoder(clip_name, d_model, image_size, freeze_clip)
        # max_cond_len theo ĐỘ PHÂN GIẢI THẬT: 512px -> 1024 patch, 1024px -> 4096.
        self.decoder = BoxDiT(d_model, n_layer, n_head, coord_dim, dropout=dropout,
                              roi_k=roi_k, max_cond_len=self.encoder.num_patches + 128)
        self.num_timesteps = num_timesteps
        self.sampling_steps = sampling_steps
        self.snr_scale = snr_scale
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(num_timesteps),
                             persistent=False)

    # ------------------------------------------------------------------ train

    def build_inputs(self, targets, num_proposals, valid_h, generator=None):
        """GT -> (x_t [B,N,4] trong không gian khuếch tán, t [B], is_gt [B,N]).

        MỖI ẢNH MỘT `t`, như DiffusionDet (`detector.py:375,419`). `device=dev` bắt buộc
        để `randint` và `prepare_diffusion_concat` cùng dùng một generator.
        """
        dev = self.alphas_cumprod.device
        _check_generator(generator, dev)
        ts = torch.randint(0, self.num_timesteps, (len(targets),), device=dev,
                           generator=generator)
        xs, gts = [], []
        for i, gt in enumerate(targets):
            x_t, _, is_gt = prepare_diffusion_concat(
                gt.to(dev), num_proposals, int(ts[i]), self.alphas_cumprod,
                self.snr_scale, valid_h=float(valid_h[i]), generator=generator,
            )
            xs.append(x_t)
            gts.append(is_gt)
        return torch.stack(xs), ts.long(), torch.stack(gts)

    def forward(self, x_t, timesteps, pixel_values=None, texts=None,
                patch_raw=None, text_raw=None, valid_h=None):
        """`x_t` [B,N,4] trong không gian khuếch tán -> list[(boxes [0,1], logits)]."""
        memory, praw = self.encoder(pixel_values, texts, patch_raw, text_raw,
                                    return_patch_raw=True)
        return self.decoder(decode_diffusion(x_t, self.snr_scale), timesteps, memory,
                            praw, valid_h)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def ddim_sample(self, num_proposals, pixel_values=None, texts=None,
                    patch_raw=None, text_raw=None, valid_h=None, eta=1.0,
                    generator=None, return_all_layers=False):
        """Sinh N box từ nhiễu thuần.

        - `x_T ~ N(0, I)`, std 1,0 — KHÔNG nhân `snr_scale`.
        - Mỗi bước: dự đoán `x_start` từ tầng cuối -> tính lại `pred_noise` từ bản đó.
        - `valid_h` kéo `cy` của `x_T` về vùng ảnh thật (như lúc train).
        - `eta=1.0` như DiffusionDet (khi đó DDIM suy biến thành DDPM); `eta=0` là DDIM
          tất định. Logit KHÔNG ảnh hưởng quỹ đạo — chỉ box.
        """
        memory, praw = self.encoder(pixel_values, texts, patch_raw, text_raw,
                                    return_patch_raw=True)
        B, dev = memory.shape[0], memory.device
        _check_generator(generator, dev)

        img = torch.randn(B, num_proposals, 4, device=dev, generator=generator)
        if valid_h is not None:
            img = encode_diffusion(
                clamp_to_valid(decode_diffusion(img, self.snr_scale), valid_h),
                self.snr_scale)

        layers = None
        for t, t_next in ddim_time_pairs(self.num_timesteps, self.sampling_steps):
            tb = torch.full((B,), t, dtype=torch.long, device=dev)
            layers = self.decoder(decode_diffusion(img, self.snr_scale), tb, memory,
                                  praw, valid_h)
            boxes, logits = layers[-1]

            x_start = encode_diffusion(boxes, self.snr_scale)
            if t_next < 0:
                break
            pred_noise = predict_noise_from_start(img, t, x_start, self.alphas_cumprod)
            a, a_next = self.alphas_cumprod[t], self.alphas_cumprod[t_next]
            sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).clamp(min=0).sqrt()
            c = (1 - a_next - sigma ** 2).clamp(min=0).sqrt()
            img = x_start * a_next.sqrt() + c * pred_noise
            if eta > 0:
                img = img + sigma * torch.randn(img.shape, device=dev, generator=generator)

        return layers if return_all_layers else (boxes, logits)


def build_model(cfg, dropout=None):
    """Config -> CELocDetector. MỘT chỗ dựng mô hình cho mọi điểm vào.

    `dropout=0.0` cho eval/visualise; train truyền None để lấy giá trị trong config.
    """
    m, d = cfg["model"], cfg["diffusion"]
    return CELocDetector(
        m["clip_name"], m["d_model"], m["n_layer"], m["n_head"],
        cfg["data"]["image_size"], d["num_timesteps"], d["snr_scale"],
        d["sampling_steps"], m["dropout"] if dropout is None else dropout,
        m["freeze_clip"], roi_k=m.get("roi_k", 3), coord_dim=m.get("coord_dim", 64))
