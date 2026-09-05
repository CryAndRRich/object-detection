"""CE-Loc vòng 2 — thân Diffusion Policy transformer, mượn cơ chế sinh N box của
DiffusionDet.

    ẢNH  -> CLIP ViT-B/16 FROZEN -> 1024 patch token -> Linear(768->256) ─┐
    TEXT -> CLIP text   FROZEN   -> 1 token          -> Linear(768->256) ─┤
    t    -> sinusoidal                                                    ─┤
                                                          memory 1026 token
    box nhiễu [B,N,4] -> sinPE(cx,cy,w,h) -> decoder 6 layer ─────────────┘
                              (self-attn giữa box + cross-attn vào memory)
                                        |
                            box [B,N,4] + score [B,N]

Khác CE-Loc gốc: memory 2 token -> 1026 token CÓ VỊ TRÍ. Vòng 1 đo được với 2
token thì cả N box nhận CÙNG một vector 256-d, và gradient trên box unmatched có
hướng ngẫu nhiên (cosine -0,0074 = tung đồng xu).

Đầu ra là TOẠ ĐỘ TRỰC TIẾP, không phải epsilon. DiffusionDet cũng vậy
(`objective='pred_x0'`) — nó dự đoán x_start qua delta rồi SUY NGƯỢC ra pred_noise.
Ba lý do: (1) set-matching bắt buộc, (2) GIoU chỉ định nghĩa được trên toạ độ,
(3) hệ số 1/sqrt(ab) tới 20.291x ở t=999 nên dự đoán eps rồi suy ra x0 khuếch đại
sai số chừng đó lần.
"""

import torch
import torch.nn as nn

from models.box_transformer import BoxTransformer
from models.clip_encoder import CLIPConditionEncoder
from utils.box_ops import decode_diffusion, encode_diffusion
from utils.diffusion_math import (
    cosine_alphas_cumprod, ddim_time_pairs, predict_noise_from_start,
    prepare_diffusion_concat,
)

__all__ = ["CELocDetector"]


class CELocDetector(nn.Module):
    def __init__(self, clip_name="openai/clip-vit-base-patch16", d_model=256,
                 n_layer=6, n_head=8, image_size=512, num_timesteps=1000,
                 snr_scale=2.0, sampling_steps=4, dropout=0.1, freeze_clip=True):
        super().__init__()
        self.encoder = CLIPConditionEncoder(clip_name, d_model, image_size, freeze_clip)
        self.decoder = BoxTransformer(d_model, n_layer, n_head, dropout=dropout)

        self.num_timesteps = num_timesteps
        self.snr_scale = snr_scale
        self.sampling_steps = sampling_steps
        self.register_buffer("alphas_cumprod",
                             cosine_alphas_cumprod(num_timesteps).float(), persistent=False)

    # ------------------------------------------------------------------ train

    def build_inputs(self, targets, num_proposals, valid_h, generator=None):
        """Dựng x_t cho cả batch. `t` là MỘT giá trị cho cả ảnh (đúng DiffusionDet)."""
        dev = self.alphas_cumprod.device
        t = int(torch.randint(0, self.num_timesteps, (1,), generator=generator).item())

        xs, gts = [], []
        for i, gt in enumerate(targets):
            x_t, _, is_gt = prepare_diffusion_concat(
                gt.to(dev), num_proposals, t, self.alphas_cumprod, self.snr_scale,
                valid_h=float(valid_h[i]), generator=generator,
            )
            xs.append(x_t)
            gts.append(is_gt)
        t_batch = torch.full((len(targets),), t, dtype=torch.long, device=dev)
        return torch.stack(xs), t_batch, torch.stack(gts)

    def forward(self, x_t, timesteps, pixel_values=None, texts=None,
                patch_raw=None, text_raw=None):
        """x_t [B,N,4] trong không gian diffusion -> (box [0,1], logits)."""
        memory = self.encoder(pixel_values, texts, patch_raw, text_raw)
        boxes_norm = decode_diffusion(x_t, self.snr_scale)
        return self.decoder(boxes_norm, timesteps, memory)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def ddim_sample(self, num_proposals, pixel_values=None, texts=None,
                    patch_raw=None, text_raw=None, eta=1.0, generator=None):
        """Sinh N box từ nhiễu thuần.

        x_T ~ N(0, I) std 1,0 — KHÔNG nhân snr_scale (lỗi 3 của vòng 1).
        Mỗi bước: dự đoán x_start -> CLAMP -> TÍNH LẠI pred_noise từ bản đã clamp.

        `eta=1.0` là mặc định của DiffusionDet (`detector.py:97`) — DDIM suy biến
        về DDPM. Hệ quả đã đo, KHÔNG phải lỗi: ở bước đầu (t=999 -> 749) thì
        sigma=0,925 và c=0,0000, tức pred_noise bị nhân 0 và cả bước là
        `x_start*sqrt(ab_next) + nhiễu`. Đặt eta=0 nếu muốn DDIM tất định.
        """
        memory = self.encoder(pixel_values, texts, patch_raw, text_raw)
        B, dev = memory.shape[0], memory.device

        img = torch.randn(B, num_proposals, 4, device=dev, generator=generator)
        boxes = logits = None

        for t, t_next in ddim_time_pairs(self.num_timesteps, self.sampling_steps):
            tb = torch.full((B,), t, dtype=torch.long, device=dev)
            boxes, logits = self.decoder(decode_diffusion(img, self.snr_scale), tb, memory)

            x_start = encode_diffusion(boxes, self.snr_scale)          # đã trong miền
            if t_next < 0:
                break

            pred_noise = predict_noise_from_start(img, t, x_start, self.alphas_cumprod)
            a, a_next = self.alphas_cumprod[t], self.alphas_cumprod[t_next]
            sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).clamp(min=0).sqrt()
            c = (1 - a_next - sigma ** 2).clamp(min=0).sqrt()
            img = (x_start * a_next.sqrt() + c * pred_noise
                   + sigma * torch.randn(img.shape, device=dev, generator=generator))

        return boxes, logits
