"""Decoder trên N box token — TransformerForDiffusion của Diffusion Policy, sửa
3 chỗ để box là một TẬP chứ không phải chuỗi có thứ tự.

Mỗi hàng [cx,cy,w,h] thành MỘT token. Cùng trọng số PE+Linear cho mọi hàng nên
khác biệt duy nhất giữa các box là 4 con số toạ độ -> HOÁN VỊ BẤT BIẾN.

Mỗi layer, một token box làm 3 việc:
  1. self-attention với N token box (kể cả chính nó) — "35 con cừu biết về nhau",
     đây là chỗ intra-category coherence sống
  2. cross-attention vào 1026 token condition — "box đọc ảnh tại vị trí của nó",
     tương đương chức năng RoIAlign nhưng học được
  3. FFN

BA SỬA BẮT BUỘC so với `transformer_for_diffusion.py` gốc:

  (a) BỎ `pos_emb` học được trên box token. Với action thì "bước thứ 3" có nghĩa;
      với box thì "box thứ 3" VÔ NGHĨA — thứ tự do prepare_diffusion_concat sinh
      ngẫu nhiên và matcher hoán vị tự do. Giữ lại thì mạng học "slot 0 thường là
      GT thật, slot 90 thường là placeholder" — đúng thứ KHÔNG được phép học vì
      lúc inference mọi slot đều từ randn.
      Vị trí đến từ SINUSOIDAL PE TRÊN TOẠ ĐỘ, không phải chỉ số mảng.

  (b) `causal_attn=False`, bỏ cả tgt_mask lẫn memory_mask. Box i phải thấy mọi
      box j và TOÀN BỘ memory.

  (c) `T`, `T_cond` động -> N đổi tự do giữa train (100) và eval (300). Làm được
      vì đã bỏ pos_emb theo chỉ số. `cond_pos_emb` thì GIỮ (memory CÓ thứ tự:
      patch thứ 500 luôn là cùng vùng ảnh).

VÌ SAO SINUSOIDAL PE CHỨ KHÔNG Linear(4->D) THÔ: Linear là tuyến tính nên vị trí
vào mạng dưới dạng ĐỘ LỚN — box ở x=0,4 cho vector gấp đôi box ở x=0,2. Sinusoidal
cho mỗi vị trí một CHỮ KÝ với tích vô hướng giảm theo khoảng cách, đúng thứ
attention cần. Quan trọng hơn: ViT dùng cùng cơ chế cho patch token nên box và
patch NÓI CÙNG NGÔN NGỮ về vị trí. (DiffusionDet không cần vì RoIAlign dùng toạ
độ trực tiếp để lấy mẫu.)
"""

import math

import torch
import torch.nn as nn

__all__ = ["BoxTransformer", "SinusoidalCoordEmbedding"]


class SinusoidalCoordEmbedding(nn.Module):
    """Mỗi toạ độ -> `dim` chiều sin/cos ở nhiều tần số; 4 toạ độ nối lại."""

    def __init__(self, dim=64, temperature=10000.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim, self.temperature = dim, temperature

    def forward(self, boxes):
        """[..., 4] trong [0,1] -> [..., 4*dim]."""
        half = self.dim // 2
        freq = torch.arange(half, device=boxes.device, dtype=torch.float32)
        freq = self.temperature ** (2 * freq / self.dim)
        x = boxes.unsqueeze(-1) * 100.0 / freq          # scale 100: [0,1] -> dải hữu dụng
        emb = torch.cat([x.sin(), x.cos()], dim=-1)
        return emb.flatten(-2)


class SinusoidalTimeEmbedding(nn.Module):
    """Time embedding chuẩn DDPM (giống `components.py::SinusoidalPosEmb`)."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        f = math.log(10000) / (half - 1)
        f = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * -f)
        a = t.float()[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], dim=-1)


class BoxTransformer(nn.Module):
    def __init__(self, d_model=256, n_layer=6, n_head=8, coord_dim=64,
                 dim_feedforward=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        self.coord_emb = SinusoidalCoordEmbedding(coord_dim)          # SỬA (a)
        self.box_proj = nn.Linear(4 * coord_dim, d_model)
        self.time_emb = SinusoidalTimeEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.Mish(), nn.Linear(d_model * 4, d_model)
        )

        # memory CÓ thứ tự -> giữ pos_emb cho nó (khác box token)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, 4096, d_model))
        nn.init.trunc_normal_(self.cond_pos_emb, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,            # comment của tác giả: "important for stability"
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layer)
        self.ln_f = nn.LayerNorm(d_model)

        self.box_head = nn.Linear(d_model, 4)      # toạ độ TRỰC TIẾP (không delta)
        self.score_head = nn.Linear(d_model, 1)    # 1 chiều: sigmoid ≡ softmax 2 chiều

    def forward(self, boxes_norm, timesteps, memory):
        """
        boxes_norm : [B, N, 4] cxcywh trong [0,1]
        timesteps  : [B] long — MỘT giá trị mỗi ảnh
        memory     : [B, M, d_model] (text + patch token)
        -> (pred_boxes [B,N,4] trong [0,1], logits [B,N])
        """
        tgt = self.box_proj(self.coord_emb(boxes_norm))               # SỬA (a)

        t_tok = self.time_mlp(self.time_emb(timesteps)).unsqueeze(1)  # [B,1,D]
        mem = torch.cat([t_tok, memory], dim=1)                       # SỬA (c): động
        mem = mem + self.cond_pos_emb[:, : mem.shape[1]]

        # SỬA (b): KHÔNG mask nào cả
        h = self.ln_f(self.decoder(tgt=tgt, memory=mem))
        return self.box_head(h).sigmoid(), self.score_head(h).squeeze(-1)
