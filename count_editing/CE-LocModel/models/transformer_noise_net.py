import torch
import torch.nn as nn

from models.components import SinusoidalPosEmb

# Rewritten from the CNN vs Transformer ablation in Diffusion Policy
# (Chi et al., RSS 2023, arXiv:2303.04137, Fig 2c / Table 8) — see
# refs/DiffusionPolicy.md. Rewritten from the paper's formulas, not imported
# from repos/diffusion_policy/ (read-only reference).
#
# Diffusion Policy always has an observation condition and a fixed T=horizon,
# so this port drops the encoder-only ("BERT") and causal-attention branches
# of the original TransformerForDiffusion — CE-Loc always has `cond`.
# horizon=1 (a single box) is the default and reproduces the original
# behavior exactly; horizon=N>1 is variant (c) — sinh N box cùng lúc. The
# decoder's own self-attention sublayer among `tgt` tokens (not disabled here,
# no causal mask) is what lets the N box tokens see each other, on top of each
# one cross-attending to the shared [time, cond] memory.


class TransformerNoisePredNet(nn.Module):
    def __init__(
        self,
        input_dim,
        cond_dim,
        horizon=1,
        n_layer=8,
        n_head=8,
        n_emb=256,
        p_drop_emb=0.1,
        p_drop_attn=0.3,
        n_cond_layers=0,
        num_classes=0,
    ):
        super().__init__()
        # box token stream: horizon=1 for variants (a)/(b) (a single
        # [x,y,w,h] box), horizon=N for variant (c) (N boxes denoised jointly)
        T = horizon
        # condition stream: 1 timestep token + 1 observation (vision+text) token
        T_cond = 2

        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))

        # Condition encoder. Diffusion Policy's own configs all ship
        # n_cond_layers=0, i.e. an MLP rather than a transformer encoder — the
        # transformer branch exists but is never the released default, so it is
        # opt-in here too (`noise_net.transformer.n_cond_layers` in the yaml).
        if n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
                dropout=p_drop_attn, activation="gelu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_cond_layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(n_emb, 4 * n_emb),
                nn.Mish(),
                nn.Linear(4 * n_emb, n_emb),
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb, nhead=n_head, dim_feedforward=4 * n_emb,
            dropout=p_drop_attn, activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, input_dim)
        # Nhánh đầu ra thứ 2: điểm cho mỗi box, song song với toạ độ.
        #   num_classes == 0  -> KHÔNG có head này (hành vi cũ, bit-identical).
        #   num_classes == 1  -> score head: "box này có phải vật thật không".
        #                        Nhãn lấy từ Hungarian matching (matched=1, còn lại=0).
        #                        Dùng cho CE-130, nơi mỗi ảnh chỉ có ĐÚNG 1 class
        #                        (đã đo: 3598/3598 ảnh) nên phân loại class là vô nghĩa.
        #   num_classes == C  -> class head đúng DiffusionDet: C logit/box, nền là
        #                        "mọi logit đều thấp" (focal loss, không cần lớp nền riêng).
        self.num_classes = num_classes
        self.class_head = nn.Linear(n_emb, num_classes) if num_classes > 0 else None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout, SinusoidalPosEmb, nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer, nn.TransformerEncoder,
            nn.TransformerDecoder, nn.ModuleList, nn.Mish, nn.Sequential,
        )
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = ["in_proj_weight", "q_proj_weight", "k_proj_weight", "v_proj_weight"]
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ["in_proj_bias", "bias_k", "bias_v"]
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerNoisePredNet):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError(f"Unaccounted module {module}")

    def forward(self, sample, timestep, cond, **kwargs):
        """
        sample: (B, T, input_dim) — noisy box(es); T=1 for (a)/(b), T=N for (c)
        timestep: (B,) or scalar — one timestep per IMAGE, shared by every box
                  in that image's `sample` (matches DiffusionDet: one t per
                  image, not per proposal)
        cond: (B, 1, cond_dim) — vision+text embedding, one token
        output: (B, T, input_dim), hoặc (boxes, logits) khi num_classes > 0
        """
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B,1,n_emb)

        cond_obs_emb = self.cond_obs_emb(cond)  # (B,1,n_emb)
        cond_embeddings = torch.cat([time_emb, cond_obs_emb], dim=1)  # (B,2,n_emb)
        x = self.drop(cond_embeddings + self.cond_pos_emb)
        memory = self.encoder(x)  # self-attention over [time, obs]

        input_emb = self.input_emb(sample)  # (B,1,n_emb)
        x = self.drop(input_emb + self.pos_emb)
        x = self.decoder(tgt=x, memory=memory)  # cross-attention: box query -> K/V from memory

        x = self.ln_f(x)
        boxes = self.head(x)  # (B,T,input_dim)
        if self.class_head is None:
            return boxes
        return boxes, self.class_head(x)  # (B,T,input_dim), (B,T,num_classes)
