"""EXPERIMENT A (vòng 2) — box là luồng chính, ảnh vào liên tục theo toạ độ.

    ảnh  -> CLIP ViT-B/16 frozen -+-> patch_raw [B,1024,768] ----> cho RoI
                                  +-> Linear(768->256) --+
    text -> CLIP text  frozen ------> Linear(768->256) --+--> memory [B,1025,256]

    x_T ~ N(0,I) [B,30,4]                                     <- LUỒNG CHÍNH
      |
      +-- 4 bước DDIM, mỗi bước chạy 6 tầng DiTBlock:
      |     (1) r <- gate(r, roi(patch_raw, x))      lấy ảnh TẠI x hiện tại
      |     (2) seq = [h ; r + mark(x)]
      |     (3) self-attn 2N có mask, adaLN theo t
      |     (4) chỉ r cross-attn vào memory
      |     (5) x <- update_box(x, box_delta(h))     CỘNG DỒN
      |
      +--> 30 box + 30 score

KHÁC VÒNG 1 Ở ĐÂU (chi tiết trong `docs/EXPERIMENT_A_PLAN.md` mục 8):

  gốc hồi quy   anchor cố định cả 6 tầng   ->  x của tầng trước, cộng dồn
  RoI           1 lần, trước 6 tầng        ->  6 lần, tại x mới, có gate
  trộn RoI      tgt = tgt + roi(...)       ->  nối token, self-attn 2N có mask
  cross-attn    h đọc memory               ->  chỉ r đọc memory
  score head    Linear(h) / Linear([h,rf]) ->  Linear(r)
  thời gian     token thứ 1025 trong mem   ->  adaLN-Zero
  x_T inference randn, không biết valid_h  ->  nhân valid_h cho cy
  N             100 train / 300 eval       ->  30 / 30

ĐẦU RA LÀ TOẠ ĐỘ TRỰC TIẾP, KHÔNG PHẢI EPSILON — giữ nguyên lựa chọn của vòng 1, và
DiffusionDet cũng vậy (`objective='pred_x0'`). Ba lý do: (1) set matching cần toạ độ,
(2) GIoU chỉ định nghĩa được trên toạ độ, (3) hệ số 1/sqrt(alpha_bar) lên tới 20.291x
ở t=999 nên dự đoán epsilon rồi suy ra x0 sẽ khuếch đại sai số bằng đúng chừng đó.
"""

import torch
import torch.nn as nn

from models.clip_encoder import CLIPConditionEncoder
from models.dit_blocks import (
    BoxCoordEmbedder,
    DiTBlock,
    TimestepConditioner,
    build_cross_mask,
    clamp_to_valid,
)
from utils.box_ops import decode_diffusion, encode_diffusion
from utils.diffusion_math import (
    cosine_alphas_cumprod,
    ddim_time_pairs,
    predict_noise_from_start,
    prepare_diffusion_concat,
)

__all__ = ["BoxDiT", "CELocDetector", "build_model"]


def _check_generator(generator, dev):
    """`torch.randn(device=X, generator=g)` đòi `g.device == X`, nếu không sẽ ném lỗi
    khó hiểu. Kiểm sớm để báo đúng chỗ sai."""
    if generator is not None and generator.device.type != torch.device(dev).type:
        raise ValueError(
            f"generator nằm trên {generator.device} còn model trên {dev}; "
            f"tạo generator bằng torch.Generator(device='{torch.device(dev).type}')")


class BoxDiT(nn.Module):
    """Thân mô hình: N box nhiễu + điều kiện ảnh/text -> N box + N score, mỗi tầng một
    cặp.

    Trả về DANH SÁCH `(boxes, logits)` theo thứ tự tầng, cũ trước. Đây không phải tuỳ
    chọn mà là yêu cầu: mọi bài trong khảo sát (8/8) đều đặt loss ở MỌI tầng và chạy
    lại matcher cho từng tầng; DETR đo +8,2 AP giữa tầng 1 và tầng 6 nhờ cơ chế này.
    Nếu chỉ đặt loss ở tầng cuối thì `delta` của 5 tầng đầu không có tín hiệu trực tiếp
    và việc cộng dồn mất ý nghĩa.
    """

    def __init__(self, d_model=256, n_layer=6, n_head=8, coord_dim=64,
                 dim_feedforward=None, dropout=0.1, roi_dim=768, roi_k=3,
                 n_class=1, max_cond_len=1152):
        # 1152 = 1024 patch (512px) + 128 dư. Người gọi nên truyền theo độ phân giải
        # thật thay vì dựa vào mặc định này.
        super().__init__()
        self.d_model, self.n_class = d_model, n_class

        self.box_embed = BoxCoordEmbedder(d_model, coord_dim)
        self.time_cond = TimestepConditioner(d_model)

        # Memory CÓ thứ tự (patch 500 luôn là cùng một vùng ảnh) nên giữ pos_emb cho nó.
        # Token box thì KHÔNG: thứ tự do `prepare_diffusion_concat` sinh ngẫu nhiên và
        # matcher hoán vị tự do, nên pos_emb theo chỉ số sẽ dạy mạng "khe 0 thường là GT
        # thật" — đúng thứ nó không được học. Vị trí của box đến từ sin/cos trên TOẠ ĐỘ.
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, d_model))
        nn.init.trunc_normal_(self.cond_pos_emb, std=0.02)

        # Hai embedding phân biệt loại token trong chuỗi nối.
        self.type_emb = nn.Parameter(torch.zeros(2, d_model))
        nn.init.trunc_normal_(self.type_emb, std=0.02)

        self.layers = nn.ModuleList([
            DiTBlock(d_model, n_head, dim_feedforward, dropout, roi_dim, roi_k)
            for _ in range(n_layer)
        ])

        # Head score RIÊNG cho từng tầng. Vòng 1 đo được đặc trưng tới head đổi độ lớn
        # qua các tầng (chuẩn 1,44 -> 0,48), một head dùng chung phải phục vụ hai phân
        # phối bằng một hàm; head riêng hội tụ tốt hơn (0,00087 vs 0,00134).
        # Đọc `r` chứ không phải `h`: "có vật ở đây không" là câu hỏi về NỘI DUNG. Đây
        # là tổng quát hoá cơ chế đã cho 4,2x ở vòng 1 (score head chuyển sang đọc RoI).
        # KHÔNG dùng cat([h, r]) — làm vậy mở lại đường score -> toạ độ mà mask đang chặn.
        self.score_head = nn.ModuleList([nn.Linear(d_model, n_class)
                                         for _ in range(n_layer)])
        self.norm_r = nn.LayerNorm(d_model)

    def forward(self, boxes_norm, timesteps, memory, patch_raw, valid_h=None):
        """
        boxes_norm : [B,N,4] cxcywh trong [0,1]
        timesteps  : [B] long — MỘT giá trị cho mỗi ảnh
        memory     : [B,M,d_model]
        patch_raw  : [B,P,d_in]
        valid_h    : [B] hoặc None
        -> list[(boxes [B,N,4], logits [B,N] hoặc [B,N,C])] dài n_layer
        """
        if memory.shape[1] > self.cond_pos_emb.shape[1]:
            raise ValueError(
                f"memory có {memory.shape[1]} token nhưng cond_pos_emb chỉ giữ "
                f"{self.cond_pos_emb.shape[1]}; tăng max_cond_len")
        memory = memory + self.cond_pos_emb[:, : memory.shape[1]]

        x = boxes_norm
        t_emb = self.time_cond(timesteps)
        h = self.box_embed(x) + self.type_emb[0]
        # Khởi tạo luồng ảnh bằng 0 rồi để tầng đầu nạp qua gate, thay vì gọi RoI thêm
        # một lần ở đây: giữ đúng một chỗ duy nhất gọi RoI (bên trong DiTBlock).
        r = torch.zeros_like(h) + self.type_emb[1]

        mask = build_cross_mask(x.shape[1], x.device)
        outs = []
        for i, layer in enumerate(self.layers):
            x, h, r = layer(x, h, r, memory, patch_raw, t_emb, mask, valid_h)
            logits = self.score_head[i](self.norm_r(r))
            outs.append((x, logits.squeeze(-1) if self.n_class == 1 else logits))
        return outs


class CELocDetector(nn.Module):
    """CLIP frozen + BoxDiT + khuếch tán. Bao trọn một thí nghiệm."""

    def __init__(self, clip_name="openai/clip-vit-base-patch16", d_model=256,
                 n_layer=6, n_head=8, image_size=512, num_timesteps=1000,
                 snr_scale=2.0, sampling_steps=4, dropout=0.1, freeze_clip=True,
                 roi_k=3, n_class=1, use_text=True, coord_dim=64):
        super().__init__()
        if (n_class > 1) != (not use_text):
            raise ValueError(
                f"n_class={n_class} với use_text={use_text}: lớp phải đến với mô hình "
                f"qua ĐÚNG MỘT đường — hoặc token text (n_class=1), hoặc head nhiều "
                f"lớp (use_text=False).")
        self.n_class = n_class
        self.encoder = CLIPConditionEncoder(clip_name, d_model, image_size,
                                            freeze_clip, use_text=use_text)
        # `max_cond_len` PHẢI suy từ độ phân giải thật, không phải hằng số: ở 512px
        # memory là 1024 patch + 1 text, ở 1024px là 4096 + 1. Để hằng 1152 thì train
        # trên cache 1024px ném lỗi ngay batch đầu (may là lỗi rõ, không sai âm thầm).
        self.decoder = BoxDiT(d_model, n_layer, n_head, coord_dim,
                              dropout=dropout, roi_k=roi_k, n_class=n_class,
                              max_cond_len=self.encoder.num_patches + 128)
        self.num_timesteps = num_timesteps
        self.sampling_steps = sampling_steps
        self.snr_scale = snr_scale
        self.register_buffer("alphas_cumprod", cosine_alphas_cumprod(num_timesteps),
                             persistent=False)

    # ------------------------------------------------------------------ train

    def build_inputs(self, targets, num_proposals, valid_h, generator=None):
        """Dựng `x_t` cho cả batch. `t` là MỘT giá trị cho mỗi ảnh (như DiffusionDet).

        `device=dev` là BẮT BUỘC: nếu thiếu, `randint` tạo tensor CPU và đòi generator
        CPU, còn `prepare_diffusion_concat` tạo tensor CUDA và đòi generator CUDA — một
        generator không thể thoả cả hai.
        """
        dev = self.alphas_cumprod.device
        _check_generator(generator, dev)
        t = int(torch.randint(0, self.num_timesteps, (1,), device=dev,
                              generator=generator).item())
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
                patch_raw=None, text_raw=None, valid_h=None):
        """`x_t` [B,N,4] trong KHÔNG GIAN KHUẾCH TÁN -> list[(boxes [0,1], logits)]."""
        memory, praw = self.encoder(pixel_values, texts, patch_raw, text_raw,
                                    return_patch_raw=True)
        boxes_norm = decode_diffusion(x_t, self.snr_scale)
        return self.decoder(boxes_norm, timesteps, memory, praw, valid_h)

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def ddim_sample(self, num_proposals, pixel_values=None, texts=None,
                    patch_raw=None, text_raw=None, valid_h=None, eta=1.0,
                    generator=None, return_all_layers=False):
        """Sinh N box từ nhiễu thuần.

        `x_T ~ N(0, I)` với độ lệch chuẩn 1,0 — KHÔNG nhân `snr_scale` (lỗi số 3 của
        vòng 1). Mỗi bước: dự đoán `x_start` -> CLAMP -> tính lại `pred_noise` từ bản
        đã clamp.

        `valid_h` KÉO `cy` VỀ VÙNG ẢNH THẬT. Vòng 1 quên chỗ này: lúc train
        `build_inputs` có nhận `valid_h` (qua `make_placeholders`) nhưng lúc suy luận
        `x_T` chỉ là `randn` thuần, nên khoảng 30 % box khởi tạo rơi vào vùng đệm —
        một sai lệch train/inference. Vòng 1 chịu được vì chỉ lấy RoI một lần; A lấy 6
        lần mỗi khối nên phải sửa.

        `eta=1.0` là mặc định của DiffusionDet, khi đó DDIM suy biến thành DDPM. Đây là
        hệ quả ĐÃ ĐO chứ không phải lỗi: ở bước đầu (t=999 -> 749) sigma=0,925 và
        c=0,0000, nên `pred_noise` bị nhân 0 và cả bước trở thành
        `x_start*sqrt(ab_next) + noise`. Đặt `eta=0` để có DDIM tất định.
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
            boxes_norm = decode_diffusion(img, self.snr_scale)
            layers = self.decoder(boxes_norm, tb, memory, praw, valid_h)
            boxes, logits = layers[-1]           # tầng cuối là dự đoán dùng thật

            x_start = encode_diffusion(boxes, self.snr_scale)
            if t_next < 0:
                break
            pred_noise = predict_noise_from_start(img, t, x_start, self.alphas_cumprod)
            a, a_next = self.alphas_cumprod[t], self.alphas_cumprod[t_next]
            sigma = eta * ((1 - a / a_next) * (1 - a_next) / (1 - a)).clamp(min=0).sqrt()
            c = (1 - a_next - sigma ** 2).clamp(min=0).sqrt()
            img = x_start * a_next.sqrt() + c * pred_noise
            if eta > 0:
                img = img + sigma * torch.randn(img.shape, device=dev,
                                                generator=generator)

        if return_all_layers:
            return layers
        return boxes, logits


def build_model(cfg, dropout=None):
    """Config -> CELocDetector. MỘT chỗ dựng mô hình duy nhất cho mọi điểm vào.

    Vòng 1 có sáu script tự viết lại cùng một danh sách mười một tham số; đó đúng là
    cách một công cụ âm thầm dựng ra mô hình KHÁC với mô hình đang train.

    `dropout=0.0` do các công cụ eval/visualise truyền vào; train truyền None để lấy
    giá trị trong config.
    """
    m, d = cfg["model"], cfg["diffusion"]
    return CELocDetector(
        m["clip_name"], m["d_model"], m["n_layer"], m["n_head"],
        cfg["data"]["image_size"], d["num_timesteps"], d["snr_scale"],
        d["sampling_steps"], m["dropout"] if dropout is None else dropout,
        m["freeze_clip"], roi_k=m.get("roi_k", 3),
        n_class=m.get("n_class", 1), use_text=m.get("use_text", True),
        coord_dim=m.get("coord_dim", 64))
