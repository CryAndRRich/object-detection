"""CLIP ViT-B/16 FROZEN -> 1024 patch token + 1 text token.

Thay ResNet18 + SpatialSoftmax (nén cả ảnh thành MỘT vector 128-d) của CE-Loc
gốc. Đây là thay đổi cốt lõi của vòng 2: memory của decoder từ 2 token lên 1026
token CÓ VỊ TRÍ, để box "đọc" được ảnh tại đúng chỗ của nó.

BA THỨ BẮT BUỘC ĐỂ KHÔNG OOM (đã tính, xem docs §4.8b):
  1. `attn_implementation="sdpa"` — attention matrix của ViT ở 512px batch 32 là
     32x12x1025x1025 ~ 0,8 GB/layer nếu materialize.
  2. `.eval()` + `@torch.no_grad()` THẬT SỰ — `requires_grad=False` KHÔNG đủ,
     activation vẫn được lưu nếu input có grad -> 12 layer ~ 19 GB -> OOM.
     (Bản gốc `text_encoder.py:47` dùng `torch.set_grad_enabled(not requires_grad)`
      — logic ĐẢO NGƯỢC, là bug đã biết.)
  3. Nội suy `pos_embed` từ lưới 14x14 (pretrain 224px) lên 32x32 (512px).

VÌ SAO B/16 CHỨ KHÔNG B/32: vật CE-130 median 0,069 x 0,061 canvas. Ở patch 16
là 2,20 x 1,95 patch; ở patch 32 chỉ 1,10 x 0,98 -> 46,7 % vật NHỎ HƠN MỘT PATCH,
cross-attention không có gì để trỏ vào. B/32@1024px cho cùng 1024 token và cùng
chi phí, nhưng nội suy pos_embed 4,6x (lưới 7x7) so với 2,3x của B/16.

VÌ SAO CLIP CHỨ KHÔNG DINOv2: class giữa 3 split CE-130 RỜI NHAU HOÀN TOÀN
(train 72 / val 28 / test 28, giao = 0) -> bài toán thực chất là ZERO-SHOT. CLIP
là ứng viên duy nhất có không gian ngữ nghĩa chung ảnh-text để hiểu class chưa
từng thấy. Đây cũng là lý do text encoder BẮT BUỘC freeze.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CLIPConditionEncoder"]


class CLIPConditionEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch16", d_model=256,
                 image_size=512, freeze=True):
        super().__init__()
        from transformers import CLIPModel, CLIPTokenizer

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # SDPA cho CLIP chỉ có từ transformers >= 4.45; server đang chạy 4.42 nên
        # phải tự dò rồi lùi về "eager" thay vì hard-code. Với "eager" thì attention
        # matrix ĐƯỢC materialize (~0,8 GB/layer ở 512px batch 32) nên nếu memory
        # căng thì giảm batch_size hoặc nâng transformers.
        try:
            clip = CLIPModel.from_pretrained(model_name, attn_implementation="sdpa")
        except (ValueError, TypeError):
            clip = CLIPModel.from_pretrained(model_name, attn_implementation="eager")
            print("[clip_encoder] transformers chưa hỗ trợ SDPA cho CLIP -> dùng eager. "
                  "Attention matrix sẽ được materialize; nếu OOM thì giảm batch_size "
                  "hoặc nâng transformers >= 4.45.", flush=True)

        self.vision = clip.vision_model
        self.text = clip.text_model
        self.image_size = image_size
        self.patch = self.vision.config.patch_size
        self.grid = image_size // self.patch
        self.num_patches = self.grid ** 2

        if freeze:
            for p in self.vision.parameters():
                p.requires_grad = False
            for p in self.text.parameters():
                p.requires_grad = False
            self.vision.eval()
            self.text.eval()
        self.frozen = freeze

        self._resize_pos_embed()

        # Hai lớp chiếu HỌC ĐƯỢC — tối thiểu bắt buộc để nối chiều, KHÔNG phải
        # adapter (adapter transformer trên patch token đã hoãn có chủ đích).
        d_vis = self.vision.config.hidden_size
        d_txt = self.text.config.hidden_size
        self.proj_patch = nn.Linear(d_vis, d_model)
        # KHÔNG có Mish ở cuối như bản gốc — nó bóp méo không gian ngữ nghĩa CLIP,
        # đúng thứ cần giữ nguyên cho zero-shot.
        self.proj_text = nn.Linear(d_txt, d_model)

    def _resize_pos_embed(self):
        """Nội suy positional embedding 14x14 -> grid x grid (giữ token CLS)."""
        emb = self.vision.embeddings
        old = emb.position_embedding.weight.data          # [1+14*14, D]
        n_old = old.shape[0] - 1
        g_old = int(n_old ** 0.5)
        if g_old == self.grid:
            return

        cls_tok, patch_tok = old[:1], old[1:]
        patch_tok = patch_tok.reshape(1, g_old, g_old, -1).permute(0, 3, 1, 2)
        patch_tok = F.interpolate(patch_tok, size=(self.grid, self.grid),
                                  mode="bicubic", align_corners=False)
        patch_tok = patch_tok.permute(0, 2, 3, 1).reshape(self.num_patches, -1)

        new = torch.cat([cls_tok, patch_tok], dim=0)
        emb.position_embedding = nn.Embedding(new.shape[0], new.shape[1])
        emb.position_embedding.weight.data = new
        emb.position_embedding.weight.requires_grad = False
        emb.register_buffer("position_ids",
                            torch.arange(new.shape[0]).unsqueeze(0), persistent=False)
        emb.num_patches = self.num_patches
        emb.num_positions = new.shape[0]
        emb.image_size = self.image_size
        # transformers >= 4.4x kiểm tra kích thước ảnh theo config -> phải cập nhật
        # cả hai chỗ, nếu không nó chặn ở `modeling_clip.py:244`.
        self.vision.config.image_size = self.image_size
        if hasattr(emb, "config"):
            emb.config.image_size = self.image_size

    @torch.no_grad()
    def encode_image_raw(self, pixel_values):
        """[B,3,H,W] đã chuẩn hoá CLIP -> patch token thô [B, num_patches, d_vis].

        `no_grad` ở đây là thứ giữ memory ở ~1 GB thay vì ~19 GB.
        """
        out = self.vision(pixel_values=pixel_values).last_hidden_state
        return out[:, 1:]                                  # bỏ token CLS

    @torch.no_grad()
    def encode_text_raw(self, texts, device):
        """List[str] -> [B, 1, d_txt]. Input chỉ 1 từ nên pooled không mất gì."""
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        return self.text(**tok).pooler_output.unsqueeze(1)

    def forward(self, pixel_values=None, texts=None, patch_raw=None, text_raw=None):
        """Trả memory [B, 1 + num_patches, d_model] = [text; patch...].

        Nhận `patch_raw`/`text_raw` để dùng CACHE (bỏ hẳn chi phí ViT mỗi epoch).
        Time token do model ghép sau, ở đây chưa có.
        """
        if patch_raw is None:
            patch_raw = self.encode_image_raw(pixel_values)
        if text_raw is None:
            dev = patch_raw.device
            text_raw = self.encode_text_raw(texts, dev)

        patch = self.proj_patch(patch_raw.to(self.proj_patch.weight.dtype))
        text = self.proj_text(text_raw.to(self.proj_text.weight.dtype))
        return torch.cat([text, patch], dim=1)

    def train(self, mode=True):
        """Giữ CLIP ở eval kể cả khi model cha .train() — quan trọng vì BatchNorm/
        dropout của CLIP không được cập nhật theo dữ liệu CE-130."""
        super().train(mode)
        if self.frozen:
            self.vision.eval()
            self.text.eval()
        return self
