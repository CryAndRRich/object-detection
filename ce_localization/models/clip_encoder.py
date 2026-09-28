"""CLIP ViT-B/16 ĐÓNG BĂNG -> memory `[text; patch...]` cho decoder.

Vì sao CLIP: các lớp của 3 split CE-130 rời nhau hoàn toàn (72/28/28, giao = 0) nên bài toán
là zero-shot; CLIP có không gian ảnh-chữ chung cho lớp chưa thấy. Vì vậy text encoder PHẢI
đóng băng.

Ba điều bắt buộc để không OOM:
  1. `attn_implementation="sdpa"` khi transformers hỗ trợ (>= 4.45), không thì `eager`.
  2. `.eval()` + `@torch.no_grad()` thật — chỉ `requires_grad=False` vẫn lưu activation
     (~19 GB cho 12 tầng).
  3. Nội suy `pos_embed` từ lưới 14x14 (pretrain 224px) lên lưới của `image_size`.
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
        try:
            clip = CLIPModel.from_pretrained(model_name, attn_implementation="sdpa")
        except (ValueError, TypeError):
            clip = CLIPModel.from_pretrained(model_name, attn_implementation="eager")
            print("[clip_encoder] transformers này chưa có SDPA cho CLIP -> dùng eager "
                  "(ma trận attention được vật chất hoá; OOM thì hạ batch).", flush=True)

        self.vision = clip.vision_model
        self.text = clip.text_model
        self.image_size = image_size
        self.grid = image_size // self.vision.config.patch_size
        self.num_patches = self.grid ** 2

        self.frozen = freeze
        if freeze:
            for p in list(self.vision.parameters()) + list(self.text.parameters()):
                p.requires_grad = False
            self.vision.eval()
            self.text.eval()

        self._resize_pos_embed()

        # Hai phép chiếu HỌC ĐƯỢC, chỉ để khớp chiều. Không có Mish phía sau như bản gốc:
        # nó bóp méo không gian ngữ nghĩa của CLIP, đúng thứ zero-shot cần giữ.
        self.proj_patch = nn.Linear(self.vision.config.hidden_size, d_model)
        self.proj_text = nn.Linear(self.text.config.hidden_size, d_model)

    def _resize_pos_embed(self):
        """Nội suy positional embedding 14x14 -> grid x grid (giữ token CLS)."""
        emb = self.vision.embeddings
        old = emb.position_embedding.weight.data                   # [1 + 14*14, D]
        g_old = int((old.shape[0] - 1) ** 0.5)
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
        # transformers >= 4.4x kiểm image_size với config -> phải sửa cả hai chỗ.
        self.vision.config.image_size = self.image_size
        if hasattr(emb, "config"):
            emb.config.image_size = self.image_size

    @torch.no_grad()
    def encode_image_raw(self, pixel_values):
        """[B,3,H,W] đã chuẩn hoá CLIP -> patch token THÔ [B, num_patches, 768] (bỏ CLS).

        Đây là `last_hidden_state`, tức luồng dư TRƯỚC `post_layernorm` — có vài kênh
        outlier rất lớn. Đưa thẳng vào MLP mà không chuẩn hoá thì dễ sụp.
        """
        return self.vision(pixel_values=pixel_values).last_hidden_state[:, 1:]

    @torch.no_grad()
    def encode_text_raw(self, texts, device):
        """List[str] -> [B, 1, 512]. Mỗi text là MỘT tên lớp nên pooling không mất gì."""
        tok = self.tokenizer(texts, padding=True, truncation=True,
                             return_tensors="pt").to(device)
        return self.text(**tok).pooler_output.unsqueeze(1)

    def forward(self, pixel_values=None, texts=None, patch_raw=None, text_raw=None,
                return_patch_raw=False):
        """-> memory [B, 1 + num_patches, d_model] = [text; patch...].

        Nhận `patch_raw` / `text_raw` từ CACHE để bỏ hẳn ViT khỏi vòng train.
        `return_patch_raw=True` trả thêm patch thô cho RoI (khỏi chạy ViT lần hai).
        """
        if patch_raw is None:
            patch_raw = self.encode_image_raw(pixel_values)
        if text_raw is None:
            text_raw = self.encode_text_raw(texts, patch_raw.device)
        patch = self.proj_patch(patch_raw.to(self.proj_patch.weight.dtype))
        text = self.proj_text(text_raw.to(self.proj_text.weight.dtype))
        memory = torch.cat([text, patch], dim=1)
        return (memory, patch_raw) if return_patch_raw else memory

    def train(self, mode=True):
        """CLIP luôn ở eval() kể cả khi mô hình cha gọi .train()."""
        super().train(mode)
        if self.frozen:
            self.vision.eval()
            self.text.eval()
        return self
