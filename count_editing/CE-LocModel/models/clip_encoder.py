"""CLIP ViT-B/16 FROZEN -> 1024 patch tokens + 1 text token.

Replaces the original CE-Loc's ResNet18 + SpatialSoftmax (which compressed the
whole image into ONE 128-d vector). This is the core change of round 2: the
decoder's memory goes from 2 tokens to 1026 POSITIONED tokens, so a box can
"read" the image at its own location.

THREE THINGS REQUIRED TO AVOID OOM (computed, see docs §4.8b):
  1. `attn_implementation="sdpa"` — the ViT attention matrix at 512px batch 32 is
     32x12x1025x1025 ~ 0.8 GB/layer if materialized.
  2. `.eval()` + a REAL `@torch.no_grad()` — `requires_grad=False` is NOT enough,
     activations are still stored if the input has grad -> 12 layers ~ 19 GB -> OOM.
     (The original `text_encoder.py:47` uses `torch.set_grad_enabled(not
      requires_grad)` — INVERTED logic, a known bug.)
  3. Interpolating `pos_embed` from a 14x14 grid (224px pretrain) to 32x32 (512px).

WHY B/16 RATHER THAN B/32: CE-130 objects have median 0.069 x 0.061 of the canvas.
At patch 16 that is 2.20 x 1.95 patches; at patch 32 only 1.10 x 0.98 -> 46.7 % of
objects would be SMALLER THAN ONE PATCH, leaving cross-attention nothing to point
at. B/32@1024px would give the same 1024 tokens at the same cost, but needs 4.6x
pos_embed interpolation (7x7 grid) versus B/16's 2.3x.

WHY CLIP RATHER THAN DINOv2: the classes of the 3 CE-130 splits are COMPLETELY
DISJOINT (train 72 / val 28 / test 28, intersection = 0) -> the task is really
ZERO-SHOT. CLIP is the only candidate with a joint image-text semantic space able
to handle never-seen classes. This is also why the text encoder MUST be frozen.
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
        # SDPA for CLIP only exists from transformers >= 4.45; the server runs
        # 4.42, so detect and fall back to "eager" instead of hard-coding. With
        # "eager" the attention matrix IS materialized (~0.8 GB/layer at 512px
        # batch 32), so if memory gets tight, lower batch_size or upgrade
        # transformers.
        try:
            clip = CLIPModel.from_pretrained(model_name, attn_implementation="sdpa")
        except (ValueError, TypeError):
            clip = CLIPModel.from_pretrained(model_name, attn_implementation="eager")
            print("[clip_encoder] this transformers version has no SDPA for CLIP "
                  "-> using eager. The attention matrix will be materialized; if "
                  "you OOM, lower batch_size or upgrade transformers >= 4.45.",
                  flush=True)

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

        # Two LEARNABLE projections — the bare minimum needed to match dims, NOT
        # an adapter (a transformer adapter on patch tokens is deliberately
        # deferred).
        d_vis = self.vision.config.hidden_size
        d_txt = self.text.config.hidden_size
        self.proj_patch = nn.Linear(d_vis, d_model)
        # NO trailing Mish unlike the original — it distorts CLIP's semantic
        # space, exactly what must be preserved for zero-shot.
        self.proj_text = nn.Linear(d_txt, d_model)

    def _resize_pos_embed(self):
        """Interpolate the positional embedding 14x14 -> grid x grid (CLS kept)."""
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
        # transformers >= 4.4x validates image size against the config -> both
        # places must be updated, otherwise it rejects the input at
        # `modeling_clip.py:244`.
        self.vision.config.image_size = self.image_size
        if hasattr(emb, "config"):
            emb.config.image_size = self.image_size

    @torch.no_grad()
    def encode_image_raw(self, pixel_values):
        """[B,3,H,W] CLIP-normalised -> raw patch tokens [B, num_patches, d_vis].

        The `no_grad` here is what keeps memory at ~1 GB instead of ~19 GB.
        """
        out = self.vision(pixel_values=pixel_values).last_hidden_state
        return out[:, 1:]                                  # drop the CLS token

    @torch.no_grad()
    def encode_text_raw(self, texts, device):
        """List[str] -> [B, 1, d_txt]. Input is a single word, so pooling loses nothing."""
        tok = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        return self.text(**tok).pooler_output.unsqueeze(1)

    def forward(self, pixel_values=None, texts=None, patch_raw=None, text_raw=None):
        """Returns memory [B, 1 + num_patches, d_model] = [text; patches...].

        Accepts `patch_raw`/`text_raw` to use the CACHE (removing the per-epoch
        ViT cost entirely). The time token is prepended later by the model, not here.
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
        """Keep CLIP in eval even when the parent calls .train() — important
        because CLIP's BatchNorm/dropout must not adapt to CE-130 data."""
        super().train(mode)
        if self.frozen:
            self.vision.eval()
            self.text.eval()
        return self
