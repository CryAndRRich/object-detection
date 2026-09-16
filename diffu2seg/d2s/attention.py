r"""Stable Diffusion self-attention extraction (SD 1.5 / SD2).

COPIED from refs/repos/m2n2/src/stable_diffusion_2_attention_aggregator.py
(CVPR 2025, Karmann & Urfalioglu) and modified. That repo is read-only and
nothing here imports from it -- the file was copied into this sub-project and
edited, which is what the project rules allow.

WHY REUSE THIS PART RATHER THAN REWRITE IT: the hook is the fiddly bit. It
subclasses AttnProcessor2_0, re-implements scaled_dot_product_attention by hand
so it can intercept the tensor immediately AFTER torch.softmax and BEFORE
dropout, and then returns x untouched so the UNet proceeds exactly as if nobody
were listening. Getting the interception point wrong yields an attention matrix
that looks plausible and is not the one the model used.

WHAT A IS. Not a feature map. softmax(QK^T) is an (N, N) matrix over patch
tokens, where A[i, j] reads "how much does token i attend to token j" and each
row sums to 1. It is a GRAPH over the image, not a description of each point --
which is precisely why it can be propagated over.

FOUR CHANGES from the original:
  1. cv2 -> PIL. cv2 appeared only twice (resize here, imread in main) and
     dropping it removes a dependency the project does not otherwise need.
  2. `prompt_text` is now a parameter instead of a hard-coded ''. Stage 1 still
     passes '' (class-agnostic, as Diffuse2Seg specifies), but CE-130 ships a
     caption per image and the wiring is ready for the day that becomes a
     deliberate variable.
  3. extract_attention takes a LIST of timesteps and returns a list, so the
     paper's two-timestep blend is expressible. Stage 1 passes exactly one.
  4. main() removed; a bad processor type raises instead of calling exit().

WORKS FOR BOTH SD1.5 AND SD2 UNCHANGED. Measured from their unet/config.json:
both have sample_size 64 and block_out_channels [320, 640, 1280, 1280], and both
end in three CrossAttnUpBlock2D -- so `up_blocks.3.attentions.{0,1,2}` names the
same layers in each. They differ in cross_attention_dim (768 vs 1024) and
attention_head_dim (8 vs 5), and NEITHER touches this path: we hook `attn1`
(image<->image self-attention, never cross-attention) and average over all heads.
M2N2 confirms it -- their SD1 and SD2 aggregators differ in exactly two lines,
the default repo id and the default attention_resolution.

⚠️ The project now defaults to SD 1.5 because every `stabilityai/stable-diffusion-2*`
repo returns HTTP 401 (measured from three machines, 2026-09-15). See
config/base.py for the evidence.

DEFAULTS DIFFER from M2N2's, and each difference is a measurement:
  timestep             100 -> 150   Diffuse2Seg's value, TUNED FOR SD2 -- the
                                    SD1.5 timestep scale need not put the best
                                    features at the same place. Gate 0 should
                                    sweep a few values of t before this is fixed.
  attention_resolution 128 -> 64    A is N x N with N = r^2: 0.07 GB here
                                    versus 1.07 GB at r=128. r=64 also makes
                                    the input exactly 512 px, the project's
                                    canonical canvas.
  up_block weights     unchanged at 0.5/0.5 -- M2N2 measured this (Table 1 on
                                    DAVIS: up_0 alone 6.90, up_1 alone 7.10,
                                    together 6.72 NoC90; the down blocks are
                                    far worse at 15.25 / 13.18).
"""

import math
import os
from math import sqrt
from typing import Optional

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.models.attention_processor import Attention, AttnProcessor2_0
from diffusers.utils import deprecate

__all__ = ["StableDiffusion2AttentionAggregator", "sd2_inject_attention_wrappers"]


class AttnProcessor2_0Wrapper(AttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    # Trên ngưỡng này thì SDPA chạy từng head một (xem
    # scaled_dot_product_attention). Là thuộc tính lớp chứ không phải hằng số
    # viết thẳng, để test hạ được ngưỡng và chạy ĐÚNG nhánh chunked thay vì
    # chép lại vòng lặp — bản chép lại không bắt được lỗi trong code thật.
    CHUNK_THRESHOLD_ELEMS = 4e8

    def __init__(self, other, path=None, callback_func=None):
        super().__init__()

        # copy all members of the class we want to wrap
        self.__dict__ = other.__dict__.copy()

        # Adding our own members for tracking
        self.path = path
        self.wrapper_callback_func = callback_func

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
        # attn_bias chỉ cần khi có causal mask hoặc attn_mask. Dựng vô điều kiện
        # tốn thêm (L, S) = 0,77 GB fp16 ở N=19600 cho một tensor toàn số 0.
        attn_bias = None
        if is_causal or attn_mask is not None:
            attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if is_causal:
            assert attn_mask is None
            temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(query.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # ⚠️ BỘ NHỚ — ĐÂY LÀ CHỖ VỠ Ở r=140, KHÔNG PHẢI Ở A.
        #
        # Hook này thay scaled_dot_product_attention của PyTorch bằng bản viết
        # tay, vì chỉ có thế mới chặn được tensor NGAY SAU softmax và TRƯỚC
        # dropout. Cái giá là attn_weight (B, H, N, N) phải tồn tại tường minh,
        # trong khi SDPA gốc của PyTorch không bao giờ dựng nó.
        #
        # Ở N=19600 (r=140), 8 head fp16: MỖI bản attn_weight là 6,1 GB. Bản
        # gốc của M2N2 dựng bốn bản chồng nhau — matmul, +=, softmax, dropout —
        # cộng bản fp32 12,3 GB trong callback. Trên A30 24 GB thì OOM ngay ảnh
        # đầu tiên (đo 2026-09-15). Ở r=64 của M2N2 cùng đoạn code chỉ tốn
        # 0,27 GB/bản nên không ai thấy vấn đề.
        #
        # Ba sửa đổi, tất cả đều IN-PLACE để không nhân bản:
        #
        # ⚠️ TỪNG HEAD MỘT khi tensor lớn. Bản cũ dựng cả (B, H, N, N) cùng lúc
        # = 6,15 GB fp16 ở N=19600, H=8, và bản đó còn SỐNG suốt lúc callback
        # chạy (callback được gọi từ trong hàm này). Cộng bộ đệm fp32 của
        # callback thì đỉnh vượt 13,7 GB — quá nhiều khi GPU dùng chung chỉ
        # còn ~17 GB (đo 2026-09-16: OOM ở "Tried to allocate 1.43 GiB").
        #
        # Vòng theo head giữ đúng MỘT (N, N) fp16 sống mỗi lúc = 0,77 GB.
        # Kết quả GIỐNG HỆT: softmax chạy trên dim cuối nên độc lập theo head,
        # và `attn_weight @ value` cũng tách được theo head. Không phải xấp xỉ.
        #
        # Ngưỡng 4e8 phần tử ~ N=20000 một head: dưới mức đó đường cũ nhanh hơn
        # (một matmul lớn) và bộ nhớ không thành vấn đề, nên giữ nguyên.
        B, H, L_q = query.shape[0], query.shape[1], query.shape[-2]
        big = (B * H * L_q * key.shape[-2]) > self.CHUNK_THRESHOLD_ELEMS

        if not big:
            attn_weight = query @ key.transpose(-2, -1)
            attn_weight.mul_(scale_factor)      # thay `* scale_factor`
            if attn_bias is not None:
                attn_weight.add_(attn_bias)     # thay `+= attn_bias`
            torch.softmax(attn_weight, dim=-1, out=attn_weight)   # thay bản mới

            # Callback chạy TRƯỚC dropout -- đó là toàn bộ lý do hàm này tồn tại.
            if self.wrapper_callback_func is not None:
                attn_weight = self.wrapper_callback_func(self.path, attn_weight)

            # dropout_p = 0 ở suy luận, và torch.dropout vẫn cấp phát một bản
            # 6,1 GB dù p=0. Bỏ qua hẳn khi không dropout.
            if dropout_p > 0.0:
                attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
            result = attn_weight @ value
            self._finish_layer_if_any()
            return result

        out = torch.empty(B, H, L_q, value.shape[-1],
                          dtype=query.dtype, device=query.device)
        for b in range(B):
            for h in range(H):
                w = query[b, h] @ key[b, h].transpose(-2, -1)   # (N, N) fp16
                w.mul_(scale_factor)
                if attn_bias is not None:
                    w.add_(attn_bias)
                torch.softmax(w, dim=-1, out=w)

                # Callback nhận (1, 1, N, N) để giữ nguyên hợp đồng (B, H, N, N).
                if self.wrapper_callback_func is not None:
                    w = self.wrapper_callback_func(self.path, w[None, None])[0, 0]

                if dropout_p > 0.0:
                    w = torch.dropout(w, dropout_p, train=True)
                out[b, h] = w @ value[b, h]
                del w
        self._finish_layer_if_any()
        return out

    def _finish_layer_if_any(self):
        """Báo cho aggregator rằng layer này đã chạy xong hết các head.

        Aggregator cộng dồn tổng thô qua nhiều lần gọi callback (một lần cho cả
        tensor, hay một lần mỗi head), nên chỉ nó mới biết lúc nào chia trung
        bình. Wrapper biết ranh giới layer; aggregator biết phép tính.
        """
        fin = getattr(self.wrapper_callback_func, "__self__", None)
        if fin is not None and hasattr(fin, "_finish_layer"):
            fin._finish_layer()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = self.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


def _list_attention_paths(module, path='', out=None) -> set:
    """Mọi path Attention trong UNet, KHÔNG wrap gì cả.

    Dùng để kiểm layout: sd2_inject_attention_wrappers giờ chỉ wrap các block
    có trọng số khác 0, nên collect_wrappers không còn đủ để xác nhận rằng
    các block ta mong đợi thật sự tồn tại.
    """
    if out is None:
        out = set()
    if isinstance(module, Attention):
        out.add(path)
    elif hasattr(module, 'children'):
        for k, v in list(module.named_children()):
            _list_attention_paths(v, path + '.' + k, out)
    return out


def sd2_inject_attention_wrappers(module, callback_func=None, path='', collect_wrappers=None,
                                  active_paths=None) -> dict:
    """
    module: stable diffusion pipe.unet as input
    callback_func: Will be called whenever the self attention is used. Parameters are (path: str, x: torch.Tensor) and expects to return a
        torch.Tensor in the same shape, device and dtype as the given x. (to replace the current attention. Easiest is just to return original x to not modify)
    active_paths: nếu khác None, CHỈ wrap những path nằm trong tập này; mọi
        module khác giữ nguyên AttnProcessor2_0 gốc của PyTorch.

    ⚠️ BỘ NHỚ — vì sao active_paths tồn tại.
    Wrapper thay SDPA của PyTorch bằng bản viết tay, và bản viết tay BẮT BUỘC
    phải dựng attn_weight (B, H, N, N) tường minh (xem ghi chú ở
    scaled_dot_product_attention). SDPA gốc (flash attention) không bao giờ
    dựng tensor đó.

    UNet SD1.5 có 32 module Attention. Ở canvas 1120 (r=140):
      down_blocks.0 / up_blocks.3 : N = 19600 -> 6,15 GB mỗi bản fp16 8-head
      down_blocks.1 / up_blocks.2 : N =  4900 -> 0,38 GB
      các tầng còn lại            : không đáng kể
    Ta chỉ DÙNG 2 module (up_blocks.3.attentions.{0,1}). Wrap hết nghĩa là
    down_blocks.0.attentions.{0,1} và up_blocks.3.attentions.2 mỗi cái vẫn
    dựng 6,15 GB rồi bị callback vứt đi vì weight == 0.
    """
    if collect_wrappers is None:
        collect_wrappers = dict()

    if isinstance(module, Attention):
        if active_paths is not None and path not in active_paths:
            return collect_wrappers
        if not isinstance(module.processor, AttnProcessor2_0):
            # M2N2 called exit() here. Raising instead: a hard exit inside a
            # library call is untestable and kills a batch job with no context.
            raise RuntimeError(
                f"attention processor at {path!r} is "
                f"{module.processor.__class__.__name__}, expected AttnProcessor2_0. "
                "This usually means the installed diffusers version changed its "
                "default processor; pin diffusers==0.31.0."
            )
        module.set_processor(AttnProcessor2_0Wrapper(module.processor, path=path, callback_func=callback_func))
        collect_wrappers[path] = module.processor
        return collect_wrappers
    elif hasattr(module, 'children'):
        for k, v in list(module.named_children()):
            sd2_inject_attention_wrappers(v, callback_func, path + '.' + k, collect_wrappers,
                                          active_paths=active_paths)
    return collect_wrappers


def sd2_perform_single_image_diffusion_step(pipe, img: np.ndarray, timestep, device,
                                            torch_dtype, prompt_text=""):
    """One denoising step, run only for its side effect on the hooked attentions.

    img: (H, W, 3) uint8, side divisible by 64.

    NOTE the latent is NOT noised: vae.encode(...).latent_dist.mode() is the
    clean encoding, and `timestep` only tells the UNet which noise level to
    assume. That is what makes this a feature extractor rather than a generation
    step -- the structure read out of the attentions is the structure of THIS
    image, not of something being synthesised.

    prompt_text: "" reproduces Diffuse2Seg's class-agnostic setting ("we omit
    the text prompt and only pass null text embeddings"). CE-130 does carry a
    per-image caption, so this is left as a parameter rather than hard-coded.
    """
    prompt_embeds, _ = pipe.encode_prompt(prompt_text, device, 1, False)
    preprocessed_image = pipe.image_processor.preprocess(img / 255).to(torch_dtype).to(device)

    init_latents = pipe.vae.config.scaling_factor * pipe.vae.encode(preprocessed_image).latent_dist.mode()

    with torch.no_grad():
        _ = pipe.unet(
            init_latents,
            timestep,
            encoder_hidden_states=prompt_embeds,
            timestep_cond=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            return_dict=False,
        )[0]


class StableDiffusion2AttentionAggregator(object):
    """Runs one SD2 denoising step and returns the aggregated self-attention.

    Defaults are Diffu2Seg's, not M2N2's -- see the module docstring for why
    each one moved.
    """

    def __init__(self,
                 timestep=150,
                 attention_resolution=64,
                 weight_down_block_0=0.0,
                 weight_down_block_1=0.0,
                 weight_up_block_0=0.5,
                 weight_up_block_1=0.5,
                 weight_up_block_2=0.0,
                 hugging_face_model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
                 prompt_text="",
                 device='cuda:0',
                 torch_dtype=torch.float16,
                 extra_active_paths=None):
        self.stable_diffusion_img_size = (8 * attention_resolution, 8 * attention_resolution)
        self.attn_target_resolution = (attention_resolution, attention_resolution)
        self.current_merged_tensor = None
        # Bộ đệm thô của layer đang chạy — xem collect_attention_tensors_callback.
        self._raw_acc = None
        self._raw_heads = 0
        self._raw_weight = 0.0
        self.timestep = timestep
        self.prompt_text = prompt_text
        self.device = device
        self.torch_dtype = torch_dtype

        # safety_checker / feature_extractor are DELIBERATELY None. They exist to
        # screen GENERATED images; we generate nothing -- the UNet runs one step
        # with a hook reading self-attention, and no image ever leaves the VAE.
        # Loading them costs 1.2 GB for a component that is never called, so the
        # local weights directory does not ship them at all. Passing None here is
        # what makes that directory loadable.
        #
        # variant="fp16" picks *.fp16.safetensors / *.fp16.bin. The local dir
        # keeps only fp16 files, so without this from_pretrained looks for the
        # fp32 names and fails on a directory that is in fact complete.
        load_kwargs = dict(torch_dtype=torch_dtype, safety_checker=None,
                           feature_extractor=None, requires_safety_checker=False)
        try:
            try:
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    hugging_face_model_id, variant="fp16", **load_kwargs).to(device)
            except Exception:
                # A hub repo, or a local dir holding fp32 files, has no fp16
                # variant; retry without it rather than making the caller know
                # which kind of source they passed.
                self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                    hugging_face_model_id, **load_kwargs).to(device)
        except Exception as exc:
            # The server cannot reach HuggingFace: a PUBLIC repo comes back as
            # 401 / "Repository Not Found" / "Invalid username or password" even
            # though HF_TOKEN is empty and no token file exists (measured
            # 2026-09-15). There is no credential to be wrong, so the block is at
            # the network layer -- a token would not help.
            if os.path.isdir(hugging_face_model_id):
                raise RuntimeError(
                    f"Không load được SD2 từ thư mục local {hugging_face_model_id!r}. "
                    f"Thư mục tồn tại nhưng from_pretrained thất bại — nhiều khả "
                    f"năng thiếu file. Kiểm: model_index.json, tokenizer/, "
                    f"scheduler/, và BA file trọng số fp16 —\n"
                    f"  unet/diffusion_pytorch_model.fp16.safetensors\n"
                    f"  vae/diffusion_pytorch_model.fp16.bin (hoặc .safetensors)\n"
                    f"  text_encoder/model.fp16.safetensors\n"
                    f"safety_checker/ và feature_extractor/ KHÔNG cần (đã truyền None).\n"
                    f"Lỗi gốc: {exc}"
                ) from exc
            raise RuntimeError(
                f"Không load được SD2 từ {hugging_face_model_id!r}.\n"
                f"Nếu máy này không ra được HuggingFace (server aiotlab thì KHÔNG), "
                f"hãy tải SD2 ở local rồi đưa lên theo quy ước weights/:\n"
                f"    weights/diffu2seg/stable-diffusion-v1-5/\n"
                f"config.local_model_dir trỏ sẵn vào đó; có thư mục là tự dùng, "
                f"không cần token.\nLỗi gốc: {exc}"
            ) from exc
        # The five self-attention blocks at the two highest UNet resolutions.
        # Everything else gets weight 0 and is skipped before any reshape.
        named = {
            '.down_blocks.0.attentions.0.transformer_blocks.0.attn1': weight_down_block_0,
            '.down_blocks.0.attentions.1.transformer_blocks.0.attn1': weight_down_block_1,
            '.up_blocks.3.attentions.0.transformer_blocks.0.attn1': weight_up_block_0,
            '.up_blocks.3.attentions.1.transformer_blocks.0.attn1': weight_up_block_1,
            '.up_blocks.3.attentions.2.transformer_blocks.0.attn1': weight_up_block_2,
        }

        # ⚠️ Wrap CHỈ những block thực sự có trọng số khác 0. Một block bị wrap
        # là một tensor (B, H, N, N) tường minh, 6,15 GB ở r=140 cho các tầng
        # 19600 token — kể cả khi callback vứt nó đi ngay vì weight == 0.
        # extract_per_layer() cần từng layer riêng nên phải tính theo `named`,
        # không theo trọng số đang dùng của lần chạy này.
        active = {k for k, w in named.items() if w != 0}
        if extra_active_paths:
            active |= set(extra_active_paths)
        self._active_paths = active

        self.attention_wrappers = sd2_inject_attention_wrappers(
            self.pipe.unet, callback_func=self.collect_attention_tensors_callback,
            active_paths=active)

        self.path_to_weight_dict = dict()
        for attn_key in self.attention_wrappers.keys():
            self.path_to_weight_dict[attn_key] = named.get(attn_key, 0)

        # Kiểm tra layout trên TOÀN BỘ UNet, không chỉ các path đang wrap.
        # Nếu chỉ kiểm active thì một diffusers đổi cách đặt tên sẽ lọt câm
        # lặng: active toàn bộ tồn tại, còn block ta tưởng đang dùng thì không.
        all_attn_paths = _list_attention_paths(self.pipe.unet)
        missing = [k for k in named if k not in all_attn_paths]
        if missing:
            raise RuntimeError(
                f"expected SD2 attention blocks not found: {missing}. "
                "The UNet layout differs from the expected SD1.5/SD2 layout "
                "(up_blocks.3.attentions.{0,1,2})."
            )

    def collect_attention_tensors_callback(self, path, x: torch.Tensor):
        weight = self.path_to_weight_dict.get(path, 0)
        if weight == 0:
            return x

        # Trung bình theo head và batch, rồi reshape (N, N) -> (h, w, h, w).
        #
        # ⚠️ KHÔNG dùng `x.float()` trên cả tensor. M2N2 viết
        #     attn = torch.mean(torch.mean(x.float(), dim=1), dim=0)
        # và ở r=64 (N=4096) bản fp32 chỉ tốn 0,5 GB nên không ai để ý. Ở
        # r=140 của paper, x là (1, 8, 19600, 19600) fp16 và `.float()` dựng
        # một bản fp32 **12,3 GB** — nhân 3 block attention. OOM trên A30 24 GB
        # ngay ảnh đầu tiên (đo 2026-09-15: "Tried to allocate 11.45 GiB").
        #
        # Cộng dồn TỪNG HEAD vào một bộ đệm fp32 (N, N) = 1,54 GB thay vì dựng
        # (B, H, N, N) fp32. fp32 vẫn bắt buộc: g**(p-2) có số mũ ÂM và eps của
        # fp16 (~6e-8) sẽ tràn.
        # ⚠️ Callback có thể được gọi MỘT LẦN cho cả (B, H, N, N), hoặc TỪNG
        # HEAD một với (1, 1, N, N) khi SDPA chạy đường tiết kiệm bộ nhớ. Không
        # thể chia cho B*H của riêng lần gọi này: gọi từng head thì B*H == 1 và
        # kết quả sẽ gấp 8 lần.
        #
        # Nên cộng dồn TỔNG THÔ vào bộ đệm và đếm số head đã cộng; phép chia
        # trung bình + nhân trọng số dời sang finish_layer(), gọi sau khi UNet
        # chạy xong. Hai đường cho ra con số y hệt.
        # ⚠️ `.float()` chỉ SAO CHÉP khi dtype đổi. Nếu x đã là fp32 thì nó trả
        # về VIEW, và `self._raw_acc = head` khiến bộ đệm TRỎ THẲNG vào x. Các
        # head sau cộng vào bộ đệm là ghi đè lên x[0,0], rồi SDPA dùng chính x
        # đó cho `attn_weight @ value` -> head 0 sai, head khác đúng.
        # Đo 2026-09-16: output lệch 4.97 ở phần tử đầu, trùng khít ở phần tử
        # cuối. Trên đường chạy thật x là fp16 nên `.float()` có sao chép và
        # lỗi bị che — chỉ lộ ra khi test chạy fp32.
        # `.to(torch.float32, copy=True)` sao chép trong MỌI trường hợp.
        B, H = x.shape[0], x.shape[1]
        for b in range(B):
            for h in range(H):
                if self._raw_acc is None:
                    self._raw_acc = x[b, h].to(torch.float32, copy=True)
                else:
                    # add_ nhận trực tiếp view fp16, không cần bản fp32 tạm.
                    self._raw_acc.add_(x[b, h])
                self._raw_heads += 1
                self._raw_weight = weight
        return x

    def _finish_layer(self):
        """Chốt bộ đệm thô -> trung bình theo head, nhân trọng số, cộng vào tổng.

        Tách khỏi callback vì callback không biết nó nhận cả tensor hay từng
        head. Gọi sau mỗi lần UNet chạy xong một block có trọng số.
        """
        if self._raw_acc is None:
            return
        acc = self._raw_acc.div_(float(self._raw_heads))
        width = int(round(sqrt(acc.shape[-1])))
        attn = acc.reshape(width, width, width, width)

        # In-place: `a + b * w` dựng HAI tensor tạm, mỗi cái 1,54 GB ở r=140.
        # `acc` đã là bộ đệm riêng nên ghi đè nó là an toàn.
        if self.current_merged_tensor is None:
            self.current_merged_tensor = attn.mul_(self._raw_weight)
        else:
            self.current_merged_tensor.add_(attn.mul_(self._raw_weight))
        self._raw_acc = None
        self._raw_heads = 0

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """PIL instead of cv2. A no-op when the canvas is already 8 * r."""
        target = self.stable_diffusion_img_size
        if image.shape[:2] == (target[1], target[0]):
            return image
        return np.asarray(Image.fromarray(image).resize(target, Image.BILINEAR))

    def extract_attention_at(self, image: np.ndarray, timestep) -> torch.Tensor:
        """One timestep -> (h, w, h, w), last two axes summing to 1 per (i, j)."""
        self.current_merged_tensor = None
        sd2_perform_single_image_diffusion_step(
            pipe=self.pipe,
            img=self._resize(image),
            timestep=timestep,
            device=self.device,
            torch_dtype=self.torch_dtype,
            prompt_text=self.prompt_text,
        )
        merged = self.current_merged_tensor
        if merged is None:
            raise RuntimeError("no attention was captured; all block weights are 0")

        h, w = self.attn_target_resolution[1], self.attn_target_resolution[0]
        denom = torch.sum(merged.reshape(h, w, -1), dim=2)[:, :, None, None]
        return merged / denom

    def extract_per_layer(self, image: np.ndarray, timestep, paths=None):
        """Trích attention TỪNG layer, CHƯA chuẩn hoá. -> {path: (h,w,h,w)}

        VÌ SAO CẦN: `w1/w2` được áp TRONG callback, nên quét `w1` theo cách
        thường phải chạy lại SD cho mỗi giá trị — mà SD chiếm 2,3 s/ảnh. Trích
        một lần rồi trộn ngoài cho phép quét `w1` gần như miễn phí.

        ⚠️ TỐN BỘ NHỚ: giữ 2 tensor (h,w,h,w) fp32 cùng lúc = 3,1 GB ở r=140,
        thay vì 1,54 GB khi trộn ngay. Vẫn vừa A30 sau khi sửa OOM, nhưng đây
        là lý do hàm này KHÔNG phải đường mặc định.
        """
        if paths is None:
            paths = [k for k, w in self.path_to_weight_dict.items() if w != 0]

        # Chỉ block ĐANG ĐƯỢC WRAP mới gọi callback. Đặt weight=1.0 cho một
        # path không wrap sẽ không bắt được gì và hàm trả về thiếu layer trong
        # im lặng — dạng lỗi âm thầm đúng nghĩa. Bắt sớm, kèm cách sửa.
        not_wrapped = [p for p in paths if p not in self.attention_wrappers]
        if not_wrapped:
            raise RuntimeError(
                f"các block này không được wrap nên không trích được: {not_wrapped}.\n"
                f"Aggregator chỉ wrap block có trọng số khác 0 (để tiết kiệm "
                f"6,15 GB mỗi block ở r=140). Muốn trích thêm, truyền "
                f"extra_active_paths={not_wrapped!r} khi khởi tạo."
            )

        saved = dict(self.path_to_weight_dict)
        out = {}
        try:
            for path in paths:
                for k in self.path_to_weight_dict:
                    self.path_to_weight_dict[k] = 1.0 if k == path else 0.0
                # ⚠️ KHÔNG dùng extract_attention_at: nó CHUẨN HOÁ kết quả.
                #
                # Đường chạy thật trộn các layer TRƯỚC rồi chuẩn hoá TỔNG một
                # lần. Nếu ở đây chuẩn hoá từng layer rồi mới trộn, hai phép
                # không giao hoán và kết quả lệch (đo: max diff 7,8e-3 trên
                # tensor ngẫu nhiên) — bảng quét sẽ không so được với
                # run_paper.py. Nên trả về tensor THÔ, để người gọi trộn rồi
                # tự chuẩn hoá đúng thứ tự.
                self.current_merged_tensor = None
                sd2_perform_single_image_diffusion_step(
                    pipe=self.pipe, img=self._resize(image), timestep=timestep,
                    device=self.device, torch_dtype=self.torch_dtype,
                    prompt_text=self.prompt_text)
                if self.current_merged_tensor is None:
                    raise RuntimeError(f"không bắt được attention ở {path}")
                out[path] = self.current_merged_tensor
                self.current_merged_tensor = None
        finally:
            self.path_to_weight_dict = saved
        return out

    def extract_attention(self, image: np.ndarray, timesteps=None):
        """List of timesteps -> list of (h, w, h, w) tensors.

        Returns a list even for one timestep so callers need no special case.
        Stage 1 passes exactly one: blending two is a second variable, and the
        project rule is one variable per step.
        """
        if timesteps is None:
            timesteps = [self.timestep]
        elif isinstance(timesteps, (int, float)):
            timesteps = [timesteps]
        return [self.extract_attention_at(image, t) for t in timesteps]
