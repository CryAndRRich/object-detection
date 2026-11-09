"""Adapter theo HỌ VLM cho finetune SFT + LoRA (ZETA4.1 RePlan, ZETA8.1–8.3 — `docs/EXPERIMENT_ZETA.md` mục 14.1, 16): mọi thứ phụ thuộc
kiến trúc nằm ở đây, `vlm/sft.py` (dữ liệu / câu đích) và `vlm/finetune.py` (vòng train, LoRA, merge) dùng chung.

Giao diện: `load(dev, attn)` -> (model bf16, processor có `.tokenizer`); `skip` = tên module tháp ảnh / projector (không gắn LoRA);
`end` = token kết thúc lượt (có nhãn — model học dừng); `encode(proc, hf_msgs, image)` -> (id token prompt, tensor ảnh); `collate_mm`;
`hidden` / `head` (thân LLM -> last hidden state, đầu từ vựng — loss chỉ chiếu `lm_head` ở token có nhãn); `grad_ckpt(model)`;
`n_image_tokens(proc, wh)` (ước độ dài chuỗi để gom micro-batch); `check(proc, image)` (ảnh model thấy lúc train == lúc vLLM sinh).

Prompt: chat template CỦA MODEL trên hội thoại `hf_messages` (system chuỗi, user = [text, ảnh] theo đúng thứ tự gửi vLLM). vLLM nhận
nội dung dạng list với các template duyệt `message['content']` (cả 3 họ ở đây) ⇒ ảnh nằm đúng chỗ như template dựng. Đã đối chiếu code
model (2026-11-08): Qwen2.5-VL / Qwen3-VL (transformers), InternVL3.5 (`OpenGVLab/InternVL3_5-8B` @ 9bb6a56, `modeling_internvl_chat.py`,
`chat_template.jinja`, tile của README + `vllm/.../internvl.py`), LLaVA-OV-2 (`lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` @ 878cde7,
`modeling_llava_onevision2.py`, `processing_llava_onevision2.py`, `chat_template.jinja`). Pilot server kiểm lại số token ảnh train ==
số token vLLM báo (`in` của responses).
"""

import torch

from vlm.parse import qwen_resized_wh

__all__ = ["ADAPTERS", "make_adapter", "hf_messages", "QwenAdapter", "LlavaOV2Adapter", "InternVLAdapter", "internvl_tiles"]


def hf_messages(messages):
    """Lượt nội bộ (`prompt.build_messages`) -> hội thoại cho `apply_chat_template`: user = [text, {"type": "image"}] (ảnh trước nếu
    `image_first`) — CÙNG thứ tự `backends.chat_messages` gửi vLLM; system / assistant = chuỗi."""
    out = []
    for t in messages:
        if t["role"] != "user":
            out.append({"role": t["role"], "content": t["text"]})
            continue
        content = [{"type": "text", "text": t["text"]}]
        if t["image"]:
            content.append({"type": "image"})
            if t.get("image_first"):
                content.reverse()
        out.append({"role": "user", "content": content})
    return out


def _pixels(ip, name, key, override=None):
    """Giới hạn pixel của image processor kiểu Qwen2-VL: khoá truyền lúc gọi > thuộc tính > `size` (tuỳ bản transformers)."""
    if override is not None:
        return int(override)
    v = getattr(ip, name, None)
    return int(v) if v is not None else int(ip.size[key])


class QwenAdapter:
    """Qwen2.5-VL (RePlan) / Qwen3-VL: class model đọc từ `config.architectures`; processor HF (`image_grid_thw`); thân `model.model`,
    đầu `model.lm_head`, tháp ảnh `visual`. Ảnh = `smart_resize` (cạnh bội số patch × merge: 28 Qwen2.5, 32 Qwen3) với `max_pixels` /
    `min_pixels` = `engine.mm_processor_kwargs` của config suy luận, truyền lúc GỌI processor như vLLM làm (processor giữ nguyên ⇒
    bản lưu cạnh model merge y hệt checkpoint gốc) ⇒ ảnh lúc train == lúc sinh."""

    skip = ("visual",)
    end = "<|im_end|>"
    mm_keys = ("pixel_values", "image_grid_thw")
    token_keys = ("mm_token_type_ids",)       # theo từng token (Qwen3-VL cần cho M-RoPE) -> `collate` đệm cùng input_ids, câu trả lời = 0

    def __init__(self, cfg):
        self.cfg = cfg
        self.mm = dict((cfg.get("engine") or {}).get("mm_processor_kwargs") or {})
        self.chat_kwargs = dict(cfg.get("chat_template_kwargs") or {})

    def load(self, dev, attn="sdpa"):
        import transformers
        from transformers import AutoConfig
        arch = AutoConfig.from_pretrained(self.cfg["model"]).architectures[0]
        model = getattr(transformers, arch).from_pretrained(self.cfg["model"], torch_dtype=torch.bfloat16, attn_implementation=attn)
        return model.to(dev), self.processor()

    def processor(self):
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(self.cfg["model"])

    @staticmethod
    def grad_ckpt(model):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    @staticmethod
    def factor(proc):
        ip = proc.image_processor
        return int(ip.patch_size) * int(ip.merge_size)

    def resized_wh(self, proc, wh):
        ip = proc.image_processor
        return qwen_resized_wh(wh, self.factor(proc), _pixels(ip, "min_pixels", "shortest_edge", self.mm.get("min_pixels")),
                               _pixels(ip, "max_pixels", "longest_edge", self.mm.get("max_pixels")))

    def n_image_tokens(self, proc, wh):
        w, h = self.resized_wh(proc, wh)
        f = self.factor(proc)
        return (w // f) * (h // f)

    def call_kwargs(self):
        return self.mm

    def encode(self, proc, msgs, image):
        prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.chat_kwargs)
        enc = proc(text=[prompt], images=[image], return_tensors="pt", **self.call_kwargs())
        mm = {k: enc[k] for k in self.mm_keys}
        mm.update({k: enc[k][0].tolist() for k in self.token_keys if k in enc})
        return enc["input_ids"][0].tolist(), mm

    @staticmethod
    def collate_mm(mms):
        return {k: torch.cat([m[k] for m in mms]) for k in mms[0]}

    @staticmethod
    def hidden(model, inp):
        return model.model(**inp).last_hidden_state

    @staticmethod
    def head(model):
        return model.lm_head

    def check(self, proc, image):
        """Lưới patch processor trả × cỡ patch == `smart_resize` dự tính (`resized_wh`) — lệch là ảnh / toạ độ sai thang."""
        ip = proc.image_processor
        t, h, w = (int(v) for v in ip(images=[image], return_tensors="pt", **self.call_kwargs())["image_grid_thw"][0])
        got, want = (w * int(ip.patch_size), h * int(ip.patch_size)), tuple(self.resized_wh(proc, image.size))
        if got != want:
            raise RuntimeError(f"processor resize ảnh {image.size} -> {got} khác dự tính {want} — sai max_pixels / min_pixels / patch")
        return got


class LlavaOV2Adapter(QwenAdapter):
    """LLaVA-OneVision-2 (code remote, transformers >= 5.7): `LlavaOnevision2ForConditionalGeneration` = `model` (`visual` gồm cả
    merger + `language_model` Qwen3-8B, RoPE 1-D) + `lm_head`; ảnh = Qwen2VLImageProcessor (patch 14 × merge 2, smart_resize bội 28),
    token `<|image_pad|>` nở thành t·h·w / 4 vị trí, đầu vào thêm `patch_positions`. Processor của repo là lớp thường (không nhận khoá
    ảnh lúc gọi) ⇒ đặt `max_pixels` / `min_pixels` vào thuộc tính image processor (vLLM cũng truyền vào lúc dựng image processor)."""

    mm_keys = ("pixel_values", "image_grid_thw", "patch_positions")
    token_keys = ()                           # RoPE 1-D, forward không nhận mm_token_type_ids

    def load(self, dev, attn="sdpa"):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(self.cfg["model"], trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                            attn_implementation=attn)
        return model.to(dev), self.processor()

    def processor(self):
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(self.cfg["model"], trust_remote_code=True)
        ip = proc.image_processor
        for name, key in (("min_pixels", "shortest_edge"), ("max_pixels", "longest_edge")):
            if name in self.mm:
                setattr(ip, name, int(self.mm[name]))
                if isinstance(getattr(ip, "size", None), dict):
                    ip.size[key] = int(self.mm[name])
        return proc

    def call_kwargs(self):
        return {}


# ----------------------------------------------------------------------------- InternVL3.5

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_START, IMG_END, IMG_CONTEXT = "<img>", "</img>", "<IMG_CONTEXT>"


def _closest_ratio(aspect, ratios, w, h, size):
    """`find_closest_aspect_ratio` của InternVL (README / vLLM): tỉ lệ lưới gần nhất; hoà thì lấy lưới to hơn nếu ảnh đủ lớn."""
    best_diff, best = float("inf"), (1, 1)
    for r in ratios:
        diff = abs(aspect - r[0] / r[1])
        if diff < best_diff:
            best_diff, best = diff, r
        elif diff == best_diff and w * h > 0.5 * size * size * r[0] * r[1]:
            best = r
    return best


def internvl_grid(wh, max_num, min_num=1, size=448):
    """Lưới tile (cột, hàng) InternVL chọn cho ảnh cỡ `wh` (`dynamic_preprocess`)."""
    w, h = wh
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda r: r[0] * r[1])
    return _closest_ratio(w / h, ratios, w, h, size)


def internvl_tiles(image, max_num, min_num=1, size=448, thumbnail=True):
    """`dynamic_preprocess` của InternVL: resize về lưới (cột × size, hàng × size), cắt tile size × size theo hàng; > 1 tile thì thêm
    thumbnail (cả ảnh resize size × size)."""
    cols, rows = internvl_grid(image.size, max_num, min_num, size)
    resized = image.resize((cols * size, rows * size))
    tiles = [resized.crop(((i % cols) * size, (i // cols) * size, (i % cols + 1) * size, (i // cols + 1) * size)) for i in range(cols * rows)]
    if thumbnail and len(tiles) != 1:
        tiles.append(image.resize((size, size)))
    return tiles


def _repo_file(name, fname):
    """Đường dẫn một file của model (thư mục local hoặc cache HF, HF_HUB_OFFLINE)."""
    import os
    if os.path.isdir(name):
        return os.path.join(name, fname)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(name, fname)


def remote_v5_compat(cls):
    """Code remote InternVL (`InternVLChatModel`) không gọi `post_init()` trong `__init__` ⇒ transformers 5 thiếu
    `all_tied_weights_keys` lúc `from_pretrained` (ZETA8.2 2026-11-09). Model không buộc trọng số nào (lm_head của Qwen3-8B riêng,
    có trong checkpoint) ⇒ đặt rỗng ở mức class; instance nào tự có (post_init chuẩn) thì vẫn dùng của nó."""
    if "all_tied_weights_keys" not in vars(cls):
        cls.all_tied_weights_keys = {}
    return cls


class InternVLProcessor:
    """'Processor' tối thiểu cho InternVL (repo không có processor chia tile): tokenizer + biến ảnh (tile -> resize 448 bicubic ->
    tensor -> chuẩn hoá ImageNet, như `build_transform` của README / vLLM)."""

    def __init__(self, tokenizer, max_num, size=448):
        self.tokenizer, self.max_num, self.size = tokenizer, max_num, size

    def pixel_values(self, image):
        import torchvision.transforms as T
        from torchvision.transforms import InterpolationMode
        tf = T.Compose([T.Resize((self.size, self.size), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                        T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return torch.stack([tf(t) for t in internvl_tiles(image.convert("RGB"), self.max_num, size=self.size)])

    def save_pretrained(self, d):
        self.tokenizer.save_pretrained(d)


class InternVLAdapter:
    """InternVL3.5 (code remote `InternVLChatModel`): `vision_model` (InternViT) + `mlp1` (projector) + `language_model` (Qwen3-8B,
    `.model` thân, `.lm_head`). Ảnh -> tile 448 (tối đa `engine.hf_overrides.max_dynamic_patch` như vLLM, + thumbnail), mỗi tile 256
    token; chat template của model dựng `<image>\\n` đúng chỗ, rồi thay bằng `<img>` + `<IMG_CONTEXT>` × 256 × số tile + `</img>` (y vLLM).
    `hidden`: `extract_feature` (ViT -> pixel shuffle -> mlp1) chèn vào embedding ở vị trí `<IMG_CONTEXT>` (như `forward` của model)
    rồi chạy thân LLM."""

    skip = ("vision_model", "mlp1")
    end = "<|im_end|>"

    def __init__(self, cfg):
        self.cfg = cfg
        e = cfg.get("engine") or {}
        self.max_num = int((e.get("hf_overrides") or {}).get("max_dynamic_patch", 12))
        self.chat_kwargs = dict(cfg.get("chat_template_kwargs") or {})
        self.tok_per_tile = 256

    def load(self, dev, attn="sdpa"):
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(self.cfg["model"], trust_remote_code=True, use_fast=False)
        if not getattr(tok, "chat_template", None):                                # repo để template ở chat_template.jinja
            tok.chat_template = open(_repo_file(self.cfg["model"], "chat_template.jinja")).read()
        from transformers import AutoConfig
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        conf = AutoConfig.from_pretrained(self.cfg["model"], trust_remote_code=True)
        cls = remote_v5_compat(get_class_from_dynamic_module(conf.auto_map["AutoModel"], self.cfg["model"]))
        model = cls.from_pretrained(self.cfg["model"], config=conf, torch_dtype=torch.bfloat16, use_flash_attn=False, low_cpu_mem_usage=True)
        try:                                       # code remote ép LLM về 'eager' khi không có flash-attn -> đổi lại sdpa (cùng phép tính, nhanh hơn)
            model.language_model.set_attn_implementation(attn)
        except Exception as e:                     # noqa: BLE001 — bản transformers không hỗ trợ thì giữ eager
            print(f"[internvl] giữ attention {model.language_model.config._attn_implementation} ({type(e).__name__}: {e})", flush=True)
        model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CONTEXT)         # forward của model không tự đặt
        self.tok_per_tile = int(model.num_image_token)
        return model.to(dev), InternVLProcessor(tok, self.max_num)

    @staticmethod
    def grad_ckpt(model):
        model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.language_model.config.use_cache = False

    def n_tiles(self, wh):
        cols, rows = internvl_grid(wh, self.max_num)
        return cols * rows + (1 if cols * rows != 1 else 0)

    def n_image_tokens(self, proc, wh):
        return self.n_tiles(wh) * self.tok_per_tile

    def encode(self, proc, msgs, image):
        tok = proc.tokenizer
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.chat_kwargs)
        if prompt.count("<image>") != 1:
            raise ValueError(f"chat template InternVL dựng {prompt.count('<image>')} chỗ ảnh (cần 1)")
        pv = proc.pixel_values(image)
        prompt = prompt.replace("<image>", IMG_START + IMG_CONTEXT * (self.tok_per_tile * len(pv)) + IMG_END, 1)
        return tok(prompt, add_special_tokens=False)["input_ids"], {"pixel_values": pv}

    @staticmethod
    def collate_mm(mms):
        return {"pixel_values": torch.cat([m["pixel_values"] for m in mms])}

    @staticmethod
    def hidden(model, inp):
        ids = inp["input_ids"]
        lm = model.language_model
        emb = lm.get_input_embeddings()(ids)
        vit = model.extract_feature(inp["pixel_values"].to(emb.dtype))              # [tile, 256, C]
        sel = ids == model.img_context_token_id
        if int(sel.sum()) != vit.shape[0] * vit.shape[1]:
            raise ValueError(f"{int(sel.sum())} vị trí <IMG_CONTEXT> != {vit.shape[0]} tile × {vit.shape[1]} token")
        emb = emb.masked_scatter(sel[..., None], vit.reshape(-1, vit.shape[-1]).to(emb.dtype))
        return lm.model(inputs_embeds=emb, attention_mask=inp["attention_mask"]).last_hidden_state

    @staticmethod
    def head(model):
        return model.language_model.lm_head

    def check(self, proc, image):
        n = len(proc.pixel_values(image))
        if n != self.n_tiles(image.size):
            raise RuntimeError(f"InternVL: {n} tile khác dự tính {self.n_tiles(image.size)}")
        return image.size


ADAPTERS = {"qwen": QwenAdapter, "llava_ov2": LlavaOV2Adapter, "internvl": InternVLAdapter}


def make_adapter(cfg):
    name = (cfg.get("finetune") or {}).get("adapter", "qwen")
    if name not in ADAPTERS:
        raise ValueError(f"finetune.adapter {name!r} không thuộc {sorted(ADAPTERS)}")
    return ADAPTERS[name](cfg)
