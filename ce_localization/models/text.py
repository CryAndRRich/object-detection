"""Token text: CLIP ViT-B/32 text frozen (như CE-Loc gốc), `pooler_output` 512-d.

CLIP đóng băng nên embedding của mỗi tên lớp là HẰNG SỐ -> tính MỘT lần lúc khởi động cho mọi
lớp cần dùng rồi giải phóng CLIP (không nằm trong model, không vào checkpoint). Cần
`export HF_HOME=/mnt/disk1/aiotlab/haitn/hf_cache` trên server.

⚠️ Mỗi ảnh CE-130 chỉ có MỘT lớp -> với bài detect, token text gần như không mang tín hiệu
(docs/EXPERIMENT_ALPHA.md mục 11). Vẫn giữ để đường ống giống CE-Loc.
"""

import torch

__all__ = ["encode_class_names", "TextTable"]


@torch.no_grad()
def encode_class_names(names, model_name="openai/clip-vit-base-patch32", device="cpu", batch=64, state_dict=None):
    """list[str] -> {tên: tensor [512] float32 trên CPU}. `state_dict`: weight CLIP text thay cho bản tải về (vd. CLIP
    lưu trong checkpoint CE-Loc gốc); chỉ tha `position_ids` (buffer, có / không tuỳ phiên bản transformers)."""
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name)
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        bad = [k for k in list(missing) + list(unexpected) if not k.endswith("position_ids")]
        if bad:
            raise RuntimeError(f"state_dict CLIP text lệch {model_name}: {bad[:10]}")
    model = model.to(device).eval()
    out = {}
    names = sorted(set(names))
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        inp = tok(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
        emb = model(**inp).pooler_output.float().cpu()
        out.update(zip(chunk, emb))
    del model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


class TextTable:
    """Tra embedding theo tên lớp, trả tensor [B,512] trên đúng device."""

    def __init__(self, table):
        self.table = table
        self.dim = next(iter(table.values())).numel()

    def __call__(self, names, device):
        missing = [n for n in names if n not in self.table]
        if missing:
            raise KeyError(f"thiếu embedding text cho {missing[:5]}")
        return torch.stack([self.table[n] for n in names]).to(device)
