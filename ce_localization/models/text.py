"""Text: CLIP text frozen — `pooler_output` 512-d cho cond (`model.clip_text`, như CE-Loc gốc) + token chữ từng từ (`last_hidden_state`)
của CLIP text `refiner.text_model_name` cho refiner, kèm vân tay sha256 weight để kiểm khớp checkpoint. CLIP đóng băng ⇒ mỗi tên lớp
tính MỘT lần lúc khởi động rồi giải phóng CLIP (không vào model / checkpoint). Server: `export HF_HOME=...` trước.
"""

import torch

from ce_localization.utils.device import to_device

__all__ = ["encode_class_names", "encode_class_tokens", "TextTable"]


@torch.no_grad()
def _run_clip_text(names, model_name, device, batch, take, state_dict=None, fingerprint=False):
    """Nạp CLIP text `model_name` (+ weight `state_dict` thay bản tải về), chạy theo batch (`padding=True, truncation=True`) trên các
    tên khác nhau đã sắp xếp; `take(chunk, inp, out)` -> list (tên, tensor CPU). -> ({tên: tensor}, sha256 weight | None)."""
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name)
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        bad = [k for k in list(missing) + list(unexpected) if not k.endswith("position_ids")]
        if bad:
            raise RuntimeError(f"state_dict CLIP text lệch {model_name}: {bad[:10]}")
    sha = None
    if fingerprint:
        from ce_localization.models.clip_refiner import weights_sha256
        sha = weights_sha256(model)
    model = model.to(device).eval()
    out = {}
    names = sorted(set(names))
    for i in range(0, len(names), batch):
        chunk = names[i:i + batch]
        inp = tok(chunk, padding=True, truncation=True, return_tensors="pt").to(device)
        out.update(take(chunk, inp, model(**inp)))
    del model
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out, sha


def encode_class_names(names, model_name="openai/clip-vit-base-patch32", device="cpu", batch=64, state_dict=None):
    """list[str] -> {tên: tensor [512] float32 trên CPU} (`pooler_output`). `state_dict`: weight CLIP text thay cho bản tải về (vd.
    CLIP lưu trong checkpoint CE-Loc gốc); chỉ tha `position_ids` (buffer, có / không tuỳ phiên bản transformers)."""
    return _run_clip_text(names, model_name, device, batch, lambda chunk, inp, o: zip(chunk, o.pooler_output.float().cpu()),
                          state_dict)[0]


def encode_class_tokens(names, model_name="openai/clip-vit-base-patch16", device="cpu", batch=64):
    """list[str] -> ({tên: tensor [L_tên, H] float32 CPU — `last_hidden_state` của MỌI token thật (BOS .. EOS)}, sha256 weight CLIP
    text). Tokenize như refiner của tác giả (`padding=True, truncation=True`); CLIP text causal + đệm phải ⇒ token thật không phụ
    thuộc phần đệm / các tên khác trong batch ⇒ tính một lần mỗi tên = tính theo batch của tác giả."""
    def take(chunk, inp, o):
        hid, mask = o.last_hidden_state.float().cpu(), inp["attention_mask"].bool().cpu()
        for n, m in zip(chunk, mask):
            if not m[0]:
                raise ValueError(f"tên lớp {n!r}: token đầu không phải token thật (đệm trái?)")
        return [(n, h[m].clone()) for n, h, m in zip(chunk, hid, mask)]
    return _run_clip_text(names, model_name, device, batch, take, fingerprint=True)


class TextTable:
    """Tra embedding theo tên lớp, trả tensor [B,512] trên đúng device. `tokens` (DELTA): {tên: [L, H]} token chữ cho refiner,
    `token_sha`: vân tay (bytes 32) của CLIP text sinh ra chúng."""

    def __init__(self, table, tokens=None, token_sha=None):
        self.table = table
        self.dim = next(iter(table.values())).numel()
        self.token_table, self.token_sha = tokens, token_sha

    def __call__(self, names, device):
        missing = [n for n in names if n not in self.table]
        if missing:
            raise KeyError(f"thiếu embedding text cho {missing[:5]}")
        return to_device(torch.stack([self.table[n] for n in names]), device)

    def tokens(self, names, device):
        """-> (token [B, L_max, H] float32 đệm 0 bên phải, mask [B, L_max] bool True = token thật) trên `device`."""
        if self.token_table is None:
            raise ValueError("TextTable không có token chữ (model.refiner cần — train.build_text_table)")
        missing = [n for n in names if n not in self.token_table]
        if missing:
            raise KeyError(f"thiếu token chữ cho {missing[:5]}")
        seqs = [self.token_table[n] for n in names]
        L = max(s.shape[0] for s in seqs)
        pad = seqs[0].new_zeros(len(seqs), L, seqs[0].shape[1])
        mask = torch.zeros(len(seqs), L, dtype=torch.bool)
        for i, s in enumerate(seqs):
            pad[i, :s.shape[0]] = s
            mask[i, :s.shape[0]] = True
        return to_device(pad, device), to_device(mask, device)
