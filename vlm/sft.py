"""SFT planner VLM cho bài ADD (ZETA4.1 RePlan, ZETA8.x Qwen3-VL / InternVL / LLaVA — `docs/EXPERIMENT_ZETA.md` mục 14.1, 16): dữ liệu,
câu đích, gom batch. Phần phụ thuộc kiến trúc (processor, token ảnh, chat template) ở `vlm/adapters.py`.

Mẫu = mẫu train CE-Loc (`baseline.data.train_items`: ảnh đã xoá vật 1..t, lỗ `holes[:t]` pixel ảnh gốc). Prompt = ĐÚNG đường suy luận:
`prompt.build_messages` (cùng hàm `run.py` dùng) rồi `adapters.hf_messages` (cùng thứ tự text / ảnh `backends.chat_messages` gửi vLLM).
Câu đích theo `cfg.prompt`, MỘT lỗ GT rút mỗi epoch (`pick_hole`, tất định theo seed / epoch / mẫu):
- `replan`: câu greedy THẬT của RePlan trên test (cùng think / gen_image / hint), chỉ thay `bbox_2d` (toạ độ `qwen_resized`);
- prompt tác giả (ZETA8, `ask_k` 1): JSON `{"candidates": [{"name": lớp, "box": [x1, y1, x2, y2], "score": 1.0}]}`, toạ độ theo
  `cfg.coord` (`norm1000` = int thang 0–1000 như cách các model này được huấn luyện) — đúng thứ `parse.parse_candidates` đọc lại.

Gom batch: prompt qua processor (ảnh -> token ảnh, như vLLM lúc sinh), câu đích tokenize RIÊNG rồi nối (đúng cách model tự sinh sau
prompt), nhãn = -100 trên prompt + đệm (đệm phải) ⇒ loss chỉ trên câu trả lời kể cả token kết thúc lượt (`adapter.end`).
"""

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.adapters import hf_messages
from vlm.backends import stable_int
from vlm.parse import qwen_resized_wh
from vlm.prompt import build_messages

__all__ = ["THINK", "HINT", "target_text", "coord_box", "resized_box", "answer_text", "pick_hole", "SFTDataset", "collate"]

THINK = ("The user's instruction is to add one more {cls} to the image. This can be achieved by inserting a new {cls} into the scene, "
         "ensuring it blends naturally with the existing {cls} and the environment.")
HINT = "add a new {cls} to the image"


def target_text(cls, box):
    """Câu trả lời planner RePlan với box int đã ở pixel ảnh resized."""
    b = ", ".join(str(int(v)) for v in box)
    return (f"<think>{THINK.format(cls=cls)}</think><gen_image>keep remaining part of image unchanged.</gen_image>"
            f'<region>[{{"bbox_2d": [{b}], "hint": "{HINT.format(cls=cls)}"}}]</region>')


def _to_canvas(box, wh, canvas):
    """Box xyxy pixel ảnh gốc cỡ `wh` -> int trên canvas (cw, ch), kẹp trong canvas, mỗi cạnh >= 1."""
    cw, ch = canvas
    sx, sy = cw / wh[0], ch / wh[1]
    x1, y1 = int(np.clip(round(box[0] * sx), 0, cw - 1)), int(np.clip(round(box[1] * sy), 0, ch - 1))
    return [x1, y1, int(np.clip(round(box[2] * sx), x1 + 1, cw)), int(np.clip(round(box[3] * sy), y1 + 1, ch))]


def resized_box(box, orig_wh, smart_resize=None):
    """Box pixel ảnh gốc -> int pixel ảnh Qwen thấy (`qwen_resized_wh`) — toạ độ `bbox_2d` của RePlan."""
    return _to_canvas(box, orig_wh, qwen_resized_wh(orig_wh, **(smart_resize or {})))


def coord_box(box, orig_wh, cfg):
    """Box pixel ảnh gốc -> toạ độ int theo `cfg.coord` (pixel | norm1000 | qwen_resized) — nghịch đảo của `parse.parse_candidates`."""
    coord = cfg.get("coord") or "pixel"
    if coord == "qwen_resized":
        return resized_box(box, orig_wh, cfg.get("smart_resize"))
    return _to_canvas(box, orig_wh, (1000, 1000) if coord == "norm1000" else orig_wh)


def answer_text(cls, box, cfg):
    """Câu trả lời đích (chưa có token kết thúc lượt) cho box đã ở toạ độ `cfg.coord`."""
    if cfg.get("prompt") == "replan":
        return target_text(cls, box)
    return json.dumps({"candidates": [{"name": cls, "box": [int(v) for v in box], "score": 1.0}]})


def pick_hole(n, seed, epoch, idx):
    """Chỉ số lỗ đích của mẫu `idx` ở `epoch` trong n lỗ (tất định, không phụ thuộc thứ tự / worker / resume)."""
    return stable_int(f"{seed}:{epoch}:{idx}", 8, "little") % n


class SFTDataset(Dataset):
    """`items` = `train_items(...)`; `cfg` = config VLM; `end` = token kết thúc lượt của adapter. `set_epoch` đổi lỗ đích."""

    def __init__(self, items, cfg, end):
        if cfg.get("icl") not in (None, 0, "0"):
            raise ValueError(f"SFT chỉ hỗ trợ zero-shot (icl 0), config có icl {cfg.get('icl')!r}")
        self.items, self.cfg, self.end, self.epoch = items, cfg, end, 0
        self.seed = int((cfg.get("finetune") or {}).get("seed", 0))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def image_size(self, i):
        with Image.open(self.items[i]["image"]) as im:            # chỉ đọc header
            return im.size

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        holes = np.asarray(it["holes"], dtype=np.float64).reshape(-1, 4)
        box = coord_box(holes[pick_hole(len(holes), self.seed, self.epoch, i)], img.size, self.cfg)
        msgs = build_messages({"class": it["class"]}, {"b64": True, "mime": None, "canvas_wh": img.size}, self.cfg)   # ảnh: chỉ đánh dấu
        return {"image": img, "messages": hf_messages(msgs), "target": answer_text(it["class"], box, self.cfg) + self.end}


def collate(batch, processor, adapter):
    """-> dict tensor cho forward (`input_ids`, `attention_mask`, tensor ảnh của adapter) + `labels` + `n_label` (số token có nhãn)."""
    tok = processor.tokenizer
    rows, mms = [], []
    for s in batch:
        p, mm = adapter.encode(processor, s["messages"], s["image"])
        tgt = tok(s["target"], add_special_tokens=False)["input_ids"]
        rows.append((p + tgt, [-100] * len(p) + tgt))
        mms.append(mm)
    L, pad = max(len(r[0]) for r in rows), tok.pad_token_id
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for j, (x, y) in enumerate(rows):
        ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    per_tok = {}
    for k in getattr(adapter, "token_keys", ()):           # khoá theo token của prompt (vd. mm_token_type_ids): đệm 0 tới L
        vals = [m.pop(k, None) for m in mms]
        if all(v is not None for v in vals):
            t = torch.zeros((len(rows), L), dtype=torch.long)
            for j, v in enumerate(vals):
                t[j, :len(v)] = torch.tensor(v)
            per_tok[k] = t
    return {"input_ids": ids, "attention_mask": att, **per_tok, **adapter.collate_mm(mms), "labels": lab,
            "n_label": int((lab[:, 1:] != -100).sum())}
