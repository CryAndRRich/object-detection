"""SFT planner RePlan cho bài ADD (ZETA4.1, docs/EXPERIMENT_ZETA.md mục 14.1): dữ liệu, câu đích, gom batch.

Mẫu = mẫu train CE-Loc (`baseline.data.train_items`: ảnh đã xoá vật 1..t, lỗ `holes[:t]` pixel ảnh gốc). Prompt y hệt suy luận ZETA4
(`vlm.prompt.replan_text`, một lượt user, ảnh trước text). Câu đích = câu greedy THẬT của RePlan trên test (cùng think / gen_image /
hint), chỉ thay `bbox_2d` = MỘT lỗ GT rút ngẫu nhiên mỗi epoch (`pick_hole`, tất định theo seed / epoch / mẫu), toạ độ int pixel ảnh
sau `smart_resize` (`vlm.parse.qwen_resized_wh` — đúng thứ `coord: qwen_resized` đọc lại).

Gom batch: prompt qua processor (ảnh -> token ảnh, như vLLM lúc suy luận), câu đích tokenize RIÊNG rồi nối (đúng cách model tự sinh
sau prompt), nhãn = -100 trên prompt + đệm (đệm phải) ⇒ loss chỉ trên câu trả lời kể cả `<|im_end|>` (token dừng).
"""

import hashlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from vlm.parse import qwen_resized_wh
from vlm.prompt import replan_text

__all__ = ["THINK", "HINT", "END", "target_text", "resized_box", "pick_hole", "SFTDataset", "collate"]

THINK = ("The user's instruction is to add one more {cls} to the image. This can be achieved by inserting a new {cls} into the scene, "
         "ensuring it blends naturally with the existing {cls} and the environment.")
HINT = "add a new {cls} to the image"
END = "<|im_end|>"


def target_text(cls, box):
    """Câu trả lời planner (định dạng RePlan) với box int đã ở pixel ảnh resized."""
    b = ", ".join(str(int(v)) for v in box)
    return (f"<think>{THINK.format(cls=cls)}</think><gen_image>keep remaining part of image unchanged.</gen_image>"
            f'<region>[{{"bbox_2d": [{b}], "hint": "{HINT.format(cls=cls)}"}}]</region>')


def resized_box(box, orig_wh, smart_resize=None):
    """Box xyxy pixel ảnh gốc -> int pixel ảnh model thấy (`qwen_resized_wh`), kẹp trong ảnh, mỗi cạnh >= 1."""
    rw, rh = qwen_resized_wh(orig_wh, **(smart_resize or {}))
    sx, sy = rw / orig_wh[0], rh / orig_wh[1]
    x1, y1 = int(np.clip(round(box[0] * sx), 0, rw - 1)), int(np.clip(round(box[1] * sy), 0, rh - 1))
    return [x1, y1, int(np.clip(round(box[2] * sx), x1 + 1, rw)), int(np.clip(round(box[3] * sy), y1 + 1, rh))]


def pick_hole(n, seed, epoch, idx):
    """Chỉ số lỗ đích của mẫu `idx` ở `epoch` trong n lỗ (tất định, không phụ thuộc thứ tự / worker / resume)."""
    h = hashlib.blake2b(f"{seed}:{epoch}:{idx}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "little") % n


class SFTDataset(Dataset):
    """`items` = `train_items(...)`; `cfg` = config VLM (instruction, smart_resize, finetune.seed). `set_epoch` đổi lỗ đích."""

    def __init__(self, items, cfg):
        self.items, self.cfg, self.epoch = items, cfg, 0
        self.seed = int((cfg.get("finetune") or {}).get("seed", 0))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = Image.open(it["image"]).convert("RGB")
        holes = np.asarray(it["holes"], dtype=np.float64).reshape(-1, 4)
        box = resized_box(holes[pick_hole(len(holes), self.seed, self.epoch, i)], img.size, self.cfg.get("smart_resize"))
        return {"image": img, "user": replan_text(self.cfg, it["class"]), "target": target_text(it["class"], box) + END}


def collate(batch, processor):
    """-> dict tensor cho `Qwen2_5_VLForConditionalGeneration.forward` + `labels` + `n_label` (số token có nhãn)."""
    tok = processor.tokenizer
    rows, pix, grid = [], [], []
    for s in batch:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": s["user"]}]}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        enc = processor(text=[prompt], images=[s["image"]], return_tensors="pt")
        tgt = tok(s["target"], add_special_tokens=False)["input_ids"]
        p = enc["input_ids"][0].tolist()
        rows.append((p + tgt, [-100] * len(p) + tgt))
        pix.append(enc["pixel_values"])
        grid.append(enc["image_grid_thw"])
    L, pad = max(len(r[0]) for r in rows), tok.pad_token_id
    ids = torch.full((len(rows), L), pad, dtype=torch.long)
    lab = torch.full((len(rows), L), -100, dtype=torch.long)
    att = torch.zeros((len(rows), L), dtype=torch.long)
    for j, (x, y) in enumerate(rows):
        ids[j, :len(x)], lab[j, :len(y)], att[j, :len(x)] = torch.tensor(x), torch.tensor(y), 1
    return {"input_ids": ids, "attention_mask": att, "pixel_values": torch.cat(pix), "image_grid_thw": torch.cat(grid), "labels": lab,
            "n_label": int((lab[:, 1:] != -100).sum())}
