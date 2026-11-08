"""ZETA4.1 (SFT + LoRA planner RePlan), CPU, model / processor giả: câu đích đọc lại đúng lỗ qua parser ZETA4; lỗ đích tất định theo
epoch; nhãn chỉ trên câu trả lời; LoRA B = 0 ⇒ y hệt model gốc, merge = LoRA chưa gộp, bỏ tháp ảnh; trọn luồng finetune (cộng dồn
micro = batch đủ, snapshot mỗi epoch, resume, merge); run.py --model / --name."""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from ce_localization.utils.config import load_config
from tests.baseline.helpers import fake_ce130, fake_cocount
from vlm import finetune as ft
from vlm.lora import LoRALinear, add_lora, lora_parameters, merge_lora
from vlm.parse import parse_candidates, qwen_resized_wh
from vlm.prompt import prompt_hash
from vlm.sft import END, collate, pick_hole, resized_box, target_text

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vlm", "config", "server2", "replan-7b-sft.yaml")


# ----------------------------------------------------------------------------- processor / model giả

class Tok:
    """Mỗi ký tự một token; token đặc biệt một token."""
    SPECIAL = {"<|im_end|>": 2, "<|image_pad|>": 3, "<|im_start|>": 4, "<|vision_start|>": 5, "<|vision_end|>": 6}
    pad_token_id = 0

    def encode(self, text):
        out, i = [], 0
        while i < len(text):
            for s, v in self.SPECIAL.items():
                if text.startswith(s, i):
                    out.append(v)
                    i += len(s)
                    break
            else:
                out.append(10 + ord(text[i]) % 100)
                i += 1
        return out

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}


class Proc:
    tokenizer = Tok()

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        c = msgs[0]["content"]
        assert [x["type"] for x in c] == ["image", "text"]                       # ảnh trước text như suy luận ZETA4
        return f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{c[1]['text']}<|im_end|>\n<|im_start|>assistant\n"

    def __call__(self, text, images, return_tensors="pt"):
        w, h = qwen_resized_wh(images[0].size)
        n = (h // 28) * (w // 28)
        ids = []
        for t in self.tokenizer.encode(text[0]):
            ids += [3] * n if t == 3 else [t]
        px = torch.from_numpy(np.asarray(images[0].resize((8, 8)), dtype=np.float32)).reshape(-1)[:192].repeat(4 * n, 1)[:, :12] / 255
        return {"input_ids": torch.tensor([ids]), "pixel_values": px, "image_grid_thw": torch.tensor([[1, h // 14, w // 14]])}

    def save_pretrained(self, d):
        open(os.path.join(d, "preprocessor_config.json"), "w").write("{}")


class _Body:
    """`Tiny.model` (không phải nn.Module ⇒ không lặp tham số): thân -> `.last_hidden_state`."""

    def __init__(self, m):
        self.m = m

    def __call__(self, **kw):
        return SimpleNamespace(last_hidden_state=self.m.hidden(**kw))


class Tiny(nn.Module):
    """Giao diện Qwen2.5-VL tối thiểu: forward(input_ids, attention_mask, pixel_values, image_grid_thw, labels) -> .loss."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(120, 16)
        self.visual = nn.Module()
        self.visual.up_proj = nn.Linear(12, 16)                                  # tên trùng target nhưng phải bị bỏ (skip visual)
        self.q_proj, self.up_proj, self.down_proj = nn.Linear(16, 16), nn.Linear(16, 32), nn.Linear(32, 16)
        self.lm_head = nn.Linear(16, 120)
        self.config = SimpleNamespace(use_cache=True)

        self.model = _Body(self)                                                  # như HF: `.model` = thân, `.lm_head` = đầu từ vựng

    def hidden(self, input_ids, attention_mask, pixel_values, image_grid_thw):
        n = [int(t * hh * ww) for t, hh, ww in image_grid_thw.tolist()]               # patch của từng ảnh (như Qwen2.5-VL)
        v = torch.stack([x.mean(0) for x in self.visual.up_proj(pixel_values).split(n)])
        h = self.embed(input_ids) + v[:, None]
        h = h + self.q_proj(h)
        return h + self.down_proj(torch.relu(self.up_proj(h)))

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, labels=None):
        logits = self.lm_head(self.hidden(input_ids, attention_mask, pixel_values, image_grid_thw))
        loss = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 120), labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits)

    def save_pretrained(self, d, **kw):
        os.makedirs(d, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(d, "model.pt"))


@pytest.fixture
def tiny_hf(monkeypatch):
    monkeypatch.setattr(ft, "load_hf", lambda cfg, dev: (Tiny().to(dev), Proc()))


def _tiny_cfg(tmp_path, **fin):
    p = tmp_path / "sft_tiny.yaml"
    f = {"epochs": 2, "batch_size": 2, "micro_batch": 1, "lr": 1e-2, "warmup": 0.0, "num_workers": 0, "log_every": 1, "grad_ckpt": False,
         "lora": {"r": 4, "alpha": 8, "dropout": 0.0}, **fin}
    yaml.safe_dump({"base": os.path.abspath(CFG), "finetune": f}, open(p, "w"))
    return str(p)


# ----------------------------------------------------------------------------- câu đích / lỗ / nhãn

@pytest.mark.parametrize("wh", [(683, 384), (150, 100), (5000, 4000)])           # 5000×4000 vượt max_pixels ⇒ thu nhỏ
def test_target_parses_back_to_hole(wh):
    cfg = load_config(CFG)
    hole = [0.21 * wh[0], 0.4 * wh[1], 0.33 * wh[0], 0.58 * wh[1]]
    box = resized_box(hole, wh)
    txt = target_text("sheep", box) + END
    got = parse_candidates(txt, cfg, wh, wh)
    rw, _ = qwen_resized_wh(wh)
    assert len(got) == 1 and got[0]["score"] == 1.0                                # câu greedy = box chính
    assert np.allclose(got[0]["box"], hole, atol=wh[0] / rw + 1e-6)                 # sai số <= 1 px ảnh resized
    edge = resized_box([-5, -5, wh[0] + 9, 0.5], wh)                                 # kẹp trong ảnh, cạnh >= 1
    assert edge[0] == 0 and edge[1] == 0 and edge[3] - edge[1] >= 1 and edge[2] == rw


def test_pick_hole_deterministic_and_varies():
    a = [pick_hole(4, 0, 0, i) for i in range(200)]
    assert a == [pick_hole(4, 0, 0, i) for i in range(200)] and set(a) == {0, 1, 2, 3}
    assert a != [pick_hole(4, 0, 1, i) for i in range(200)] and pick_hole(1, 0, 3, 7) == 0


def test_collate_labels_only_answer():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (100, 150, 3), dtype=np.uint8))
    ans = [target_text("cup", [1, 2, 30, 40]) + END, target_text("cup", [5, 6, 7, 8]) + END + "xyz"]
    b = collate([{"image": img, "user": "U", "target": t} for t in ans], Proc())
    tok = Tok()
    for j, t in enumerate(ans):
        lab = b["labels"][j][b["labels"][j] != -100].tolist()
        assert lab == tok.encode(t)                                                  # đúng câu trả lời, gồm <|im_end|>
        ids = b["input_ids"][j][b["attention_mask"][j].bool()].tolist()
        assert ids[-len(lab):] == lab and 3 in ids[:-len(lab)]                       # nối sau prompt có token ảnh
    assert b["n_label"] == sum(len(tok.encode(t)) for t in ans) and b["image_grid_thw"].shape == (2, 3)


def test_answer_loss_equals_full_logits_loss():
    """Chỉ chiếu lm_head ở vị trí có nhãn (tiết kiệm bộ nhớ) phải ra đúng loss + grad như loss HF trên logits cả chuỗi (đệm phải,
    độ dài câu trả lời khác nhau)."""
    torch.manual_seed(0)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (100, 150, 3), dtype=np.uint8))
    ans = [target_text("cup", [1, 2, 30, 40]) + END, target_text("cup", [5, 6, 7, 8]) + END + "xyz"]
    b = collate([{"image": img, "user": "U", "target": t} for t in ans], Proc())
    inp = {k: v for k, v in b.items() if k != "n_label"}
    m = Tiny()
    ref = m(**inp).loss
    g_ref = torch.autograd.grad(ref, m.q_proj.weight)[0]
    got = ft.answer_loss(m, inp)
    g_got = torch.autograd.grad(got, m.q_proj.weight)[0]
    torch.testing.assert_close(got, ref)
    torch.testing.assert_close(g_got, g_ref)
    assert "labels" in inp                                                           # không sửa dict của người gọi


# ----------------------------------------------------------------------------- LoRA

def test_lora_identity_trainable_merge():
    torch.manual_seed(0)
    m = Tiny()
    x = {"input_ids": torch.randint(10, 100, (2, 7)), "attention_mask": torch.ones(2, 7, dtype=torch.long),
         "pixel_values": torch.rand(8, 12), "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]), "labels": torch.randint(10, 100, (2, 7))}
    y0 = m(**x).logits.detach()
    n = add_lora(m, ("q_proj", "up_proj", "down_proj"), r=4, alpha=8, skip=("visual",))
    assert n == 3 and not isinstance(m.visual.up_proj, LoRALinear)
    torch.testing.assert_close(m(**x).logits, y0)                                   # B = 0
    assert {id(p) for p in m.parameters() if p.requires_grad} == {id(p) for p in lora_parameters(m)}
    with torch.no_grad():
        for p in lora_parameters(m):
            p.normal_(0, 0.1)
    y1 = m(**x).logits.detach()
    assert (y1 - y0).abs().max() > 1e-3
    assert merge_lora(m) == 3 and isinstance(m.q_proj, nn.Linear) and not lora_parameters(m)
    torch.testing.assert_close(m(**x).logits, y1, rtol=1e-4, atol=1e-5)


# ----------------------------------------------------------------------------- trọn luồng

def test_finetune_flow_accum_resume_merge(tmp_path, tiny_hf, monkeypatch):
    root, samples, ti = fake_ce130(str(tmp_path / "data"))                            # 4 mẫu train (2 ảnh × 2 lượt)
    data = ["--turn-index", ti, "--samples-root", samples, "--gt-root", root, "--device", "cpu"]
    grads, clip = {}, torch.nn.utils.clip_grad_norm_

    def spy(params, *a, **k):                                                          # grad ngay trước bước tối ưu đầu
        grads.setdefault(spy.micro, [p.grad.clone() for p in params])
        return clip(params, *a, **k)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy)
    for micro in (1, 2):                                                              # cộng dồn micro 1 × 2 = một micro 2 (TB theo token)
        spy.micro = micro
        ft.main(["--config", _tiny_cfg(tmp_path, micro_batch=micro), "--save-dir", str(tmp_path / f"m{micro}"), "--max-steps", "1"] + data)
        assert (tmp_path / f"m{micro}" / "snapshots" / "it00001.pth").exists()
    for g1, g2 in zip(grads[1], grads[2]):
        torch.testing.assert_close(g1, g2, rtol=1e-4, atol=1e-6)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    w = tmp_path / "w"
    cfg2 = _tiny_cfg(tmp_path)
    ft.main(["--config", cfg2, "--save-dir", str(w)] + data)                           # 2 epoch × 2 bước
    assert sorted(os.listdir(w / "snapshots")) == ["ep001.pth", "ep002.pth"] and (w / "history.json").exists()
    st = torch.load(w / "last.pth", weights_only=False)
    assert (st["epoch"], st["iter"]) == (2, 4) and len(st["history"]) == 2
    e1, e2 = (torch.load(w / "snapshots" / f"ep00{i}.pth", weights_only=False)["lora"] for i in (1, 2))
    assert any((e1[k] - e2[k]).abs().max() > 0 for k in e1)
    ft.main(["--config", _tiny_cfg(tmp_path, epochs=3), "--save-dir", str(w), "--resume"] + data)
    st = torch.load(w / "last.pth", weights_only=False)
    assert (st["epoch"], st["iter"]) == (3, 6) and [h["epoch"] for h in st["history"]] == [1, 2, 3]
    out = tmp_path / "merged"
    ft.main(["--config", cfg2, "--save-dir", str(w), "--merge", "ep002", "--out", str(out)])
    assert (out / "model.pt").exists() and (out / "preprocessor_config.json").exists()
    sd = torch.load(out / "model.pt", weights_only=False)
    assert not any("lora" in k for k in sd) and "q_proj.weight" in sd


def test_run_model_name_override(tmp_path, monkeypatch, capsys):
    from vlm import run as vlm_run
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    review = fake_cocount(str(tmp_path / "data" / "cocount"))
    vlm_run.main(["--config", CFG, "--pilot", "1", "--dry-run", "--dataset", "ce130", "--model", "/x/merged_ep001", "--name", "ep001",
                  "--out-root", str(tmp_path / "o"), "--turn-index", ti, "--samples-root", samples,
                  "--cocount-root", str(tmp_path / "data" / "cocount"), "--cocount-review", review])
    out = capsys.readouterr().out
    assert "ep001 = /x/merged_ep001" in out and os.path.join(str(tmp_path / "o"), "ep001") in out
    cfg = load_config(CFG)
    assert prompt_hash({**cfg, "model": "/x/merged_ep001"}) != prompt_hash(cfg)
