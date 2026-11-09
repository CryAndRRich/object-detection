"""Finetune VLM (ZETA4.1 RePlan, ZETA8.1–8.3), CPU, model / tokenizer / processor giả cùng giao diện thật:
- câu đích (RePlan; JSON tác giả 0–1000) đọc lại đúng lỗ qua parser lúc chấm; lỗ đích tất định theo epoch; prompt train = đường suy luận
  (system + user, thứ tự text / ảnh); nhãn chỉ trên câu trả lời; loss chỉ trên token có nhãn = loss trên logits cả chuỗi;
- LoRA B = 0 ⇒ y hệt model gốc, merge = LoRA chưa gộp, bỏ tháp ảnh;
- trọn luồng (cộng dồn micro + gom theo độ dài = batch đủ, snapshot mỗi epoch, resume, config lệch -> dừng, merge chép file gốc);
- adapter InternVL (tile như `dynamic_preprocess`, `<IMG_CONTEXT>` × 256 × tile, chèn đặc trưng đúng vị trí), LLaVA-OV-2 (patch_positions,
  giới hạn pixel trong image processor); eval greedy + 29 câu sample -> 30 box; run.py --model / --name."""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from ce_localization.utils.config import load_config
from tests.baseline.helpers import fake_ce130
from tests.fakes import fake_cocount_review
from vlm import finetune as ft
from vlm.adapters import (IMG_CONTEXT, remote_v5_compat, InternVLAdapter, InternVLProcessor, LlavaOV2Adapter, QwenAdapter, hf_messages, internvl_grid,
                          internvl_tiles)
from vlm.backends import result
from vlm.lora import LoRALinear, add_lora, lora_parameters, merge_lora
from vlm.parse import REPLAN_SEP, parse_candidates, qwen_resized_wh
from vlm.prompt import build_messages, prompt_hash
from vlm.sft import answer_text, coord_box, collate, pick_hole, resized_box, target_text

_CD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vlm", "config", "server2")
CFG = os.path.join(_CD, "replan-7b-sft.yaml")                    # ZETA4.1 (prompt RePlan, toạ độ qwen_resized)
CFG8 = os.path.join(_CD, "qwen3-vl-4b-sft.yaml")                # ZETA8.1 (prompt tác giả, ask_k 1, norm1000, greedy + 29 sample)
CFG_IVL, CFG_OV2 = os.path.join(_CD, "internvl3.5-8b-sft.yaml"), os.path.join(_CD, "llava-ov2-8b-sft.yaml")   # ZETA8.2 / 8.3
END = "<|im_end|>"


# ----------------------------------------------------------------------------- tokenizer / processor / model giả

class CharTok:
    """Tokenizer giả: token đặc biệt (`SPECIAL`) một id, ký tự khác một id; chat template ChatML, mỗi phần ảnh -> `IMAGE` đúng chỗ."""
    SPECIAL, IMAGE = {}, ""
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

    def convert_tokens_to_ids(self, t):
        return self.SPECIAL[t]

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        out = ""
        for m in msgs:
            c = m["content"]
            body = c if isinstance(c, str) else "".join(self.IMAGE if x["type"] == "image" else x["text"] for x in c)
            out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")


class Tok(CharTok):
    """Kiểu Qwen2.5-VL / Qwen3-VL."""
    SPECIAL = {"<|im_end|>": 2, "<|image_pad|>": 3, "<|im_start|>": 4, "<|vision_start|>": 5, "<|vision_end|>": 6}
    IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"


class TokIVL(CharTok):
    """Kiểu InternVL3.5 (`chat_template.jinja`: ảnh -> '<image>\\n' đúng chỗ, không system mặc định)."""
    SPECIAL = {"<|im_end|>": 2, IMG_CONTEXT: 3, "<img>": 4, "</img>": 5, "<|im_start|>": 6}
    IMAGE, chat_template = "<image>\n", "jinja"

    def save_pretrained(self, d):
        open(os.path.join(d, "tokenizer_config.json"), "w").write("{}")


class IP:
    """image_processor giả kiểu Qwen2.5-VL: patch 14 × merge 2, smart_resize bội 28; giới hạn pixel nhận lúc gọi như HF."""
    patch_size, merge_size, min_pixels, max_pixels = 14, 2, 3136, 12845056

    def __call__(self, images, return_tensors="pt", min_pixels=None, max_pixels=None):
        w, h = qwen_resized_wh(images[0].size, 28, min_pixels or self.min_pixels, max_pixels or self.max_pixels)
        return {"image_grid_thw": torch.tensor([[1, h // 14, w // 14]])}


class Proc:
    tokenizer = Tok()
    image_processor = IP()

    def apply_chat_template(self, msgs, **kw):
        return self.tokenizer.apply_chat_template(msgs, **kw)

    def __call__(self, text, images, return_tensors="pt", **kw):
        _, gh, gw = (int(v) for v in self.image_processor(images, **kw)["image_grid_thw"][0])
        n = (gh // 2) * (gw // 2)
        ids = []
        for t in self.tokenizer.encode(text[0]):
            ids += [3] * n if t == 3 else [t]
        px = torch.from_numpy(np.asarray(images[0].resize((8, 8)), dtype=np.float32)).reshape(-1)[:192].repeat(4 * n, 1)[:, :12] / 255
        return {"input_ids": torch.tensor([ids]), "pixel_values": px, "image_grid_thw": torch.tensor([[1, gh, gw]])}

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
    monkeypatch.setattr(ft, "copy_aux_files", lambda model, out: [])                # không tải file gốc từ HF trong test


def _tiny_cfg(tmp_path, base=CFG, **fin):
    p = tmp_path / f"sft_tiny_{abs(hash((base, json.dumps(fin, sort_keys=True))))}.yaml"
    f = {"epochs": 2, "batch_size": 2, "micro_batch": 1, "lr": 1e-2, "warmup": 0.0, "num_workers": 0, "log_every": 1, "grad_ckpt": False,
         "lora": {"r": 4, "alpha": 8, "dropout": 0.0}, **fin}
    yaml.safe_dump({"base": os.path.abspath(base), "finetune": f}, open(p, "w"))
    return str(p)


def _img(w=150, h=100, seed=0):
    return Image.fromarray(np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8))


def _item(cfg, img, target):
    msgs = build_messages({"class": "cup"}, {"b64": True, "mime": None, "canvas_wh": img.size}, cfg)
    return {"image": img, "messages": hf_messages(msgs), "target": target}


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
    cfg, img = load_config(CFG), _img()
    ans = [target_text("cup", [1, 2, 30, 40]) + END, target_text("cup", [5, 6, 7, 8]) + END + "xyz"]
    items = [_item(cfg, img, t) for t in ans]
    assert [x["type"] for x in items[0]["messages"][0]["content"]] == ["image", "text"]   # RePlan: ảnh trước text như suy luận ZETA4
    b = collate(items, Proc(), QwenAdapter(cfg))
    tok = Tok()
    for j, t in enumerate(ans):
        lab = b["labels"][j][b["labels"][j] != -100].tolist()
        assert lab == tok.encode(t)                                                  # đúng câu trả lời, gồm <|im_end|>
        ids = b["input_ids"][j][b["attention_mask"][j].bool()].tolist()
        assert ids[-len(lab):] == lab and 3 in ids[:-len(lab)]                       # nối sau prompt có token ảnh
    assert b["n_label"] == sum(len(tok.encode(t)) for t in ans) and b["image_grid_thw"].shape == (2, 3)


class ProcTT(Proc):
    """Như Proc nhưng trả thêm `mm_token_type_ids` (1 ở token ảnh) như processor Qwen3-VL."""

    def __call__(self, text, images, return_tensors="pt", **kw):
        enc = super().__call__(text, images, return_tensors, **kw)
        enc["mm_token_type_ids"] = (enc["input_ids"] == 3).long()
        return enc


def test_collate_pads_token_type_ids():
    """ZETA8.1 (Qwen3-VL) vỡ vì thiếu `mm_token_type_ids`: phải đi kèm, cùng cỡ input_ids, 1 đúng ở token ảnh, câu trả lời / đệm = 0;
    processor không trả khoá này (Qwen2.5 cũ) thì không thêm."""
    cfg, img = load_config(CFG), _img()
    items = [_item(cfg, img, t) for t in (target_text("cup", [1, 2, 30, 40]) + END, target_text("cup", [5, 6, 7, 8]) + END + "xyz")]
    b = collate(items, ProcTT(), QwenAdapter(cfg))
    tt = b["mm_token_type_ids"]
    assert tt.shape == b["input_ids"].shape and torch.equal(tt, ((b["input_ids"] == 3) & b["attention_mask"].bool()).long())
    assert "mm_token_type_ids" not in collate(items, Proc(), QwenAdapter(cfg))


def test_merge_processor_without_save_pretrained(tmp_path, monkeypatch):
    """ZETA8.3: processor code remote của LLaVA-OV-2 không có `save_pretrained` -> merge vẫn xong, file processor chép từ bản gốc."""
    src, w = tmp_path / "orig", tmp_path / "w"
    src.mkdir()
    (w / "snapshots").mkdir(parents=True)
    for f in ("preprocessor_config.json", "processing_llava_onevision2.py", "model-00001.safetensors"):
        (src / f).write_text("x")
    torch.save({"lora": {}}, w / "snapshots" / "it00020.pth")
    model = SimpleNamespace(save_pretrained=lambda d, **k: (os.makedirs(d, exist_ok=True), open(os.path.join(d, "config.json"), "w")))
    monkeypatch.setattr(ft, "lora_model", lambda cfg, dev, ad: (model, SimpleNamespace(), 0))
    monkeypatch.setattr(ft, "load_lora", lambda m, sd: None)
    monkeypatch.setattr(ft, "merge_lora", lambda m: 0)
    out = tmp_path / "merged"
    a = SimpleNamespace(device="cpu", save_dir=str(w), merge="it00020", out=str(out))
    ft.merge(a, {**load_config(CFG), "model": str(src)})
    assert sorted(os.listdir(out)) == ["config.json", "preprocessor_config.json", "processing_llava_onevision2.py"]


def test_answer_loss_equals_full_logits_loss():
    """Chỉ chiếu lm_head ở vị trí có nhãn (tiết kiệm bộ nhớ) phải ra đúng loss + grad như loss HF trên logits cả chuỗi (đệm phải,
    độ dài câu trả lời khác nhau)."""
    torch.manual_seed(0)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (100, 150, 3), dtype=np.uint8))
    cfg = load_config(CFG)
    ans = [target_text("cup", [1, 2, 30, 40]) + END, target_text("cup", [5, 6, 7, 8]) + END + "xyz"]
    b = collate([_item(cfg, img, t) for t in ans], Proc(), QwenAdapter(cfg))
    inp = {k: v for k, v in b.items() if k != "n_label"}
    m = Tiny()
    ref = m(**inp).loss
    g_ref = torch.autograd.grad(ref, m.q_proj.weight)[0]
    got = ft.answer_loss(m, inp, QwenAdapter(cfg))
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
    for micro, group in ((1, False), (2, False), (1, True)):                           # cộng dồn / gom theo độ dài: cùng grad bước đầu
        spy.micro = (micro, group)
        d = tmp_path / f"m{micro}{group}"
        ft.main(["--config", _tiny_cfg(tmp_path, micro_batch=micro, group_by_length=group), "--save-dir", str(d), "--max-steps", "1"] + data)
        assert (d / "snapshots" / "it00001.pth").exists() and (d / "lengths.json").exists() == group
    for k in ((2, False), (1, True)):
        for g1, g2 in zip(grads[(1, False)], grads[k]):
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
    with pytest.raises(SystemExit, match="KHÔNG resume"):                             # đổi thứ quyết định kết quả -> không resume
        ft.main(["--config", _tiny_cfg(tmp_path, epochs=3, lr=5e-3), "--save-dir", str(w), "--resume"] + data)
    out = tmp_path / "merged"
    ft.main(["--config", cfg2, "--save-dir", str(w), "--merge", "ep002", "--out", str(out)])
    assert (out / "model.pt").exists() and (out / "preprocessor_config.json").exists()
    sd = torch.load(out / "model.pt", weights_only=False)
    assert not any("lora" in k for k in sd) and "q_proj.weight" in sd


def test_run_model_name_override(tmp_path, monkeypatch, capsys):
    from vlm import run as vlm_run
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    review = fake_cocount_review(str(tmp_path / "data" / "cocount"))
    vlm_run.main(["--config", CFG, "--pilot", "1", "--dry-run", "--dataset", "ce130", "--model", "/x/merged_ep001", "--name", "ep001",
                  "--out-root", str(tmp_path / "o"), "--turn-index", ti, "--samples-root", samples,
                  "--cocount-root", str(tmp_path / "data" / "cocount"), "--cocount-review", review])
    out = capsys.readouterr().out
    assert "ep001 = /x/merged_ep001" in out and os.path.join(str(tmp_path / "o"), "ep001") in out
    cfg = load_config(CFG)
    assert prompt_hash({**cfg, "model": "/x/merged_ep001"}) != prompt_hash(cfg)


# ----------------------------------------------------------------------------- ZETA8: prompt tác giả, 1 box / câu, 0–1000

@pytest.mark.parametrize("wh", [(683, 384), (150, 100), (5000, 4000)])
def test_author_target_parses_back_to_hole(wh):
    """Đích ZETA8 = JSON tác giả 1 candidate, toạ độ 0–1000 ⇒ parser lúc chấm (coord norm1000, câu greedy) đọc lại ≈ lỗ (<= 1 / 1000)."""
    cfg = load_config(CFG8)
    assert cfg["coord"] == "norm1000" and cfg["k"] == 30 and cfg["ask_k"] == 1 and cfg["icl"] == 0
    hole = [0.21 * wh[0], 0.4 * wh[1], 0.33 * wh[0], 0.58 * wh[1]]
    txt = answer_text("sheep", coord_box(hole, wh, cfg), cfg)
    assert json.loads(txt)["candidates"][0]["name"] == "sheep"
    got = parse_candidates(txt, cfg, wh, wh)
    assert len(got) == 1 and np.allclose(got[0]["box"], hole, atol=max(wh) / 1000 + 1e-6)
    assert coord_box([-5, -5, wh[0] + 9, 0], wh, cfg) == [0, 0, 1000, 1]                  # kẹp, cạnh >= 1


def test_author_prompt_messages_match_inference():
    """Prompt train ZETA8 = system tác giả + user (text RỒI ảnh, như `chat_messages` gửi vLLM), xin 1 box; RePlan: ảnh trước."""
    cfg, img = load_config(CFG8), _img(640, 480)
    msgs = _item(cfg, img, "")["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"] and isinstance(msgs[0]["content"], str)
    assert [x["type"] for x in msgs[1]["content"]] == ["text", "image"]
    assert "Canvas size: 640x480" in msgs[1]["content"][0]["text"] and "Return top 1 placement" in msgs[1]["content"][0]["text"]
    a = QwenAdapter(cfg)
    assert a.mm == {"max_pixels": 1048576}                                            # cùng giới hạn ảnh vLLM dùng lúc sinh
    w, h = a.resized_wh(Proc(), (4000, 3000))
    assert w * h <= 1048576 and w % 28 == 0 and a.check(Proc(), Image.new("RGB", (4000, 3000))) == (w, h)


def test_finetune_author_prompt_flow(tmp_path, tiny_hf):
    """Trọn luồng ZETA8 (prompt tác giả, system + user) với model giả: chạy, ghi snapshot, nhãn = JSON 1 candidate + <|im_end|>."""
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    w = tmp_path / "w8"
    ft.main(["--config", _tiny_cfg(tmp_path, base=CFG8, micro_batch=2), "--save-dir", str(w), "--turn-index", ti, "--samples-root", samples,
             "--gt-root", root, "--device", "cpu"])
    assert sorted(os.listdir(w / "snapshots")) == ["ep001.pth", "ep002.pth"]


def test_eval_greedy_plus_samples_30_boxes(tmp_path):
    """Eval ZETA8: backend trả câu greedy + 29 câu sample (mỗi câu JSON 1 candidate, 0–1000) ⇒ 30 box / ảnh, box[0] = câu greedy
    (score 1), câu hỏng bị bỏ (chấm lặp vòng đủ K); chấm đủ hai luật (top_score / min_cnll_30)."""
    from vlm.runner import run_dataset
    _, samples, ti = fake_ce130(str(tmp_path / "data"))
    croot = str(tmp_path / "cocount")
    review = fake_cocount_review(croot)
    cfg = load_config(CFG8)
    cfg.update(_path=CFG8, chunk=2)

    class Be:
        def batch(self, jobs):
            out = []
            for j in jobs:
                assert "Return top 1 placement" in j["messages"][-1]["text"] and len(j["messages"]) == 2
                texts = [json.dumps({"candidates": [{"name": "cup", "box": [100 + 10 * i, 100, 300 + 10 * i, 400], "score": 0.3}]})
                         for i in range(29)] + ["broken"]
                out.append(result(REPLAN_SEP.join(texts), 900, 1200, 0, 0.0, 0.1))
            return out
    ns = argparse.Namespace(dataset="ce130", split="test_all", turn_index=ti, samples_root=samples, cocount_root=croot,
                            cocount_review=review, cocount_min_obj=5, obj_size=["mean", "exemplar"], seed=0, limit=None, out=None)
    d = str(tmp_path / "o")
    run_dataset(cfg, ns, Be(), d, "test", prompt_hash(cfg), log=lambda *a: None)
    rows = [json.loads(x) for x in open(os.path.join(d, "test.jsonl"))]
    js = json.load(open(os.path.join(d, "test.json")))
    assert rows and all(r["n_valid"] == 29 and len(r["boxes"]) == 29 for r in rows)     # câu hỏng bị bỏ; chấm lặp vòng đủ K = 30
    for r in rows:
        W, H = r["orig_wh"]
        assert r["boxes"][0]["score"] == 1.0 and r["boxes"][1]["score"] is None
        assert np.allclose(r["boxes"][0]["box"], [0.1 * W, 0.1 * H, 0.3 * W, 0.4 * H])
    assert js["main"] == "top_score" and js["min_cnll_30"]["K"] == 30 and js["selected"]["n_short"] == len(rows)


class _LMBody:
    def __init__(self, lm):
        self.lm = lm

    def __call__(self, inputs_embeds, attention_mask):
        h = inputs_embeds + self.lm.q_proj(inputs_embeds)
        return SimpleNamespace(last_hidden_state=h + self.lm.down_proj(torch.relu(self.lm.up_proj(h))))


class FakeIVL(nn.Module):
    """Giao diện `InternVLChatModel` tối thiểu: vision_model / mlp1 / language_model (.model thân, .lm_head), extract_feature
    (tile -> 256 token), num_image_token, img_context_token_id; forward(labels) chèn đặc trưng như bản gốc (đối chiếu loss)."""
    num_image_token = 256

    def __init__(self):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.up_proj = nn.Linear(3, 16)                                  # tên trùng target nhưng phải bỏ (skip)
        self.mlp1 = nn.Sequential(nn.Linear(16, 16))
        self.language_model = nn.Module()
        lm = self.language_model
        lm.embed = nn.Embedding(120, 16)
        lm.q_proj, lm.up_proj, lm.down_proj = nn.Linear(16, 16), nn.Linear(16, 32), nn.Linear(32, 16)
        lm.lm_head = nn.Linear(16, 120)
        lm.get_input_embeddings = lambda: lm.embed
        lm.model = _LMBody(lm)
        lm.config = SimpleNamespace(use_cache=True)
        self.config = SimpleNamespace(use_cache=True)
        self.img_context_token_id = TokIVL.SPECIAL[IMG_CONTEXT]

    def extract_feature(self, pv):                                                  # [T,3,448,448] -> [T,256,16]
        x = nn.functional.adaptive_avg_pool2d(pv, 16).flatten(2).transpose(1, 2)    # 16 × 16 = 256 vị trí / tile
        return self.mlp1(self.vision_model.up_proj(x))

    def forward(self, input_ids, attention_mask, pixel_values, labels):
        emb = self.language_model.embed(input_ids).clone()
        sel = input_ids == self.img_context_token_id
        emb[sel] = self.extract_feature(pixel_values).reshape(-1, 16)
        logits = self.language_model.lm_head(self.language_model.model(emb, attention_mask).last_hidden_state)
        return SimpleNamespace(loss=nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 120), labels[:, 1:].reshape(-1)))

    def save_pretrained(self, d, **kw):
        os.makedirs(d, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(d, "model.pt"))


# ----------------------------------------------------------------------------- ZETA8.2 InternVL: tile / prompt / loss / trọn luồng

@pytest.mark.parametrize("wh,grid,n", [((448, 448), (1, 1), 1), ((896, 448), (2, 1), 3), ((4000, 400), (4, 1), 5),
                                       ((500, 1100), (1, 2), 3), ((900, 700), (2, 2), 5),
                                       ((400, 300), (1, 1), 1)])  # hoà tỉ lệ: ảnh đủ lớn -> lưới to hơn
def test_internvl_tiles_like_dynamic_preprocess(wh, grid, n):
    """Lưới = tỉ lệ gần nhất trong (i, j), i·j <= max_dynamic_patch 4; > 1 tile thì + thumbnail; mỗi tile 448²."""
    assert internvl_grid(wh, 4) == grid
    tiles = internvl_tiles(_img(*wh), 4)
    assert len(tiles) == n == InternVLAdapter(load_config(CFG_IVL)).n_tiles(wh) and all(t.size == (448, 448) for t in tiles)


def test_internvl_prompt_image_block_and_loss():
    cfg = load_config(CFG_IVL)
    a = InternVLAdapter(cfg)
    assert a.max_num == 4 and a.skip == ("vision_model", "mlp1")
    proc, img = InternVLProcessor(TokIVL(), a.max_num), _img(896, 448)
    tgt = answer_text("cup", coord_box([100, 100, 300, 300], img.size, cfg), cfg) + a.end
    ids, mm = a.encode(proc, _item(cfg, img, tgt)["messages"], img)
    assert mm["pixel_values"].shape == (3, 3, 448, 448)
    s = TokIVL()
    k = ids.index(4)                                                                 # <img> ngay sau text user, trước '\n<|im_end|>'
    assert ids[k + 1:k + 1 + 768] == [3] * 768 and ids[k + 769] == 5 and ids.count(3) == 768
    assert s.encode("within this canvas.")[0] in ids[:k] and ids[-len(s.encode("<|im_start|>assistant\n")):] == s.encode("<|im_start|>assistant\n")
    b = collate([_item(cfg, img, tgt), _item(cfg, _img(448, 448, 1), tgt[:-12] + a.end)], proc, a)
    assert b["pixel_values"].shape[0] == 3 + 1
    inp = {k: v for k, v in b.items() if k != "n_label"}
    torch.manual_seed(0)
    m = FakeIVL()
    ref = m(inp["input_ids"], inp["attention_mask"], inp["pixel_values"], inp["labels"]).loss
    got = ft.answer_loss(m, inp, a)
    torch.testing.assert_close(got, ref)
    gr = torch.autograd.grad(ref, m.language_model.q_proj.weight)[0]
    torch.testing.assert_close(torch.autograd.grad(ft.answer_loss(m, inp, a), m.language_model.q_proj.weight)[0], gr)
    bad = dict(inp, pixel_values=inp["pixel_values"][:2])
    with pytest.raises(ValueError, match="IMG_CONTEXT"):
        a.hidden(m, bad)


def test_internvl_finetune_flow_and_merge(tmp_path, monkeypatch):
    """Trọn luồng ZETA8.2 với model giả: LoRA chỉ trên LLM (bỏ vision_model / mlp1), 2 epoch, merge chép file gốc."""
    root, samples, ti = fake_ce130(str(tmp_path / "data"))
    cfg = load_config(CFG_IVL)
    monkeypatch.setattr(ft, "load_hf", lambda c, dev: (FakeIVL().to(dev), InternVLProcessor(TokIVL(), 4)))
    src = tmp_path / "orig"
    src.mkdir()
    for f in ("modeling_internvl_chat.py", "chat_template.jinja", "model-00001.safetensors"):
        (src / f).write_text("x")
    p = tmp_path / "ivl.yaml"
    f = {"epochs": 2, "batch_size": 2, "micro_batch": 1, "lr": 1e-2, "warmup": 0.0, "num_workers": 0, "log_every": 1, "grad_ckpt": False,
         "lora": {"r": 4, "alpha": 8, "dropout": 0.0}}
    yaml.safe_dump({"base": os.path.abspath(CFG_IVL), "model": str(src), "finetune": f}, open(p, "w"))
    w = tmp_path / "w"
    data = ["--turn-index", ti, "--samples-root", samples, "--gt-root", root, "--device", "cpu"]
    ft.main(["--config", str(p), "--save-dir", str(w)] + data)
    assert sorted(os.listdir(w / "snapshots")) == ["ep001.pth", "ep002.pth"]
    lora = torch.load(w / "snapshots" / "ep002.pth", weights_only=False)["lora"]
    assert lora and all(k.startswith("language_model.") for k in lora)
    out = tmp_path / "merged"
    ft.main(["--config", str(p), "--save-dir", str(w), "--merge", "ep002", "--out", str(out)])
    assert sorted(os.listdir(out)) == ["chat_template.jinja", "model.pt", "modeling_internvl_chat.py", "tokenizer_config.json"]
    assert cfg["finetune"]["adapter"] == "internvl"


# ----------------------------------------------------------------------------- LLaVA-OV-2

class _IPOV2:
    patch_size, merge_size, min_pixels, max_pixels = 14, 2, 3136, 4000000

    def __init__(self):
        self.size = {"shortest_edge": 3136, "longest_edge": 4000000}

    def __call__(self, images, return_tensors="pt"):
        from vlm.parse import qwen_resized_wh
        w, h = qwen_resized_wh(images[0].size, 28, self.min_pixels, self.max_pixels)
        return {"image_grid_thw": torch.tensor([[1, h // 14, w // 14]])}


def test_llava_ov2_pixels_and_mm_keys(monkeypatch):
    """`max_pixels` của config (như ZETA6) đặt vào image processor (processor repo không nhận khoá lúc gọi); khoá ảnh có
    `patch_positions`; tháp ảnh `visual` (gồm merger) bỏ khỏi LoRA."""
    cfg = load_config(CFG_OV2)
    a = LlavaOV2Adapter(cfg)
    ip = _IPOV2()
    import transformers
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", classmethod(lambda cls, *x, **k: SimpleNamespace(image_processor=ip)))
    proc = a.processor()
    assert ip.max_pixels == 1048576 and ip.size["longest_edge"] == 1048576 and a.call_kwargs() == {}
    assert a.mm_keys == ("pixel_values", "image_grid_thw", "patch_positions") and a.skip == ("visual",)
    w, h = a.check(proc, _img(4000, 3000))
    assert w * h <= 1048576 and w % 28 == 0 and a.n_image_tokens(proc, (4000, 3000)) == (w // 28) * (h // 28)
    assert cfg["chat_template_kwargs"] == {"enable_thinking": False} and cfg["engine"]["trust_remote_code"]


def test_copy_aux_files(tmp_path):
    src, out = tmp_path / "s", tmp_path / "o"
    src.mkdir()
    out.mkdir()
    for f in ("config.json", "a.py", "tokenizer.json", "w.safetensors", "model.safetensors.index.json", "README.md"):
        (src / f).write_text(f)
    (out / "config.json").write_text("mới")
    assert ft.copy_aux_files(str(src), str(out)) == ["a.py", "tokenizer.json"]
    assert (out / "config.json").read_text() == "mới"                               # file save_pretrained vừa ghi không bị đè


def test_remote_v5_compat_sets_tied_keys_once():
    """InternVL code remote thiếu `all_tied_weights_keys` (không gọi post_init) -> đặt {} ở class; class đã có thì giữ nguyên."""
    class M(nn.Module):
        pass

    m = remote_v5_compat(M)()
    assert m.all_tied_weights_keys == {}
    M.all_tied_weights_keys = {"a": "b"}
    assert remote_v5_compat(M).all_tied_weights_keys == {"a": "b"}
