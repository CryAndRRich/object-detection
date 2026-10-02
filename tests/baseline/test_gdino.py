"""Nhánh Grounding DINO (BASELINE3): dữ liệu ODVG, shim op khi thiếu CUDA, score theo cụm từ, box, lệnh train.
Không cần Open-GroundingDino thật: shim được thử trên một package `models.GroundingDINO.ms_deform_attn` giả
có đúng hành vi của file gốc (raise lúc import khi thiếu op, chọn nhánh qua MultiScaleDeformableAttnFunction)."""

import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest
import torch
import yaml

from baseline.gdino import runtime
from baseline.gdino.data import build_label_map, prepare
from baseline.gdino.runtime import (check_transformers, cxcywh_to_xyxy_px, install_msda_fallback,
                                    phrase_scores, prompt_of)
from baseline.gdino.train import OG_CFG_OVERRIDES, build_command, write_og_config
from ce_localization.data.dataset import scan_ce130
from tests.ce_localization.helpers import _fake_ce130

CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "baseline", "configs")

FAKE_MSDA = '''
try:
    import MultiScaleDeformableAttention as _C
except Exception:
    raise Exception("Wont work without MultiScaleDeformableAttention")


def multi_scale_deformable_attn_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    return ("pytorch", value)


class MultiScaleDeformableAttnFunction:
    @staticmethod
    def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
        return ("cuda", _C.ms_deform_attn_forward(value))


def forward(value):
    return MultiScaleDeformableAttnFunction.apply(value, None, None, None, None, 64)
'''


@pytest.fixture
def fake_og(tmp_path, monkeypatch):
    pkg = tmp_path / "og" / "models" / "GroundingDINO"
    pkg.mkdir(parents=True)
    (tmp_path / "og" / "models" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "ms_deform_attn.py").write_text(FAKE_MSDA)
    monkeypatch.syspath_prepend(str(tmp_path / "og"))
    for m in ("models", "models.GroundingDINO", "models.GroundingDINO.ms_deform_attn", "MultiScaleDeformableAttention"):
        monkeypatch.delitem(sys.modules, m, raising=False)
    yield
    for m in ("models", "models.GroundingDINO", "models.GroundingDINO.ms_deform_attn", "MultiScaleDeformableAttention"):
        sys.modules.pop(m, None)


def test_shim_routes_to_pytorch_when_op_missing(fake_og, monkeypatch):
    # chặn import op THẬT (Kaggle build được op -> máy có sẵn): sys.modules[...] = None -> ImportError
    monkeypatch.setitem(sys.modules, "MultiScaleDeformableAttention", None)
    with pytest.warns(UserWarning, match="dùng bản PyTorch thuần"):            # cảnh báo có chủ đích: chậm hơn op CUDA
        assert install_msda_fallback() is True
    import models.GroundingDINO.ms_deform_attn as m
    assert m.forward("v") == ("pytorch", "v")


def test_shim_not_installed_when_op_present(fake_og, monkeypatch):
    op = type(sys)("MultiScaleDeformableAttention")
    op.ms_deform_attn_forward = lambda v: v
    monkeypatch.setitem(sys.modules, "MultiScaleDeformableAttention", op)
    assert install_msda_fallback() is False
    import models.GroundingDINO.ms_deform_attn as m
    assert m.forward("v") == ("cuda", "v")



def test_real_op_loads_in_fresh_process_without_torch(tmp_path):
    """.so của op link libtorch -> chỉ import được SAU `import torch`. launch.py gọi shim trong tiến trình mới
    chưa nạp torch: bản cũ rơi sang shim dù op đã build (Kaggle 2026-10-01). Op giả dưới đây bắt chước đúng
    điều kiện đó."""
    og = tmp_path / "og" / "models" / "GroundingDINO"
    og.mkdir(parents=True)
    (tmp_path / "og" / "models" / "__init__.py").write_text("", encoding="utf-8")
    (og / "__init__.py").write_text("", encoding="utf-8")
    (og / "ms_deform_attn.py").write_text(FAKE_MSDA, encoding="utf-8")
    ops = tmp_path / "ops"
    ops.mkdir()
    (ops / "MultiScaleDeformableAttention.py").write_text(
        "import sys\nif 'torch' not in sys.modules:\n    raise ImportError('libc10.so: cannot open shared object')\n"
        "def ms_deform_attn_forward(v):\n    return v\n", encoding="utf-8")
    project = os.path.join(os.path.dirname(__file__), "..", "..")
    code = ("import sys; from baseline.gdino.runtime import install_msda_fallback; "
            "assert 'torch' not in sys.modules; print(install_msda_fallback()); "
            "import models.GroundingDINO.ms_deform_attn as m; print(m.forward('v'))")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path / "og"), str(ops), os.path.abspath(project)]))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["False", "('cuda',", "'v')"], r.stdout


def test_check_transformers_rejects_v5(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="5.0.0"))
    with pytest.raises(RuntimeError, match="transformers<5"):
        check_transformers()
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__="4.46.3"))
    assert check_transformers() == "4.46.3"

def test_phrase_score_is_mean_sigmoid_over_phrase_tokens():
    logits = torch.randn(7, 256)
    pm = torch.zeros(256)
    pm[[2, 3, 4]] = 1.0
    s = phrase_scores(logits, pm / pm.sum())
    assert torch.allclose(s, logits.sigmoid()[:, [2, 3, 4]].mean(1), atol=1e-6)


def test_box_conversion_and_clip():
    b = cxcywh_to_xyxy_px([[0.5, 0.5, 0.2, 0.4], [0.95, 0.1, 0.3, 0.4]], 200, 100)
    assert np.allclose(b[0], [80, 30, 120, 70])
    assert np.allclose(b[1], [160, 0, 200, 30])            # kẹp vào ảnh


def test_prompt_matches_training_caption_format():
    assert prompt_of("Cement  Bag") == "cement bag ."
    assert prompt_of("cup", "{name}") == "cup"
    # caption train của ODVGDataset (một lớp): ' . '.join(['cup']) + ' .'
    assert prompt_of("cup") == " . ".join(["cup"]) + " ."


def test_prepare_odvg_and_internal_val(tmp_path):
    root = str(tmp_path / "all_phase2_V2")
    _fake_ce130(root)
    ds = prepare(root, str(tmp_path / "data"), internal_val_images=1, seed=0, log=lambda *a: None)
    meta = json.load(open(ds, encoding="utf-8"))
    tr, va = meta["train"][0], meta["val"][0]
    assert tr["dataset_mode"] == "odvg" and va["dataset_mode"] == "coco" and va["label_map"] is None
    label_map = json.load(open(tr["label_map"], encoding="utf-8"))
    assert label_map == {"0": "apple", "1": "egg"} == build_label_map(scan_ce130(root, "train"))
    lines = [json.loads(l) for l in open(tr["anno"], encoding="utf-8")]
    items = {it["image_id"]: it for it in scan_ce130(root, "train")}
    assert len(lines) == len(items)
    for ln in lines:
        assert os.path.exists(os.path.join(tr["root"], ln["filename"]))
        iid = ln["filename"].split("/")[-2].split("_b")[0]
        inst = ln["detection"]["instances"]
        assert len(inst) == len(items[iid]["boxes_xyxy_px"])
        for obj in inst:
            x1, y1, x2, y2 = obj["bbox"]
            assert 0 <= x1 < x2 <= ln["width"] and 0 <= y1 < y2 <= ln["height"]
            assert label_map[str(obj["label"])] == obj["category"] == items[iid]["text"]
    coco = json.load(open(va["anno"], encoding="utf-8"))
    assert len(coco["images"]) == 1 and coco["categories"] == [{"id": 0, "name": "object", "supercategory": "object"}]
    # PostProcess của Open-GroundingDino (use_coco_eval False) trả nhãn = chỉ số trong label_list -> category id của
    # val nội bộ phải trùng đúng chỉ số đó (bản cũ: use_coco_eval True -> bảng 80 lớp COCO -> IndexError)
    assert OG_CFG_OVERRIDES["use_coco_eval"] is False
    names = OG_CFG_OVERRIDES["label_list"]
    assert isinstance(names, list) and {c["id"]: c["name"] for c in coco["categories"]} == dict(enumerate(names))
    assert {a["category_id"] for a in coco["annotations"]} <= set(range(len(names)))


def test_build_command_batch_split_and_options(tmp_path):
    cfg = yaml.safe_load(open(os.path.join(CFG_DIR, "baseline3_2_gdino_finetune.yaml"), encoding="utf-8"))
    cmd, opt = build_command(cfg, "/out/data/c.py", "/out", "/out/data/datasets.json", nproc=2, num_workers=2,
                             python="py")
    assert cmd[:3] == ["py", "-m", "torch.distributed.run"] and "--nproc_per_node=2" in cmd
    assert opt["batch_size"] == 1 and opt["epochs"] == 13
    assert "batch_size=1" in cmd and "--pretrain_model_path" in cmd
    assert cmd[cmd.index("-c") + 1] == "/out/data/c.py"
    # list / bool không đi qua --options (DictAction: 'label_list=object' -> chuỗi) — nằm trong cfg dẫn xuất
    assert not set(OG_CFG_OVERRIDES) & set(opt) and not [a for a in cmd if a.startswith(("label_list", "use_coco"))]
    cmd1, opt1 = build_command(cfg, "/og", "/out", "d.json", nproc=1, num_workers=2, python="py")
    assert cmd1[0] == "py" and cmd1[1].endswith("launch.py") and opt1["batch_size"] == 2
    cfg["finetune"]["batch_total"] = 3
    with pytest.raises(ValueError):
        build_command(cfg, "/og", "/out", "d.json", nproc=2, num_workers=2)


def test_write_og_config_keeps_base_and_overrides(tmp_path):
    cfg = yaml.safe_load(open(os.path.join(CFG_DIR, "baseline3_2_gdino_finetune.yaml"), encoding="utf-8"))
    repo = tmp_path / "og"
    (repo / "config").mkdir(parents=True)
    (repo / "config" / "cfg_odvg.py").write_text("num_queries = 900\nuse_coco_eval = True\nlr = 0.0001\n",
                                                 encoding="utf-8")
    path = write_og_config(str(repo), cfg, str(tmp_path / "data"))
    ns = {}
    exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)      # SLConfig nạp .py y như vậy
    assert ns["num_queries"] == 900 and ns["lr"] == 0.0001                      # phần gốc giữ nguyên
    assert ns["use_coco_eval"] is False and ns["label_list"] == ["object"]      # ghi đè thắng (gán sau)
    assert (repo / "config" / "cfg_odvg.py").read_text(encoding="utf-8").count("use_coco_eval = True") == 1


def test_zero_shot_and_finetune_share_the_same_model():
    a = yaml.safe_load(open(os.path.join(CFG_DIR, "baseline3_1_gdino_zeroshot.yaml"), encoding="utf-8"))
    b = yaml.safe_load(open(os.path.join(CFG_DIR, "baseline3_2_gdino_finetune.yaml"), encoding="utf-8"))
    assert a["og"] == b["og"] and a["text_encoder"] == b["text_encoder"] and a["weights"] == b["weights"]
    assert a["finetune"] is None and b["finetune"]["batch_total"] == 2
    assert a["name"] == "BASELINE3.1" and b["name"] == "BASELINE3.2"
    assert runtime.resolve(a["og"]["repo"]).endswith(os.path.join("baseline", "third_party", "Open-GroundingDino"))
    assert "Open-GroundingDino" in runtime.clone_cmd(a) and a["og"]["commit"] in runtime.clone_cmd(a)
