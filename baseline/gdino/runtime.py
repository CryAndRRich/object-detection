"""Grounding DINO qua Open-GroundingDino (bản clone ghim commit ở `baseline/third_party/`, KHÔNG phải
`refs/repos/` — refs chỉ để đọc): kiểm repo, cài shim op, dựng model, nạp weight, suy luận một ảnh
với prompt = tên lớp của ảnh.

Score mỗi query = trung bình sigmoid(logit) trên các token của cụm tên lớp — đúng `PostProcess` của
Open-GroundingDino (positive map chuẩn hoá theo tổng). Box: cxcywh chuẩn hoá -> xyxy pixel ảnh gốc, kẹp
vào ảnh (như `detector_postprocess` của detectron2). Không NMS nội bộ: bộ chấm chung tự NMS.

⚠️ Open-GroundingDino KHÔNG có nhánh PyTorch khi thiếu CUDA op `MultiScaleDeformableAttention` (nó
raise ngay lúc import). Server không có `nvcc` nên không build được op -> `install_msda_fallback` cài
module giả rồi trỏ `MultiScaleDeformableAttnFunction` sang `multi_scale_deformable_attn_pytorch` có sẵn
trong chính file đó (cùng công thức, chậm hơn, tốn bộ nhớ hơn). Kaggle build được op thì không dùng shim.
"""

import argparse
import os
import subprocess
import sys
import types
import warnings

import numpy as np
import yaml

BASELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

__all__ = ["BASELINE_DIR", "load_config", "resolve", "check_repo", "clone_cmd", "setup_og",
           "install_msda_fallback", "og_args", "build_gdino", "load_weights", "make_transform", "prompt_of",
           "phrase_positive_map", "phrase_scores", "cxcywh_to_xyxy_px", "predict_image"]


def resolve(path):
    """Đường dẫn trong config gdino tương đối so với `baseline/`."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(BASELINE_DIR, path))


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clone_cmd(cfg):
    og = cfg["og"]
    repo = resolve(og["repo"])
    return f"git clone {og['url']} {repo} && git -C {repo} checkout {og['commit']}"


def check_repo(cfg):
    """-> đường dẫn repo Open-GroundingDino; thiếu hoặc lệch commit ghim thì dừng kèm lệnh sửa."""
    og = cfg["og"]
    repo = resolve(og["repo"])
    if not os.path.exists(os.path.join(repo, "main.py")):
        raise FileNotFoundError(f"chưa có Open-GroundingDino ở {repo}. Chạy:\n  {clone_cmd(cfg)}")
    head = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if head and head != og["commit"]:
        raise RuntimeError(f"{repo} đang ở commit {head}, config ghim {og['commit']}:\n"
                           f"  git -C {repo} fetch origin && git -C {repo} checkout {og['commit']}")
    return repo


def install_msda_fallback(module_name="models.GroundingDINO.ms_deform_attn"):
    """Không có CUDA op -> dùng bản PyTorch thuần của chính Open-GroundingDino. -> True nếu đã cài shim."""
    try:
        import MultiScaleDeformableAttention  # noqa: F401  (op đã build: dùng op thật)
        return False
    except Exception:  # ImportError, hoặc .so build lệch phiên bản CUDA
        pass
    sys.modules["MultiScaleDeformableAttention"] = types.ModuleType("MultiScaleDeformableAttention")
    import importlib
    msda = importlib.import_module(module_name)

    class _PytorchMSDA:
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return msda.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations,
                                                            attention_weights)

    msda.MultiScaleDeformableAttnFunction = _PytorchMSDA
    warnings.warn("MultiScaleDeformableAttention (CUDA op) chưa build -> dùng bản PyTorch thuần "
                  "(multi_scale_deformable_attn_pytorch): chậm hơn, cùng công thức")
    return True


def setup_og(repo):
    """Đưa repo vào sys.path (code Open-GroundingDino import `models`, `util`, `datasets` ở mức gốc)
    rồi cài shim nếu cần. -> True nếu đang dùng shim."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    return install_msda_fallback()


def og_args(cfg_file, options, argv=()):
    """Namespace y như `main.py` của Open-GroundingDino dựng: tham số dòng lệnh mặc định + config .py
    + `options` ghi đè."""
    from main import get_args_parser
    from util.slconfig import SLConfig

    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args(["-c", cfg_file, "--datasets", "unused.json", *argv])
    c = SLConfig.fromfile(cfg_file)
    c.merge_from_dict(dict(options))
    for k, v in c._cfg_dict.to_dict().items():
        setattr(args, k, v)
    return args


def build_gdino(cfg, device):
    """-> (model eval trên device, args, dùng_shim)."""
    repo = check_repo(cfg)
    shim = setup_og(repo)
    cfg_file = os.path.join(repo, cfg["og"]["config"])
    args = og_args(cfg_file, {"text_encoder_type": cfg["text_encoder"], "use_coco_eval": False,
                              "label_list": ["object"]})
    args.device = device
    from main import build_model_main

    model, _, _ = build_model_main(args)
    model.to(device).eval()
    return model, args, shim


def load_weights(model, path):
    """Nạp checkpoint (weight chính thức hoặc checkpoint finetune của main.py). -> epoch (None nếu không có)."""
    import torch
    from util.utils import clean_state_dict

    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                       # torch cũ chưa có weights_only
        ck = torch.load(path, map_location="cpu")
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    res = model.load_state_dict(clean_state_dict(state), strict=False)
    n_model = len(model.state_dict())
    print(f"  [weights] {path}: thiếu {len(res.missing_keys)} / {n_model} khoá, thừa {len(res.unexpected_keys)}"
          + (f" (vd thiếu {res.missing_keys[:3]})" if res.missing_keys else ""), flush=True)
    if len(res.missing_keys) > 0.05 * n_model:
        raise RuntimeError(f"{path}: thiếu {len(res.missing_keys)} khoá — sai weight / sai config model?")
    return ck.get("epoch") if isinstance(ck, dict) else None


def make_transform(args):
    """Transform val của Open-GroundingDino: cạnh ngắn max(data_aug_scales) = 800, cạnh dài ≤ 1333."""
    from datasets.coco import make_coco_transforms

    return make_coco_transforms("val", fix_size=False, strong_aug=False, args=args)


def prompt_of(name, fmt="{name} ."):
    """Tên lớp -> caption. Cùng dạng caption lúc train của Open-GroundingDino (' . '.join(...) + ' .')."""
    return fmt.format(name=" ".join(str(name).lower().split()))


def phrase_positive_map(tokenizer, caption, phrase, max_text_len=256):
    """-> tensor [max_text_len]: 1 trên các token của `phrase` trong `caption`, chia tổng (như PostProcess)."""
    from models.GroundingDINO.groundingdino import create_positive_map

    tokenized = tokenizer(caption, padding="longest", return_tensors="pt")
    pm = create_positive_map(tokenized, [0], [phrase], caption)[0][:max_text_len]
    if float(pm.sum()) == 0:
        raise ValueError(f"không tìm được token của {phrase!r} trong caption {caption!r}")
    return pm / pm.sum()


def phrase_scores(logits, pos_map):
    """logits [nq, T] (chưa sigmoid), pos_map [T] đã chuẩn hoá -> score [nq] = TB sigmoid trên token cụm từ."""
    return logits.sigmoid() @ pos_map.to(logits.dtype)


def cxcywh_to_xyxy_px(boxes, w, h):
    """cxcywh chuẩn hoá [0,1] -> xyxy pixel ảnh gốc, kẹp vào ảnh."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2,
                     b[:, 1] + b[:, 3] / 2], axis=1) * np.array([w, h, w, h], dtype=np.float64)
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
    return xyxy


def predict_image(model, transform, img_pil, caption, pos_map, device):
    """Một ảnh -> (box xyxy pixel ảnh gốc [nq,4], score [nq]); nq = 900 query, không lọc."""
    import torch
    from util.misc import nested_tensor_from_tensor_list

    w, h = img_pil.size
    img, _ = transform(img_pil, None)
    samples = nested_tensor_from_tensor_list([img]).to(device)
    with torch.no_grad():
        out = model(samples, captions=[caption])
    scores = phrase_scores(out["pred_logits"][0].float(), pos_map.to(device))
    return cxcywh_to_xyxy_px(out["pred_boxes"][0].float().cpu().numpy(), w, h), scores.cpu().numpy()
