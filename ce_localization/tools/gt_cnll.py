#!/usr/bin/env python3
"""C-NLL của chính các BOX GT chỗ trống (docs/EXPERIMENT_GAMMA.md mục 18) — mốc để đọc C-NLL của model: box đáp án thật đạt bao nhiêu.

Cùng công thức + cùng vật tham chiếu như `eval.py` (`engine/add_eval.cnll`, F1 = [w, h, IoU lớn nhất với vật, TB khoảng cách tới tâm
3 vật gần nhất], F2 = [cx, cy, w, h]; Gaussian 4-D fit trên vật đang có; bỏ mẫu < CNLL_MIN_OBJ vật). Toạ độ pixel ảnh gốc — đặc trưng
chia (W, H) nên bất biến với letterbox của eval.
  CE-130 (ảnh inpaint lượt t, split `--split` theo `--split-source` như GAMMA2+: test samples = 4.948 mẫu): vật tham chiếu = vật CÒN
          trong ảnh ở lượt t; GT = lỗ mới nhất (`latest`, đích train) và mọi lỗ tới lượt t (`any`; `sel` = lỗ C-NLL nhỏ nhất mỗi mẫu)
  CE-CoCount (`--cocount-root`): vật tham chiếu = box vật CÙNG lớp (`Anno_with_exam_bbox`); GT = 10 `loc_bbox` (`any`, `sel`)
Báo trung vị (như bảng của eval: trung bình bị vài box cực xấu kéo lên hàng trăm) kèm trung bình. Không model, không GPU, ~1–2 phút.

  cd /mnt/disk1/aiotlab/haitn/object-detection/ce_localization
  python tools/gt_cnll.py --turn-index ../data/turn_index.json --cocount-root ../data/cocount \\
      --out /mnt/disk1/aiotlab/haitn/output/add/cocount/raw/gt_cnll.json
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ce_localization.data.cocount import read_cocount  # noqa: E402
from ce_localization.data.turns import TurnIndex  # noqa: E402
from ce_localization.engine.add_eval import CNLL_MIN_OBJ, cnll  # noqa: E402

__all__ = ["ce130_gt", "cocount_gt", "summarize"]


def ce130_gt(index, split="test", source="samples"):
    """-> list (gt [t,4] theo lượt, vật còn ở lượt t [M,4], (W, H)) cho mọi mẫu (nhánh, lượt) của split, pixel ảnh gốc."""
    out = []
    for k in index.keys(split, source):
        e = index.turns[k]
        b = index.branches[e["branch"]]
        t = e["t"]
        rm = np.asarray(b["removed"], dtype=int)
        objs = np.asarray(b["objects"], dtype=np.float64).reshape(-1, 4)[(rm == 0) | (rm > t)]
        out.append((np.asarray(b["holes"][:t], dtype=np.float64).reshape(-1, 4), objs, tuple(b["wh"])))
    return out


def cocount_gt(root, max_obj_ratio=None):
    """-> list (10 loc_bbox, vật cùng lớp, (W, H)) cho mọi mẫu CE-CoCount (`max_obj_ratio`: lọc box vật như eval.py)."""
    out = []
    for f in sorted(os.listdir(os.path.join(root, "Anno"))):
        if not f.endswith(".json"):
            continue
        name = f[:-5]
        r = read_cocount(root, name, max_obj_ratio)
        with Image.open(os.path.join(root, "Image", name + ".jpg")) as im:
            wh = im.size
        out.append((r["loc"], r["objects"], wh))
    return out


def summarize(samples, latest=False):
    """samples: list (gt, objects, wh) -> dict: với F1 / F2, `any_n1` (mọi box GT), `sel` (box GT C-NLL nhỏ nhất / mẫu), `latest_n1`
    (box GT cuối = lỗ mới nhất, chỉ khi latest=True): median / mean; n mẫu đủ vật."""
    acc = {f"{f}_{m}": [] for f in ("F1", "F2") for m in (("any_n1", "sel", "latest_n1") if latest else ("any_n1", "sel"))}
    n = 0
    for gt, objs, wh in samples:
        if len(gt) == 0 or cnll(gt, objs, wh, "F1") is None:
            continue
        n += 1
        for f in ("F1", "F2"):
            c = cnll(gt, objs, wh, f)
            acc[f"{f}_any_n1"] += c.tolist()
            acc[f"{f}_sel"].append(float(c.min()))
            if latest:
                acc[f"{f}_latest_n1"].append(float(c[-1]))
    res = {"n_samples": len(samples), "n_cnll": n, "min_obj": CNLL_MIN_OBJ}
    for k, v in acc.items():
        res[f"cnll_{k}_median"] = float(np.median(v)) if v else float("nan")
        res[f"cnll_{k}_mean"] = float(np.mean(v)) if v else float("nan")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-index", default="../data/turn_index.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--split-source", default="samples", help="như data.split_source của GAMMA2+ (samples: test 4.948 mẫu)")
    ap.add_argument("--cocount-root", default="../data/cocount", help="rỗng = bỏ CE-CoCount")
    ap.add_argument("--cocount-obj-filter", type=float, default=None, help="như eval.py: bỏ box vật > K × cạnh TB exemplar")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {"ce130": summarize(ce130_gt(TurnIndex(a.turn_index), a.split, a.split_source), latest=True)}
    res["ce130"].update(split=a.split, split_source=a.split_source)
    if a.cocount_root:
        res["cocount"] = summarize(cocount_gt(a.cocount_root, a.cocount_obj_filter))
        res["cocount"]["obj_filter"] = a.cocount_obj_filter
    for name, r in res.items():
        print(f"== {name}: {r['n_cnll']} / {r['n_samples']} mẫu có >= {r['min_obj']} vật")
        for k in sorted(k for k in r if k.startswith("cnll_") and k.endswith("_median")):
            print(f"   {k[5:-7]:14s} median {r[k]:8.3f} | mean {r[k[:-7] + '_mean']:9.3f}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
