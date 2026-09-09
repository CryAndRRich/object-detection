#!/usr/bin/env python3
"""Overfit ONE image — the gate before any long training run.

If the loss does not approach zero, the matcher or the loss is broken: STOP AND
FIX, do not keep training. Round 1 trained 5 times in a row on broken code purely
because this step was missing.

Use a small fixed t to isolate the question: at large t the input boxes are almost
pure noise, so overfitting is impossible, and that is NOT a bug.

  python3 tools/overfit_one.py --steps 300 --device cpu
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_dataset import normalize_for_clip  # noqa: E402
from data.factory import build_dataset  # noqa: E402
from models.detector import build_model  # noqa: E402
from models.criterion import SetCriterion, loss_from_output  # noqa: E402
from utils.box_ops import box_iou, cxcywh_to_xyxy  # noqa: E402
from utils.diffusion_math import prepare_diffusion_concat  # noqa: E402


def score_auc(boxes, logits, gt, iou_thr=0.5):
    """Can a reader tell the good boxes from the bad ones USING THE SCORE ALONE?

    0.5 = a coin flip, and that is not a figure of speech here: A, B and C1 all
    measured score_AUC ~0.497 on the real test set while covering 12-14 % of the
    GT, so AP threw most of their boxes away on random ranking. It is the single
    number EXPERIMENT E1 exists to move, and `loss`/`iou_matched` cannot see it --
    they are computed only over pairs the matcher already chose.

    Same definition as tools/measure_box_quality.py::roc_auc so the numbers are
    directly comparable: a prediction counts as positive if it clears iou_thr on
    ANY GT, ties get average ranks.
    """
    if gt.numel() == 0 or boxes.numel() == 0:
        return float("nan")
    m = box_iou(cxcywh_to_xyxy(boxes), cxcywh_to_xyxy(gt))[0]      # [P,G]
    y = (m.max(dim=1).values >= iou_thr).to(torch.float64).cpu().numpy()
    sc = logits.detach().to(torch.float64).cpu().numpy()
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")                                        # one class only
    order = np.argsort(sc, kind="mergesort")
    sv = sc[order]
    rank = np.empty(len(sv), dtype=np.float64)
    i = 0
    while i < len(sv):                                              # average ranks
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        rank[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    yo = y[order]
    n1 = yo.sum()
    n0 = len(yo) - n1
    return float((rank[yo == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment_a.yaml")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--t", type=int, default=50, help="fixed (small) timestep")
    ap.add_argument("--num-proposals", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(0)

    ds = build_dataset(cfg, "train")
    m = ds[a.index]
    px = torch.from_numpy(normalize_for_clip(m["image"])).unsqueeze(0).to(dev)
    gt = torch.from_numpy(m["boxes"]).float().to(dev)
    lab = torch.from_numpy(m["labels"]).long().to(dev)
    print(f"[image] {m['image_id']} '{m['text']}' | {len(gt)} GT | N={a.num_proposals} "
          f"| t={a.t}", flush=True)

    model = build_model(cfg, dropout=0.0).to(dev)
    model.train()
    crit = SetCriterion(cfg["matcher"]["method"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=a.lr)

    with torch.no_grad():                          # CLIP frozen -> encode once
        patch_raw = model.encoder.encode_image_raw(px)
        text_raw = model.encoder.encode_text_raw([m["text"]], dev)

    g = torch.Generator(device="cpu").manual_seed(0)
    x_t, _, _ = prepare_diffusion_concat(gt.cpu(), a.num_proposals, a.t,
                                         model.alphas_cumprod.cpu(),
                                         cfg["diffusion"]["snr_scale"],
                                         valid_h=m["valid_h"], generator=g)
    x_t = x_t.unsqueeze(0).to(dev)
    tt = torch.full((1,), a.t, dtype=torch.long, device=dev)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    first_loss, history = None, []
    for i in range(a.steps):
        out = model(x_t, tt, patch_raw=patch_raw, text_raw=text_raw)
        loss, st, _, lg = loss_from_output(crit, out, [gt], labels=[lab])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            bx = out[-1][0][0] if isinstance(out, list) else out[0][0]
            sc = lg[0] if lg.dim() == 2 else lg[0].max(-1).values
            auc = score_auc(bx, sc, gt)
        history.append((st["loss"], st["iou_matched"], auc))
        if first_loss is None:
            first_loss = st["loss"]
        if i % max(a.steps // 10, 1) == 0 or i == a.steps - 1:
            bn = (model.decoder.roi.branch_norm()
                  if model.decoder.roi is not None else None)
            print(f"  [{i:4d}] loss {st['loss']:7.4f} | l1 {st['loss_l1']:.4f} "
                  f"giou {st['loss_giou']:.4f} ce {st['loss_ce']:.4f} | "
                  f"IoU {st['iou_matched']:.4f} | score_AUC {auc:.4f}"
                  + (f" | roi_branch_norm {bn:.4f}" if bn is not None else ""),
                  flush=True)

    # Judge by the BEST IoU and the mean of the last 10 steps, NOT by the single
    # final step: with a fixed LR the final step is just a random sample of the
    # oscillation (seen in practice: IoU reached 0.69 at step 210 then swung back
    # to 0.47 at step 299).
    best_iou = max(x[1] for x in history)
    iou_last10 = float(np.mean([x[1] for x in history[-10:]]))
    loss_last10 = float(np.mean([x[0] for x in history[-10:]]))
    aucs = [x[2] for x in history if not np.isnan(x[2])]
    best_auc = max(aucs) if aucs else float("nan")
    auc_last10 = float(np.mean(aucs[-10:])) if aucs else float("nan")
    ratio = loss_last10 / max(first_loss, 1e-9)

    print(f"\nloss {first_loss:.4f} -> {loss_last10:.4f} "
          f"({100*ratio:.1f} % remaining, mean of last 10 steps)")
    print(f"IoU_matched: best {best_iou:.4f} | last 10 steps {iou_last10:.4f}")
    print(f"score_AUC:   best {best_auc:.4f} | last 10 steps {auc_last10:.4f}"
          f"   (0.5 = coin flip; A/B/C1 measured ~0.497 on the real test set)")

    roi = model.decoder.roi
    score_roi = getattr(model.decoder, "score_roi", False)
    if roi is not None:
        print(f"roi_branch_norm: {roi.branch_norm():.4f}"
              f"   ({'E1: score head reads it' if score_roi else 'B: added to decoder input'})")

    if best_iou > 0.6 and ratio < 0.35:
        print("[PASS] matcher and loss work correctly, safe to continue training")
        if iou_last10 < best_iou - 0.1:
            print("  (IoU oscillates at the end — normal when overfitting 1 image, "
                  "not a bug)")
    elif ratio < 0.5:
        print("[PARTIAL] loss clearly drops but IoU is not high. Check before long training:")
        print("  run tools/visualize_data.py to see whether GT boxes bound the objects.")
    else:
        print("[FAIL] the matcher or the loss is broken. STOP, do not train long.")

    # ---- EXPERIMENT E1 gate -------------------------------------------------
    # The checks above judge the BOX path, and they pass for A, B, C1 and E1
    # alike -- E1's boxes are bit-identical to A's by construction, so they say
    # nothing about the one thing E1 changes. Without the two checks below this
    # tool would print [PASS] on a completely dead RoI branch, which is exactly
    # the failure that the zero-init deadlock produced (score frozen at a
    # constant, loss still falling, no warning anywhere).
    if score_roi:
        print()
        bn = roi.branch_norm() if roi is not None else 0.0
        if bn <= 1e-6:
            print("[E1 FAIL] roi_branch_norm is still ~0: the RoI branch never "
                  "learned, so the score head is reading a dead input. STOP -- "
                  "training would reproduce EXPERIMENT A with a constant score.")
        elif np.isnan(best_auc):
            print("[E1 SKIP] score_AUC is undefined (no box cleared IoU 0.5 on "
                  "this image, or all did). Try --index on a different image.")
        elif best_auc > 0.90:
            print(f"[E1 PASS] score_AUC {best_auc:.4f} on one image: the score "
                  f"head can separate good boxes from bad using the image inside "
                  f"them. Train for real.")
        elif best_auc > 0.70:
            print(f"[E1 PARTIAL] score_AUC {best_auc:.4f}. Above chance, but on a "
                  f"SINGLE overfitted image it should be near 1.0. Worth training, "
                  f"but expect a modest gain.")
        else:
            print(f"[E1 FAIL] score_AUC {best_auc:.4f} — barely above the 0.5 coin "
                  f"flip even when overfitting ONE image, where the head has every "
                  f"advantage. The RoI signal is not reaching the score. STOP: a "
                  f"linear probe on these same frozen features reaches 0.888, so "
                  f"this is a wiring problem, not a capacity limit.")


if __name__ == "__main__":
    main()
