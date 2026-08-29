import datetime
import os
import time
import yaml
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from models.diffusion_module import ObjectPlacementPolicy
from data.dataset import ObjectPlacementDataset
from utils.visualization import visualize_result
from utils.cnll import load_branch_lookup, find_all_bboxes, prepare_cnll, compute_cnll_prepared
from utils.matcher import _cxcywh_to_xyxy, box_iou_normalized
from train import load_config

N_SAMPLES = 30
TARGET_SIZE = 512  # ObjectPlacementDataset.target_size default


def _format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def save_pred_box(pred_boxes, scale, save_path, original_size=(512, 512)):
    # This function is now replaced by visualize_result which also saves the image with the box drawn.
    # If you want to save just the box coordinates, you can implement that here.
    # 2. Denormalize Coordinates
    # unpack normalized box (Range: -1 to 1)
    processed_boxes = []
    for pred_box in pred_boxes:
        x_norm, y_norm, w_norm, h_norm = pred_box
        target_w, target_h = original_size

        # Shift to [0, 1] then scale to the Model Input Dimension
        # (We use target_w for x/w and target_h for y/h)
        x_in_model = ((x_norm + 1) / 2) * target_w
        y_in_model = ((y_norm + 1) / 2) * target_h
        w_in_model = ((w_norm + 1) / 2) * target_w
        h_in_model = ((h_norm + 1) / 2) * target_h

        # 3. Scale back to Original Image Space
        # We divide by 'scale' to undo the resizing.
        # Note: This assumes Top-Left padding (0,0).
        # If you used Centered padding, you would subtract the padding offset here first.
        x = x_in_model / scale
        y = y_in_model / scale
        w = w_in_model / scale
        h = h_in_model / scale

        # 4. Convert Center-Format (x,y) to Top-Left-Format (x1,y1,x2,y2)
        # The model predicts the center of the box. PIL needs corners.
        x1 = x - (w / 2)
        y1 = y - (h / 2)
        x2 = x + (w / 2)
        y2 = y + (h / 2)
        box_coords = [float(x1), float(y1), float(x2), float(y2)]
        processed_boxes.append(box_coords)
    # Prepare the dictionary structure
    data = {
        'pred_box': processed_boxes
    }
    # Save to JSON file
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Successfully saved box to {save_path}")

def calculate_iou(box1, box2):
    # (Same IoU function as before)
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2

    xi1, yi1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    xi2, yi2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - inter_area
    return inter_area / (union_area + 1e-6)


def denormalize_cxcywh(norm_box, scale, target_size=TARGET_SIZE):
    """Undo ObjectPlacementDataset's [-1,1] normalization to get pixel-space
    (cx, cy, w, h) in the ORIGINAL image (before resize+pad), so it matches
    all_phase2_V2's fixed_annotation.json coordinates."""
    x_norm, y_norm, w_norm, h_norm = norm_box
    x_in_model = ((x_norm + 1) / 2) * target_size
    y_in_model = ((y_norm + 1) / 2) * target_size
    w_in_model = ((w_norm + 1) / 2) * target_size
    h_in_model = ((h_norm + 1) / 2) * target_size
    return (x_in_model / scale, y_in_model / scale, w_in_model / scale, h_in_model / scale)


def get_args():
    parser = argparse.ArgumentParser()
    # Path to the specific checkpoint you want to test
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--variant", type=str, required=True, help="variant name, e.g. resnet34_transformer")
    # Optional override for data path
    parser.add_argument("--test_data", type=str, default=None, help="Override path to test data")
    parser.add_argument(
        "--all_phase2_dir", type=str, default="../../data/all_phase2_V2",
        help="path to the original CE-130 dump (all_bboxes/inpainted_bboxes) used for C-NLL",
    )
    parser.add_argument(
        "--use_cache", action="store_true",
        help="read pre-resized samples from cache_512.u8 (build with tools/build_cache.py)",
    )
    parser.add_argument(
        "--inference_steps", type=int, default=None,
        help="number of reverse steps; fewer steps stride across the schedule, DDIM-style. "
             "Default for single-box variants = the model's own num_timesteps; for the "
             "multi-box variant = diffusion.sampling_steps from its config (4).",
    )
    parser.add_argument(
        "--pr_iou_threshold", type=float, default=0.5,
        help="IoU threshold for the multi-box (variant c) precision/recall metric",
    )
    return parser.parse_args()


def match_boxes_for_pr(pred_boxes, gt_boxes, iou_threshold):
    """Greedy one-to-one matching by IoU, for the variant (c) precision/recall
    metric only (not used by C-NLL/IoU@K, which are unchanged for a/b/c).
    pred_boxes/gt_boxes: iterables of normalized cxcywh. Returns (n_matched,).

    Uses utils.matcher's IoU, NOT this file's `calculate_iou`. CE-Loc normalizes
    w/h as (2 * fraction_of_image - 1), so a box smaller than half the image has
    a negative norm_w and `calculate_iou` — which treats norm_w as a raw width —
    builds an inverted box from it (see utils/matcher.py's COORDINATE SPACE note).
    `calculate_iou` is left as-is for IoU@10/IoU@30 so those stay comparable with
    the paper's published numbers, but this precision/recall metric is new here
    and has no such back-compatibility constraint, so it uses the correct geometry.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0
    pred_t = torch.as_tensor(np.asarray(pred_boxes), dtype=torch.float32)
    gt_t = torch.as_tensor(np.asarray(gt_boxes), dtype=torch.float32)
    iou_mat = box_iou_normalized(_cxcywh_to_xyxy(pred_t), _cxcywh_to_xyxy(gt_t))
    pairs = []
    for pi in range(iou_mat.shape[0]):
        for gi in range(iou_mat.shape[1]):
            iou = float(iou_mat[pi, gi])
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    pairs.sort(key=lambda x: -x[0])
    used_pred, used_gt = set(), set()
    n_matched = 0
    for iou, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        n_matched += 1
    return n_matched


@torch.no_grad()
def sample_boxes(model, global_cond, device, inference_steps, n_samples=N_SAMPLES):
    """DDPM/DDIM reverse process (Ho et al. 2020 Eq. 11; Song et al. 2021 Eq. 12)
    — replaces the previous 'mock update' (box -= noise/inference_steps) that
    ignored alphas/betas entirely.

    With inference_steps == num_timesteps this is exactly ancestral DDPM. With
    fewer steps it strides across the FULL schedule rather than walking only the
    low-noise tail: reversed(range(inference_steps)) would visit t=99..0 of a
    1000-step schedule, i.e. start the chain from a near-clean latent while the
    initial box is in fact pure N(0, I).
    """
    T = model.num_timesteps
    if inference_steps is None or inference_steps >= T:
        timesteps = list(reversed(range(T)))
    else:
        timesteps = list(reversed(
            torch.linspace(0, T - 1, inference_steps).round().long().tolist()
        ))

    alphas_cumprod = model.alphas_cumprod
    box = torch.randn((n_samples, 4), device=device)

    for i, t in enumerate(timesteps):
        t_batch = torch.full((n_samples,), t, device=device, dtype=torch.long)
        noise_pred = model.predict_noise(box, t_batch, global_cond)

        alpha_bar_t = alphas_cumprod[t]
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
        alpha_bar_prev = alphas_cumprod[t_prev] if t_prev >= 0 else torch.ones_like(alpha_bar_t)

        # Effective alpha/beta for the (possibly strided) step t -> t_prev.
        alpha_t = alpha_bar_t / alpha_bar_prev
        beta_t = 1.0 - alpha_t

        mean = (1.0 / torch.sqrt(alpha_t)) * (
            box - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
        )
        if t_prev >= 0:
            # posterior variance beta_tilde, exact for the full-schedule case
            sigma_t = torch.sqrt(beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            box = mean + sigma_t * torch.randn_like(box)
        else:
            box = mean
    return box


@torch.no_grad()
def sample_boxes_multibox(model, global_cond, device, sampling_steps):
    """DDIM reverse process for variant (c) — N boxes denoised JOINTLY in one
    chain (one [1, N, 4] latent), cross-checked against DiffusionDet's
    ddim_sample (object-detection/diffusiondet/diffusiondet/detector.py:185).

    Deliberately NOT ported from ddim_sample: `box_renewal` (drops/replenishes
    proposals by score > 0.5) and `use_ensemble` (accumulates+NMS across
    steps) — both are gated on a classification score CE-Loc has no head for.
    Skipping them means every one of the N output boxes always comes from the
    single final step, unfiltered; see README's "Hạn chế đã biết" for what
    this costs.

    Returns: [N, 4] normalized [-1,1] cxcywh (SNR-descaled already).
    """
    T = model.num_timesteps
    N = model.num_proposals
    # DiffusionDet: torch.linspace(-1, T-1, steps+1) -> pairs (T-1,T-2), ..., (0,-1)
    times = torch.linspace(-1, T - 1, steps=sampling_steps + 1)
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))

    alphas_cumprod = model.alphas_cumprod
    # x_T ~ N(0, I) — NOT scaled by snr_scale. The forward process is
    # x_t = sqrt(a_bar)*x_start + sqrt(1-a_bar)*eps with eps ~ N(0,I); at t=T-1
    # a_bar is 4e-05, so x_T has unit variance regardless of how x_start was
    # scaled. DiffusionDet likewise inits with a plain randn (detector.py:197).
    img = torch.randn((1, N, 4), device=device)

    for time, time_next in time_pairs:
        # Clamp the latent before it reaches the network, as DiffusionDet does
        # at detector.py:171 (`x_boxes = torch.clamp(x, -scale, scale)`).
        img = torch.clamp(img, min=-model.snr_scale, max=model.snr_scale)

        t_batch = torch.full((1,), time, device=device, dtype=torch.long)
        noise_pred = model.predict_noise(img, t_batch, global_cond)

        alpha_bar_t = alphas_cumprod[time]
        pred_x_start = (img - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        pred_x_start = torch.clamp(pred_x_start, min=-model.snr_scale, max=model.snr_scale)
        # Re-derive the noise FROM the clamped x_start, exactly as DiffusionDet
        # does (detector.py:182 `pred_noise = predict_noise_from_start(x, t, x_start)`).
        # Using the raw noise_pred here would let the DDIM update walk along a
        # direction inconsistent with the x_start it is interpolating toward,
        # silently undoing most of the clamp above.
        noise_pred = (img - torch.sqrt(alpha_bar_t) * pred_x_start) / torch.sqrt(1 - alpha_bar_t)

        if time_next < 0:
            img = pred_x_start
            break

        alpha_next = alphas_cumprod[time_next]
        # eta=0 (deterministic DDIM, DiffusionDet's own default ddim_sampling_eta),
        # so DiffusionDet's c = sqrt(1 - alpha_next - sigma^2) reduces to sqrt(1 - alpha_next).
        c = torch.sqrt(1 - alpha_next)
        img = pred_x_start * torch.sqrt(alpha_next) + c * noise_pred

    return (img.squeeze(0) / model.snr_scale).clamp(-1.0, 1.0)  # [N, 4]


@torch.no_grad()
def test():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Output Setup
    output_dir = f"./samples_cocount/processed_dataset/output_multiple_sampling_density1class_inf100_{args.variant}"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Configs — read from the checkpoint's own result dir (written by
    # train_w_args.py), not a hard-coded "best_ckpt" path.
    ckpt_dir = os.path.dirname(args.checkpoint)
    model_cfg = yaml.safe_load(open(os.path.join(ckpt_dir, "model_config_final.yaml"), "r"))
    train_cfg = load_config(os.path.join(ckpt_dir, "train_config_final.yaml"))

    # 3. SMART CHECKPOINT LOADING
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return

    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    if "args" in checkpoint and "num_steps" in checkpoint["args"]:
        trained_steps = checkpoint["args"]["num_steps"]
        if trained_steps is not None:
            print(f"Detected training steps from checkpoint: {trained_steps}")
            if "diffusion" not in model_cfg:
                model_cfg["diffusion"] = {}
            model_cfg["diffusion"]["num_timesteps"] = trained_steps
    else:
        print("Warning: Could not detect num_steps in checkpoint. Assuming default (100).")

    # 4. Initialize Model
    model = ObjectPlacementPolicy(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    multi_box = model.num_proposals > 1
    # --inference_steps overrides the config for BOTH paths; without this it
    # would silently do nothing on a multi-box checkpoint.
    sampling_steps = model_cfg.get("diffusion", {}).get("sampling_steps", 4)
    if multi_box and args.inference_steps is not None:
        sampling_steps = args.inference_steps
    if multi_box:
        print(f"multi_box eval: num_proposals={model.num_proposals}, "
              f"sampling_steps={sampling_steps}, snr_scale={model.snr_scale}")

    # 5. Dataset — variant (c) needs the same all_bboxes target the precision/
    # recall metric below is scored against (dataset space, not diffusion
    # space — see data/dataset.py's _build_multi_box_target docstring).
    test_path = args.test_data if args.test_data else train_cfg["training"]["data"].get("test_path", "data/test")
    print(f"Testing data from: {test_path}")
    dataset_kwargs = {"use_cache": args.use_cache}
    if multi_box:
        dataset_kwargs.update(multi_box=True, num_proposals=model.num_proposals,
                              all_phase2_dir=args.all_phase2_dir)
    dataset = ObjectPlacementDataset(test_path, **dataset_kwargs)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 6. C-NLL data — build the {id -> [(branch, turn, box, all_bboxes)]} lookup once.
    # Scan ALL splits under all_phase2_dir (train/test/val): samples/{train,test}
    # does not line up with all_phase2_V2's own 3-way split, so restricting to
    # a single matching split silently drops ~18% of images (measured).
    print(f"Loading all_phase2_V2 lookup (all splits) from {args.all_phase2_dir}...")
    branch_lookup = load_branch_lookup(args.all_phase2_dir)

    ious_k10, ious_k30, cnlls = [], [], []
    n_ambiguous_matches, n_unmatched, n_no_bj = 0, 0, 0
    pr_matched, pr_n_pred, pr_n_gt = 0, 0, 0  # variant (c) only
    eval_start = time.monotonic()

    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"eval[{args.variant}]"):
        rgb = batch["pixel_values"].to(device)
        density = batch["density_map"].to(device)
        gt_box = batch["bbox"].numpy()[0]
        text = batch["text"]
        scale = batch["scale"].item()

        if multi_box:
            # One DDIM chain samples all N boxes jointly (they see each other
            # via the decoder's self-attention — see sample_boxes_multibox).
            vis_emb = model.vision_encoder(rgb, density)
            text_emb = model.text_encoder(text)
            global_cond = torch.cat([vis_emb, text_emb], dim=-1)
            box = sample_boxes_multibox(model, global_cond, device, sampling_steps)
            pred_boxes = box.cpu().numpy()  # [N, 4] normalized [-1,1] cxcywh

            # Precision/recall vs. the true same-category box set — only
            # meaningful for (c), since (a)/(b) sample 30 candidates for the
            # SAME single position, not a set covering the whole image.
            gt_set = batch["boxes"][0][batch["box_mask"][0]].numpy()  # real boxes only, no padding
            n_matched_pr = match_boxes_for_pr(pred_boxes, gt_set, args.pr_iou_threshold)
            pr_matched += n_matched_pr
            pr_n_pred += len(pred_boxes)
            pr_n_gt += len(gt_set)
        else:
            vis_emb = model.vision_encoder(rgb, density)
            text_emb = model.text_encoder(text)
            global_cond = torch.cat([vis_emb, text_emb], dim=-1).expand(N_SAMPLES, -1)
            box = sample_boxes(model, global_cond, device, args.inference_steps, N_SAMPLES)
            pred_boxes = box.cpu().numpy()  # normalized [-1,1] cxcywh

        # Mean IoU @K — best-of-K against the removed ground-truth box. Kept
        # identical across (a)/(b)/(c) so the 3 variants stay comparable on
        # the same scale — for (c), K=10/30 index into the first 10/30 of the
        # N sampled boxes (K <= N required; num_proposals=100 satisfies this).
        ious_k10.append(max(calculate_iou(b, gt_box) for b in pred_boxes[:10]))
        ious_k30.append(max(calculate_iou(b, gt_box) for b in pred_boxes[:30]))

        # C-NLL — no GT involved: fit {B_j} from the source image, pick the
        # candidate with highest q(a_i) (paper §3.2's own selection rule),
        # report its C-NLL.
        filename = dataset.files[i]
        img_id = filename.split(".")[0].rsplit("_", 1)[0]
        target_bbox_px = denormalize_cxcywh(gt_box, scale)
        b_j, n_matches = find_all_bboxes(img_id, target_bbox_px, branch_lookup)
        if n_matches == 0:
            n_unmatched += 1
        elif n_matches > 1:
            n_ambiguous_matches += 1

        if b_j is not None and len(b_j) >= 2:
            pred_boxes_px = [denormalize_cxcywh(b, scale) for b in pred_boxes]
            # The Gaussian fit and the calibration term depend only on {B_j},
            # so they are computed once per image rather than once per
            # candidate (they dominated eval time at num_proposals=100).
            prepared = prepare_cnll(b_j)
            if prepared is None:
                n_no_bj += 1
            else:
                per_box_cnll = [compute_cnll_prepared(b, b_j, prepared) for b in pred_boxes_px]
                # highest q(a_i) == lowest -log q(a_i); the calibration term is
                # a constant shared by every candidate, so comparing per_box_cnll
                # directly still selects the same argmax-q candidate.
                cnlls.append(min(per_box_cnll))
        else:
            n_no_bj += 1

        save_path = os.path.join(output_dir, f"{filename.split('.')[0]}.json")
        save_pred_box(pred_boxes, scale, save_path)

    total_time = time.monotonic() - eval_start
    results = {
        "variant": args.variant,
        "checkpoint": args.checkpoint,
        "n_images": len(dataset),
        "mean_iou_k10": float(np.mean(ious_k10)),
        "mean_iou_k30": float(np.mean(ious_k30)),
        "cnll": float(np.mean(cnlls)) if cnlls else None,
        "n_cnll_images": len(cnlls),
        "n_ambiguous_branch_matches": n_ambiguous_matches,
        "n_unmatched_branch": n_unmatched,
        "n_no_bj": n_no_bj,
        "eval_time_seconds": total_time,
    }
    if multi_box:
        # Reported separately from the 3 metrics above — meaningful only for
        # variant (c) (a set-vs-set comparison), not a fair (a)/(b) baseline.
        results["multibox_precision"] = pr_matched / pr_n_pred if pr_n_pred else None
        results["multibox_recall"] = pr_matched / pr_n_gt if pr_n_gt else None
        results["multibox_pr_iou_threshold"] = args.pr_iou_threshold
        results["multibox_num_proposals"] = model.num_proposals
        results["multibox_sampling_steps"] = sampling_steps
    print(json.dumps(results, indent=2))
    print(f"Eval Complete for variant={args.variant}. Total time: {_format_duration(total_time)}")

    results_path = os.path.join(ckpt_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved eval results to {results_path}")


if __name__ == "__main__":
    test()
