"""Train and eval loops.

Two things here are specific to this project.

**Timestep-weighted loss.** The model returns ``diffusion_loss_weight`` alongside
its predictions; the training loop passes it into the criterion. On the baseline
path the key is absent and the criterion is called exactly as before.

**Warm-up freeze driven by the global step.** ``apply_diffusion_warmup`` is called
every iteration with ``epoch * len(loader) + it``, so freezing/unfreezing is a pure
function of training progress and survives a resume with no persisted flag.
"""

import math
import os
import sys
from typing import Iterable, Optional

import torch

import util.misc as utils
from gdino_datasets.coco_eval import CocoEvaluator
from util.param_dicts import apply_diffusion_warmup
from util.vl_utils import build_caption


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    wo_class_error: bool = False,
    lr_scheduler=None,
    args=None,
    logger=None,
):
    scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(args, "amp", False)))

    model.train()
    criterion.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    if getattr(args, "use_diffusion", False):
        metric_logger.add_meter("diff_t", utils.SmoothedValue(window_size=20, fmt="{avg:.0f}"))
    header = f"Epoch: [{epoch}]"

    steps_per_epoch = len(data_loader)
    warmup_active_last = None
    seen = 0

    for it, (samples, targets) in enumerate(metric_logger.log_every(data_loader, 10, header, logger=logger)):
        global_step = epoch * steps_per_epoch + it

        warmup_active = apply_diffusion_warmup(args, model, global_step)
        if warmup_active != warmup_active_last:
            message = (
                f"[step {global_step}] diffusion warm-up: freezing {args.diff_warmup_freeze_keywords}"
                if warmup_active
                else f"[step {global_step}] diffusion warm-up over, all towers trainable"
            )
            (logger.info if logger else print)(message)
            warmup_active_last = warmup_active

        samples = samples.to(device)
        captions = [t["caption"] for t in targets]
        cap_list = [t["cap_list"] for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if torch.is_tensor(v)} for t in targets]

        with torch.cuda.amp.autocast(enabled=bool(getattr(args, "amp", False))):
            outputs = model(samples, targets=targets, captions=captions)
            loss_dict = criterion(
                outputs,
                targets,
                cap_list,
                captions,
                t_weight=outputs.get("diffusion_loss_weight"),
            )
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict if k in weight_dict)

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f"{k}_unscaled": v for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_value = sum(loss_dict_reduced_scaled.values()).item()

        if not math.isfinite(loss_value):
            (logger.error if logger else print)(f"loss is {loss_value}, stopping training\n{loss_dict_reduced}")
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)
        if getattr(args, "amp", False):
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if getattr(args, "onecyclelr", False) and lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if "diffusion_t" in outputs:
            metric_logger.update(diff_t=outputs["diffusion_t"].float().mean().item())

        seen += 1
        if getattr(args, "debug", False) and seen % 15 == 0:
            (logger.info if logger else print)("debug mode: breaking out of the epoch early")
            break

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}


@torch.no_grad()
def evaluate(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error: bool = False,
    args=None,
    logger=None,
):
    """Score the val set with COCO AP.

    Every image gets the *same* prompt -- the full category list of the dataset --
    which is the OD-style open-vocabulary protocol: ask for all categories, expect
    all instances back.
    """
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    iou_types = tuple(k for k in ("bbox",) if k in postprocessors)
    use_cats = getattr(args, "useCats", True)
    if not use_cats:
        (logger.info if logger else print)("class-agnostic evaluation (useCats=False)")
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=use_cats)

    cat_list = postprocessors["bbox"].cat_list
    caption = build_caption(cat_list)
    (logger.info if logger else print)(f"eval prompt ({len(cat_list)} categories): {caption[:200]}")

    if getattr(args, "use_diffusion", False):
        steps = model.module.diffusion.sampling_timesteps if hasattr(model, "module") else model.diffusion.sampling_timesteps
        (logger.info if logger else print)(f"diffusion sampling: {steps} decoder evaluations per image")

    seen = 0
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: utils.to_device(v, device) for k, v in t.items()} for t in targets]

        batch_size = samples.tensors.shape[0]
        with torch.cuda.amp.autocast(enabled=bool(getattr(args, "amp", False))):
            outputs = model(samples, captions=[caption] * batch_size)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        coco_evaluator.update({t["image_id"].item(): r for t, r in zip(targets, results)})

        seen += 1
        if getattr(args, "debug", False) and seen % 15 == 0:
            (logger.info if logger else print)("debug mode: breaking out of eval early")
            break

    metric_logger.synchronize_between_processes()
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    return stats, coco_evaluator


__all__ = ["evaluate", "train_one_epoch"]
