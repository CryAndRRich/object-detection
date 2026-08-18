"""Training / evaluation entry point.

    python main.py -c config/cfg_odvg_diffusion.py --datasets data/coco_minitrain.json \
        --output_dir output/diffu_run1 \
        --pretrain_model_path ../weights/diffu_grounding_dino/groundingdino_swint_ogc.pth \
        --finetune_ignore time_ diffusion

Note the ordering constraint marked below around ``get_param_dict``: the optimizer
must be built while every parameter is still visible, or the warm-up freeze becomes
permanent.
"""

import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils
from gdino_datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.lora import inject_lora
from util.config import Config, DictAction
from util.logger import setup_logger
from util.misc import BestMetricHolder, clean_state_dict
from util.param_dicts import apply_freeze_keywords, describe_trainable, get_param_dict


def get_args_parser():
    parser = argparse.ArgumentParser("DiffuGroundingDINO", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="override config fields as key=value"
    )
    parser.add_argument("--datasets", type=str, required=True, help="path to the datasets json")

    parser.add_argument("--output_dir", default="", help="where to write checkpoints and logs")
    parser.add_argument("--note", default="", help="free-form note recorded in the log")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from a checkpoint (optimizer included)")
    parser.add_argument("--pretrain_model_path", help="load weights only, e.g. groundingdino_swint_ogc.pth")
    parser.add_argument(
        "--finetune_ignore",
        type=str,
        nargs="+",
        help="checkpoint keys containing any of these substrings are not loaded "
        "(use 'time_ diffusion' when starting a diffusion run from a baseline checkpoint)",
    )
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument(
        "--max_epochs_this_run",
        type=int,
        default=None,
        help="exit cleanly after this many epochs in THIS process, regardless of --start_epoch "
        "-- for splitting a run across several time-limited sessions (e.g. a Kaggle Save & Run "
        "All that gets killed at its session cap). --epochs stays the real total for the LR "
        "schedule; resuming from the checkpoint this leaves behind continues it correctly. "
        "Counting from --start_epoch (rather than an absolute epoch number) means each "
        "session's notebook cell can just say 'run 1 more epoch' without knowing which epoch a "
        "resumed checkpoint left off at. Without this, a session that hits its time limit is "
        "killed mid-epoch: no checkpoint for that epoch, and no cell after the training cell "
        "(log read, final eval sweep) ever gets to run.",
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--fix_size", action="store_true")
    parser.add_argument("--debug", action="store_true", help="break out of loops after a few steps")
    parser.add_argument("--find_unused_params", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--local-rank", type=int, dest="local_rank")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="mixed precision. Off by default: DiffusionDet breaks under fp16 and this "
        "model family shares its multi-stage box refinement. The diffusion schedule is "
        "pinned to fp32 regardless.",
    )
    return parser


def load_config_into_args(args, logger=None):
    cfg = Config.fromfile(args.config_file)
    if args.options:
        cfg.merge_from_dict(args.options)

    for key, value in cfg.to_dict().items():
        if hasattr(args, key):
            raise ValueError(f"config key {key!r} collides with a command-line argument")
        setattr(args, key, value)
    return cfg


def build_optimizer_and_freeze(args, model_without_ddp, logger):
    """Build parameter groups, then apply the permanent freeze -- in that order.

    ``get_param_dict`` decides group membership from the parameters it is shown. If
    a parameter is frozen *before* this call it is dropped from the optimizer for
    the entire run, and any later unfreeze does nothing. So: collect first
    (``include_frozen=True``), freeze second.
    """
    param_dicts = get_param_dict(args, model_without_ddp, include_frozen=True)

    # Under LoRA, inject_lora() already froze the base towers (and only the base --
    # not the lora_A/B adapters sharing the same name prefix). Running the
    # keyword-based freeze on top would be redundant at best; skip it so the two
    # freezing mechanisms never have to be reconciled.
    if getattr(args, "freeze_keywords", None) and not getattr(args, "use_lora", False):
        apply_freeze_keywords(model_without_ddp, args.freeze_keywords)
        logger.info(f"permanently frozen by keywords {args.freeze_keywords}")

    logger.info("trainable parameters after freezing:\n" + describe_trainable(model_without_ddp))
    return torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)


def load_pretrained_weights(args, model_without_ddp, logger):
    """Load a weights-only checkpoint, skipping ``finetune_ignore`` keys."""
    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = clean_state_dict(state)

    ignore = args.finetune_ignore or []
    skipped = [k for k in state if any(kw in k for kw in ignore)]
    filtered = {k: v for k, v in state.items() if k not in skipped}

    result = model_without_ddp.load_state_dict(filtered, strict=False)
    logger.info(f"loaded {args.pretrain_model_path}")
    if ignore:
        logger.info(f"ignored {len(skipped)} keys matching {ignore}")

    # A freshly built diffusion model legitimately misses the timestep modules; any
    # other missing key means a module was renamed and is silently untrained.
    unexpected_missing = [k for k in result.missing_keys if not any(kw in k for kw in ignore + ["time_", "diffusion"])]
    logger.info(f"missing keys: {len(result.missing_keys)} (unexpected: {len(unexpected_missing)})")
    if unexpected_missing:
        logger.warning("UNEXPECTED missing keys -- these will stay randomly initialised:")
        for key in unexpected_missing[:40]:
            logger.warning(f"  {key}")
    if result.unexpected_keys:
        logger.warning(f"checkpoint had {len(result.unexpected_keys)} keys the model does not use:")
        for key in result.unexpected_keys[:20]:
            logger.warning(f"  {key}")
    return result


def main(args):
    utils.setup_distributed(args)
    cfg = load_config_into_args(args)

    if not args.output_dir:
        raise ValueError("--output_dir is required")
    os.makedirs(args.output_dir, exist_ok=True)
    output_dir = Path(args.output_dir)

    logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"), distributed_rank=args.rank, color=False
    )
    logger.info(f"git: {utils.get_sha()}")
    logger.info("command: " + " ".join(sys.argv))
    if args.note:
        logger.info(f"note: {args.note}")

    if args.rank == 0:
        cfg.dump(output_dir / "config_cfg.py")
        (output_dir / "config_args_all.json").write_text(
            json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2), encoding="utf-8"
        )

    with open(args.datasets, encoding="utf-8") as f:
        dataset_meta = json.load(f)
    if args.use_coco_eval:
        args.coco_val_path = dataset_meta["val"][0]["anno"]

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    logger.info(f"building model (use_diffusion={getattr(args, 'use_diffusion', False)})")
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    model_without_ddp = model

    # ---------------- resume/pretrain decision + lora injection ----------------
    # Decided here, before DDP wrap and before the optimizer exists, because both of
    # those need to see the model's FINAL structure (lora_A/B in the right param
    # groups, DDP tracking the right tensors). The actual checkpoint *tensor*
    # loading (and the optimizer/lr_scheduler restore for --resume) still happens
    # further down, once the optimizer exists to load state into -- this block only
    # settles resume-vs-pretrain early enough to get LoRA injection ordering right:
    #   - fresh pretrain start: the checkpoint has flat `...weight` keys, so load
    #     it into the plain model first, THEN inject LoRA around the now-pretrained
    #     values.
    #   - resume: a checkpoint saved by a previous LoRA session already has
    #     `...base.weight`/`...lora_A` keys, so LoRA must be injected first so the
    #     structure matches before that load happens below.
    auto_resume = output_dir / "checkpoint.pth"
    if not args.resume and auto_resume.exists():
        args.resume = str(auto_resume)
        logger.info(f"auto-resuming from {auto_resume}")

    use_lora = getattr(args, "use_lora", False)
    if args.resume:
        if use_lora:
            n_wrapped = inject_lora(
                model_without_ddp, args.lora_target_prefixes, args.lora_rank, args.lora_alpha, args.lora_dropout
            )
            logger.info(f"lora: injected into {n_wrapped} layers (before resume load)")
    elif args.pretrain_model_path:
        load_pretrained_weights(args, model_without_ddp, logger)
        if use_lora:
            n_wrapped = inject_lora(
                model_without_ddp, args.lora_target_prefixes, args.lora_rank, args.lora_alpha, args.lora_dropout
            )
            logger.info(f"lora: injected into {n_wrapped} layers (after pretrain load)")

    if args.distributed:
        # The diffusion warm-up (apply_diffusion_warmup, called every training step)
        # toggles requires_grad on backbone/bert mid-run: frozen for the first
        # diff_warmup_iters steps, trainable after. static_graph=True tells DDP the
        # set of parameters producing gradients never changes across iterations --
        # exactly what this schedule violates, and would either error out or (worse)
        # silently stop syncing backbone/bert gradients across ranks once they
        # unfreeze. So skip static_graph and force find_unused_parameters whenever
        # that schedule is active; a baseline (non-diffusion or warmup-less) run is
        # unaffected and keeps the original static-graph fast path. Under LoRA the
        # warmup schedule is skipped entirely (engine.py) and requires_grad on the
        # base towers was already fixed permanently by inject_lora() above, before
        # DDP ever sees the model -- no mid-run toggle, so the fast path is safe.
        warmup_schedule_active = (
            not use_lora
            and getattr(args, "use_diffusion", False)
            and int(getattr(args, "diff_warmup_iters", 0) or 0) > 0
        )
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=args.find_unused_params or warmup_schedule_active,
        )
        if not warmup_schedule_active:
            model._set_static_graph()
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of trainable params: {n_parameters:,}")

    optimizer = build_optimizer_and_freeze(args, model_without_ddp, logger)

    # ---------------- data ----------------
    dataset_val = build_dataset(image_set="val", args=args, datasetinfo=dataset_meta["val"][0])
    base_ds = get_coco_api_from_dataset(dataset_val)

    dataset_train = None
    if not args.eval:
        train_entries = dataset_meta["train"]
        datasets = [build_dataset("train", args=args, datasetinfo=info) for info in train_entries]
        if len(datasets) == 1:
            dataset_train = datasets[0]
        else:
            from torch.utils.data import ConcatDataset

            dataset_train = ConcatDataset(datasets)
        logger.info(f"train datasets: {len(datasets)}, total samples: {len(dataset_train)}")

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_train = DistributedSampler(dataset_train) if dataset_train is not None else None
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train) if dataset_train is not None else None

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=4,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_train = None
    if dataset_train is not None:
        data_loader_train = DataLoader(
            dataset_train,
            batch_sampler=torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True),
            collate_fn=utils.collate_fn,
            num_workers=args.num_workers,
        )

    # ---------------- schedule ----------------
    if args.onecyclelr:
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2
        )
    elif args.multi_step_lr:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    # best_map_holder must exist before the resume block below -- its state has to be
    # restored FROM the checkpoint here, not recreated fresh after. Recreating it fresh
    # (as this used to do, further down) meant every resumed session's first epoch was
    # treated as a new best (0.0 initial "best" beats any real mAP), silently overwriting
    # checkpoint_best_regular.pth with a possibly worse epoch than a prior session's true
    # best -- exactly the failure mode a multi-session Kaggle resume plan depends on not
    # having.
    best_map_holder = BestMetricHolder()

    # ---------------- resume: load tensors + optimizer/scheduler state ----------------
    # (resume-vs-pretrain was already decided above, before DDP wrap/lora injection;
    # this is just the tensor/optimizer-state load, which needs `optimizer` to exist.)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        if not args.eval and {"optimizer", "lr_scheduler", "epoch"} <= set(checkpoint):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            logger.info(f"resumed at epoch {args.start_epoch}")
        if {"best", "best_epoch"} <= set(checkpoint):
            best_map_holder.best = checkpoint["best"]
            best_map_holder.best_epoch = checkpoint["best_epoch"]
            logger.info(f"restored best-so-far: {best_map_holder}")

    # ---------------- eval only ----------------
    if args.eval:
        os.environ["EVAL_FLAG"] = "TRUE"
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args, logger=logger
        )
        if utils.is_main_process():
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
            with (output_dir / "log.txt").open("a", encoding="utf-8") as f:
                f.write(json.dumps({f"test_{k}": v for k, v in test_stats.items()}) + "\n")
        return

    # ---------------- train ----------------
    logger.info("start training")
    start_time = time.time()
    epochs_run_this_session = 0

    for epoch in range(args.start_epoch, args.epochs):
        epoch_start = time.time()
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
            lr_scheduler=lr_scheduler,
            args=args,
            logger=(logger if args.save_log else None),
        )

        if not args.onecyclelr:
            lr_scheduler.step()

        def snapshot():
            return {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "best": best_map_holder.best,
                "best_epoch": best_map_holder.best_epoch,
                "args": vars(args),
            }

        checkpoint_paths = [output_dir / "checkpoint.pth"]
        if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
        for path in checkpoint_paths:
            utils.save_on_master(snapshot(), path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args, logger=(logger if args.save_log else None)
        )
        map_regular = test_stats["coco_eval_bbox"][0]
        if best_map_holder.update(map_regular, epoch):
            utils.save_on_master(snapshot(), output_dir / "checkpoint_best_regular.pth")
        logger.info(f"epoch {epoch}: mAP {map_regular:.4f}, best so far {best_map_holder}")

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "epoch_time": str(datetime.timedelta(seconds=int(time.time() - epoch_start))),
            "now_time": str(datetime.datetime.now()),
        }
        if utils.is_main_process():
            with (output_dir / "log.txt").open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                torch.save(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval" / "latest.pth")

        epochs_run_this_session += 1
        if args.max_epochs_this_run is not None and epochs_run_this_session >= args.max_epochs_this_run:
            logger.info(
                f"stopping after epoch {epoch} ({epochs_run_this_session}/{args.max_epochs_this_run} "
                f"epochs this session); resume with --resume {output_dir / 'checkpoint.pth'} to "
                f"continue toward --epochs {args.epochs}"
            )
            break

    logger.info(f"training time {datetime.timedelta(seconds=int(time.time() - start_time))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DiffuGroundingDINO", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
