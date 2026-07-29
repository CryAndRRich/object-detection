#!/usr/bin/env python3
# Dựa trên train_net.py của DiffusionDet (Shoufa Chen) và Sparse R-CNN (Peize Sun),
# bản thân chúng dựa trên tools/train_net.py của detectron2.
# Copyright (c) Facebook, Inc. and its affiliates.
"""Script train/eval DiffusionDet cho 3 dataset của repo này.

Khác bản gốc của DiffusionDet ở mấy điểm, đều để chạy được trên Kaggle và trên 3 dataset
đã chọn:

1. Tự đăng ký dataset (COCO-minitrain / VOC 07+12 / CrowdHuman) qua ``objdet.register_all``.
2. Chọn evaluator theo ``evaluator_type``: COCO -> COCOEvaluator, VOC ->
   PascalVOCDetectionEvaluator (VOC07 11-point, đúng giao thức baseline), CrowdHuman ->
   COCOEvaluator + CrowdHumanEvaluator (mMR/Recall).
3. Kiểm tra ``MODEL.DiffusionDet.NUM_CLASSES`` khớp số class thật của dataset — sai chỗ này
   thì train vẫn chạy nhưng kết quả vô nghĩa.
4. Giới hạn số checkpoint giữ lại (``SOLVER.CHECKPOINT_MAX_TO_KEEP``) vì /kaggle/working
   chỉ có 20GB mà mỗi checkpoint DiffusionDet ~0,5GB.
5. Bỏ nhánh LVIS (repo này không dùng LVIS).

Dùng:

    # train (2 GPU T4 trên Kaggle)
    python tools/train_net.py --num-gpus 2 --config-file configs/diffdet.minitrain.res50.yaml

    # train tiếp từ session trước
    python tools/train_net.py --num-gpus 2 --config-file ... --resume

    # eval, đổi số box / số bước sampling mà không cần train lại
    python tools/train_net.py --num-gpus 2 --config-file ... --eval-only \\
        MODEL.WEIGHTS output/.../model_final.pth \\
        MODEL.DiffusionDet.NUM_PROPOSALS 1000 MODEL.DiffusionDet.SAMPLE_STEP 4
"""

import itertools
import logging
import os
import sys
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import (
    AMPTrainer,
    DefaultTrainer,
    SimpleTrainer,
    create_ddp_model,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
    PascalVOCDetectionEvaluator,
    verify_results,
)
from detectron2.modeling import build_model
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusiondet import DiffusionDetDatasetMapper, DiffusionDetWithTTA, add_diffusiondet_config
from diffusiondet.util.model_ema import (
    EMADetectionCheckpointer,
    EMAHook,
    add_model_ema_configs,
    apply_model_ema_and_restore,
    may_build_model_ema,
    may_get_ema_checkpointer,
)
from objdet import dataset_num_classes, register_all
from objdet.crowdhuman_eval import CrowdHumanEvaluator


class Trainer(DefaultTrainer):
    """DefaultTrainer của detectron2, chỉnh cho DiffusionDet (giữ nguyên cách làm gốc)."""

    def __init__(self, cfg):
        # gọi __init__ của TrainerBase, bỏ qua của DefaultTrainer (giống bản gốc
        # DiffusionDet) vì cần tự dựng checkpointer có EMA
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        kwargs = {"trainer": weakref.proxy(self)}
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, **kwargs)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        logging.getLogger(__name__).info("Model:\n{}".format(model))
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Chọn evaluator theo loại dataset."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "pascal_voc":
            # VOC07 11-point metric — cùng giao thức với các baseline VOC đã công bố
            return PascalVOCDetectionEvaluator(dataset_name)

        if dataset_name.startswith("crowdhuman"):
            # AP/AP50 COCO-style + mMR/Recall theo giao thức CrowdHuman (Table 7 của paper)
            return DatasetEvaluators([
                COCOEvaluator(dataset_name, output_dir=output_folder),
                CrowdHumanEvaluator(dataset_name, output_dir=output_folder),
            ])

        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad or value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}]

        def maybe_add_full_model_gradient_clipping(optim):
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                return cls.test(cfg, model, evaluators=evaluators)
        return cls.test(cfg, model, evaluators=evaluators)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logging.getLogger("detectron2.trainer").info("Running inference with test-time augmentation ...")
        model = DiffusionDetWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA"))
            for name in cfg.DATASETS.TEST
        ]
        if cfg.MODEL_EMA.ENABLED:
            res = cls.ema_test(cfg, model, evaluators)
        else:
            res = cls.test(cfg, model, evaluators)
        return OrderedDict({k + "_TTA": v for k, v in res.items()})

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # tiết kiệm RAM/thời gian cho PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,
            hooks.LRScheduler(),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            # max_to_keep: /kaggle/working chỉ 20GB, mỗi checkpoint ~0,5GB nên phải giới hạn.
            # Vẫn giữ đủ để resume qua nhiều session.
            ret.append(hooks.PeriodicCheckpointer(
                self.checkpointer,
                cfg.SOLVER.CHECKPOINT_PERIOD,
                max_to_keep=cfg.SOLVER.CHECKPOINT_MAX_TO_KEEP,
            ))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


def add_kaggle_configs(cfg):
    """Config bổ sung của repo này (không có trong DiffusionDet gốc)."""
    # số checkpoint giữ lại; None/0 = giữ tất cả (dễ đầy đĩa trên Kaggle)
    cfg.SOLVER.CHECKPOINT_MAX_TO_KEEP = 3


def check_num_classes(cfg):
    """Bắt lỗi cấu hình số class sai — lỗi này không crash mà chỉ cho kết quả rác."""
    for split, names in (("TRAIN", cfg.DATASETS.TRAIN), ("TEST", cfg.DATASETS.TEST)):
        for name in names:
            expected = dataset_num_classes(name)
            if expected is None:
                continue
            got = cfg.MODEL.DiffusionDet.NUM_CLASSES
            if got != expected:
                raise ValueError(
                    f"DATASETS.{split} có '{name}' cần {expected} class nhưng "
                    f"MODEL.DiffusionDet.NUM_CLASSES = {got}. Sửa config trước khi train."
                )


def setup(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    add_kaggle_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    root = register_all()
    logging.getLogger("detectron2").info(f"OBJDET_DATA_ROOT = {os.path.abspath(root)}")
    check_num_classes(cfg)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        kwargs = may_get_ema_checkpointer(cfg, model)
        checkpointer = (EMADetectionCheckpointer if cfg.MODEL_EMA.ENABLED else DetectionCheckpointer)
        checkpointer(model, save_dir=cfg.OUTPUT_DIR, **kwargs).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.ema_test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
