#!/usr/bin/env python3
# Dựa trên train_net.py của DiffusionDet (Shoufa Chen) và Sparse R-CNN (Peize Sun),
# bản thân chúng dựa trên tools/train_net.py của detectron2.
# Copyright (c) Facebook, Inc. and its affiliates.
"""Train / eval chung cho các baseline detectron2 — một script, một config mỗi baseline:

    BASELINE0  configs/baseline0_diffusiondet.yaml   META_ARCHITECTURE DiffusionDet
    BASELINE1  configs/baseline1_sparsercnn.yaml     META_ARCHITECTURE SparseRCNN
    BASELINE2  configs/baseline2_fasterrcnn.yaml     META_ARCHITECTURE GeneralizedRCNN (Faster R-CNN)
    (configs/benchmarks/: 3 benchmark DiffusionDet cũ — COCO-minitrain / VOC / CrowdHuman, lưu trữ)

So với train_net.py của DiffusionDet gốc:
1. Tự đăng ký dataset (`objdet.register_all`), evaluator theo loại dataset: CE-130 -> COCOEvaluator
   (đường cong) + `CE130BoxQualityEvaluator` (`oracle_recall` để chọn checkpoint); VOC -> VOC07
   11-point; CrowdHuman -> COCO + mMR/Recall.
2. `check_num_classes`: số lớp của config khớp dataset (sai thì vẫn train, kết quả vô nghĩa).
3. OUTPUT_DIR chỉ có `last.pth` (model + optimizer + scheduler, ghi đè mỗi CHECKPOINT_PERIOD, để
   `--resume`), `best.pth` (CHỈ model, theo `ce130/oracle_recall` val) và `history.json` (loss / lr mỗi 20
   iter + mọi lần eval + best) — ghi nguyên tử (file tạm rồi `os.replace`). Không `model_XXXXXXX.pth`,
   không `metrics.json` / tensorboard / `inference/` như mặc định detectron2.
4. Faster R-CNN dùng optimizer chuẩn của detectron2 (SGD, không weight decay cho norm);
   DiffusionDet / Sparse R-CNN giữ optimizer của repo gốc (AdamW, clip toàn mô hình).
5. `--max-hours`: dừng sạch (ghi checkpoint, `--resume` được) — cho phiên Kaggle ≤ 11 giờ.
6. Giới hạn số checkpoint giữ lại (`SOLVER.CHECKPOINT_MAX_TO_KEEP`).
AMP phải TẮT (`SOLVER.AMP.ENABLED False`): DiffusionDet vỡ cấu trúc với fp16.

Chạy (từ object-detection/baseline/, `export OBJDET_DATA_ROOT=../data`):

    python train_net.py --num-gpus 1 --config-file configs/baseline0_diffusiondet.yaml [--resume]
    python train_net.py --num-gpus 2 --config-file ... OUTPUT_DIR /kaggle/working/ckpt --max-hours 9.5
    # eval COCO + oracle_recall trên một dataset đã đăng ký (số báo cáo: predict.py)
    python train_net.py --config-file ... --eval-only MODEL.WEIGHTS ../weights/detection/baseline0/best.pth \\
        DATASETS.TEST '("ce130_agnostic_test",)' --dump-results /mnt/disk1/aiotlab/haitn/output/baselines/x.json
"""

import itertools
import json
import logging
import math
import os
import sys
import time
import weakref
from collections import OrderedDict
from typing import Any, Dict, List, Set

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode as CN
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
from detectron2.utils.events import CommonMetricPrinter
from detectron2.utils.logger import setup_logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # object-detection/

from baseline.diffusiondet import DiffusionDetDatasetMapper, DiffusionDetWithTTA, add_diffusiondet_config  # noqa: E402
from baseline.diffusiondet.util.model_ema import (  # noqa: E402
    EMADetectionCheckpointer,
    EMAHook,
    add_model_ema_configs,
    apply_model_ema_and_restore,
    may_build_model_ema,
    may_get_ema_checkpointer,
)
from baseline.objdet import dataset_num_classes, register_all  # noqa: E402
from baseline.objdet.ce130_eval import CE130BoxQualityEvaluator  # noqa: E402
from baseline.objdet.crowdhuman_eval import CrowdHumanEvaluator  # noqa: E402
from baseline.sparsercnn import add_sparsercnn_config  # noqa: E402  (import = đăng ký META_ARCH SparseRCNN)

# khoá số lớp theo kiến trúc
NUM_CLASSES_KEY = {"DiffusionDet": ("DiffusionDet", "NUM_CLASSES"),
                   "SparseRCNN": ("SparseRCNN", "NUM_CLASSES"),
                   "GeneralizedRCNN": ("ROI_HEADS", "NUM_CLASSES")}
SELECT_METRIC = "ce130/oracle_recall"


class StopTraining(Exception):
    """Hết ngân sách giờ (`--max-hours`): đã ghi checkpoint, chạy lại với --resume."""


def save_atomic(checkpointer, name, tag_last=False, **extra):
    """Ghi `<save_dir>/<name>.pth` như `Checkpointer.save` (model + checkpointables + extra) nhưng NGUYÊN TỬ
    (file tạm rồi os.replace: đứt giữa chừng không làm hỏng bản cũ) và chỉ trỏ `last_checkpoint` khi
    `tag_last` — `Checkpointer.save` luôn trỏ, nên lưu best qua nó làm `--resume` nạp nhầm best."""
    if not comm.is_main_process():
        return
    data = {"model": checkpointer.model.state_dict()}
    for key, obj in checkpointer.checkpointables.items():
        data[key] = obj.state_dict()
    data.update(extra)
    path = os.path.join(checkpointer.save_dir, f"{name}.pth")
    with open(path + ".tmp", "wb") as f:
        torch.save(data, f)
    os.replace(path + ".tmp", path)
    if tag_last:
        checkpointer.tag_last_checkpoint(f"{name}.pth")


def write_json_atomic(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(path + ".tmp", path)


class LastCheckpointer(hooks.HookBase):
    """`last.pth` (model + optimizer + scheduler + iteration) mỗi `period` iter và ở iter cuối; `--resume`
    nạp nó. Thay `PeriodicCheckpointer` (giữ nhiều `model_XXXXXXX.pth` + `model_final.pth`)."""

    def __init__(self, checkpointer, period):
        self._ck, self._period = checkpointer, period

    def after_step(self):
        it = self.trainer.iter
        if (it + 1) % self._period == 0 or it + 1 >= self.trainer.max_iter:
            save_atomic(self._ck, "last", tag_last=True, iteration=it)


class HistoryAndBest(hooks.HookBase):
    """`history.json` = {"train": loss / lr (trung vị 20 iter), "eval": mọi lần eval, "best"} và `best.pth`
    (CHỈ model) khi `metric` của lần eval mới cao hơn best. Đứng SAU EvalHook (đọc kết quả eval từ storage).
    `--resume`: nạp history cũ, bỏ bản ghi sau iter nối tiếp; best cũ giữ nguyên (best.pth vẫn đúng weight đó)."""

    EVAL_PREFIX = ("bbox/", "ce130/")

    def __init__(self, path, best_checkpointer, metric=SELECT_METRIC, period=20):
        self._path, self._best_ck, self._metric, self._period = path, best_checkpointer, metric, period

    def before_train(self):
        start = self.trainer.start_iter
        self._h = {"train": [], "eval": [], "best": None, "metric": self._metric}
        if start > 0 and os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                old = json.load(f)
            self._h["train"] = [r for r in old.get("train", []) if r["iter"] <= start]
            self._h["eval"] = [r for r in old.get("eval", []) if r["iter"] <= start]
            self._h["best"] = old.get("best")
        self._last_eval = -1

    def _check_eval(self):
        latest = self.trainer.storage.latest()
        ev = {k: v for k, v in latest.items() if k.startswith(self.EVAL_PREFIX)}
        if not ev:
            return False
        it = max(i for _, i in ev.values())
        if it <= self._last_eval:
            return False
        self._last_eval = it
        # số iter đã train lúc eval: eval định kỳ ghi ở storage.iter = it (0-based) -> it + 1; eval cuối
        # (EvalHook.after_train) ghi ở storage.iter = max_iter -> max_iter
        done = min(it + 1, self.trainer.max_iter)
        rec = {"iter": done, **{k: float(v) for k, (v, i) in ev.items() if i == it}}
        self._h["eval"].append(rec)
        val = rec.get(self._metric)
        best = self._h["best"]
        if val is not None and math.isfinite(val) and (best is None or val > best[self._metric]):
            save_atomic(self._best_ck, "best", iteration=done - 1, **{self._metric: val})
            self._h["best"] = {"iter": done, self._metric: val}
            logging.getLogger("detectron2").info(f"best.pth <- iter {done} ({self._metric} {val:.4f})")
        return True

    def after_step(self):
        it = self.trainer.iter
        changed = self._check_eval()
        if (it + 1) % self._period == 0 or it + 1 >= self.trainer.max_iter:
            sm = self.trainer.storage.latest_with_smoothing_hint(self._period)
            rec = {"iter": it + 1}
            rec.update({k: float(v) for k, (v, _) in sm.items()
                        if k in ("total_loss", "lr", "time") or k.startswith("loss")})
            self._h["train"].append(rec)
            changed = True
        if changed:
            write_json_atomic(self._path, self._h)

    def after_train(self):
        self._check_eval()                          # eval cuối của EvalHook.after_train
        write_json_atomic(self._path, self._h)


class TimeLimit(hooks.HookBase):
    """Quá `hours` thì ghi `last.pth` của iter hiện tại rồi dừng. Mọi rank cùng quyết định (all_gather
    mỗi 20 iter) — một rank dừng một mình thì DDP treo."""

    def __init__(self, hours, every=20):
        self._limit = hours * 3600 if hours else None
        self._every = every

    def before_train(self):
        self._t0 = time.time()

    def after_step(self):
        if self._limit is None or (self.trainer.iter + 1) % self._every:
            return
        over = time.time() - self._t0 > self._limit
        if comm.get_world_size() > 1:
            over = any(comm.all_gather(over))
        if over:
            it = self.trainer.iter
            save_atomic(self.trainer.checkpointer, "last", tag_last=True, iteration=it)
            comm.synchronize()
            raise StopTraining(f"hết {self._limit / 3600:.2f} giờ ở iter {it} — đã ghi last.pth")


class Trainer(DefaultTrainer):
    """DefaultTrainer của detectron2, chỉnh cho DiffusionDet / Sparse R-CNN (giữ cách làm gốc)."""

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
        self.best_checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR)      # best.pth: chỉ model
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

        if dataset_name.startswith("ce130"):
            # COCO chỉ để xem đường cong (không ghi inference/); oracle_recall để chọn best.pth
            return DatasetEvaluators([
                COCOEvaluator(dataset_name, output_dir=None),
                CE130BoxQualityEvaluator(dataset_name),
            ])

        return COCOEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        # CÙNG mapper (lật + đa tỉ lệ + crop như D.1) cho mọi kiến trúc: augmentation giống nhau
        mapper = DiffusionDetDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.MODEL.META_ARCHITECTURE == "GeneralizedRCNN":
            # Faster R-CNN: optimizer chuẩn của detectron2 (SGD momentum, WEIGHT_DECAY_NORM)
            return super().build_optimizer(cfg, model)
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
            ret.append(LastCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # PHẢI đứng sau EvalHook: đọc kết quả eval (vd "ce130/oracle_recall") EvalHook vừa ghi vào storage
            ret.append(HistoryAndBest(os.path.join(cfg.OUTPUT_DIR, "history.json"), self.best_checkpointer))
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def build_writers(self):
        """Chỉ in log (dòng "eta: ... iter ... loss ..."); số liệu vào history.json, không metrics.json /
        tensorboard."""
        return [CommonMetricPrinter(self.max_iter)]


def add_baseline_configs(cfg):
    """Config bổ sung của repo này (không có trong detectron2 / DiffusionDet / Sparse R-CNN gốc)."""
    # không còn dùng (chỉ ghi last.pth / best.pth); giữ khoá để config benchmark cũ (Base-Kaggle-T4x2) nạp được
    cfg.SOLVER.CHECKPOINT_MAX_TO_KEEP = 3
    # tên hàng trong docs/SCORE.md ("BASELINE0"...), predict.py ghi vào dump
    cfg.BASELINE = CN()
    cfg.BASELINE.NAME = ""


def num_classes_of(cfg):
    arch = cfg.MODEL.META_ARCHITECTURE
    if arch not in NUM_CLASSES_KEY:
        raise ValueError(f"META_ARCHITECTURE {arch} chưa hỗ trợ: {sorted(NUM_CLASSES_KEY)}")
    node, key = NUM_CLASSES_KEY[arch]
    return getattr(getattr(cfg.MODEL, node), key)


def check_num_classes(cfg):
    """Bắt lỗi cấu hình số class sai — lỗi này không crash mà chỉ cho kết quả rác."""
    got = num_classes_of(cfg)
    for split, names in (("TRAIN", cfg.DATASETS.TRAIN), ("TEST", cfg.DATASETS.TEST)):
        for name in names:
            expected = dataset_num_classes(name)
            if expected is not None and got != expected:
                node, key = NUM_CLASSES_KEY[cfg.MODEL.META_ARCHITECTURE]
                raise ValueError(
                    f"DATASETS.{split} có '{name}' cần {expected} class nhưng "
                    f"MODEL.{node}.{key} = {got}. Sửa config trước khi train."
                )
    if cfg.SOLVER.AMP.ENABLED:
        raise ValueError("SOLVER.AMP.ENABLED phải False: DiffusionDet vỡ cấu trúc với fp16 (CLAUDE.md)")


def build_cfg(config_file, opts=()):
    """Config đầy đủ (mọi phần mở rộng) — dùng chung cho train_net.py và predict.py."""
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_sparsercnn_config(cfg)
    add_model_ema_configs(cfg)
    add_baseline_configs(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(list(opts))
    return cfg


def setup(args):
    cfg = build_cfg(args.config_file, args.opts)
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
            if args.dump_results:
                # Ghi kết quả ra json để script/notebook đọc bằng máy (parse log thì dễ vỡ).
                os.makedirs(os.path.dirname(os.path.abspath(args.dump_results)), exist_ok=True)
                with open(args.dump_results, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, default=float)
                logging.getLogger("detectron2").info(f"đã ghi kết quả -> {args.dump_results}")
        return res

    trainer = Trainer(cfg)
    trainer.register_hooks([TimeLimit(args.max_hours)])
    trainer.resume_or_load(resume=args.resume)
    try:
        return trainer.train()
    except StopTraining as e:
        logging.getLogger("detectron2").info(f"DỪNG: {e}")
        return None


def get_parser():
    parser = default_argument_parser()
    parser.add_argument("--dump-results", default=None,
                        help="ghi kết quả eval ra file json (chỉ dùng với --eval-only)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="dừng sạch sau số giờ này (ghi checkpoint, chạy lại với --resume)")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
