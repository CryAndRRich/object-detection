import argparse
import datetime
import os
import shutil
import time
import warnings

# cuDNN v8 heuristics đôi khi thử một execution plan không chạy được rồi TỰ
# fallback sang plan khác — kết quả vẫn đúng, chỉ ồn log. Chỉ lọc đúng message
# này, không đụng gì tới cách tính (không bật cudnn.benchmark), nên số liệu
# không đổi. Warning cuDNN khác (nếu có) vẫn hiện bình thường.
warnings.filterwarnings("ignore", message=".*Plan failed with a cudnnException.*")

import torch
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.dataset import ObjectPlacementDataset
from data.ce130_detection_dataset import CE130DetectionDataset
from data.coco_detection_dataset import CocoDetectionDataset
from models.diffusion_module import ObjectPlacementPolicy
from train import load_config, save_training_history


def _format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", type=str, required=True,
        help="config file name under config/variants/ (without .yaml), e.g. resnet34_transformer",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override training.num_epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="override training.batch_size")
    parser.add_argument("--lr", type=float, default=None, help="override training.learning_rate")
    parser.add_argument("--num_workers", type=int, default=None, help="override training.num_workers")
    parser.add_argument(
        "--use_cache", action="store_true",
        help="read pre-resized samples from cache_512.u8 (build with tools/build_cache.py); "
             "profiling showed the loop is 92%% DataLoader without it",
    )
    parser.add_argument(
        "--all_phase2_dir", type=str, default="../../data/all_phase2_V2",
        help="nguồn dữ liệu: multi-box target của bài add (--task add), hoặc toàn "
             "bộ dữ liệu của bài detection (--task detect)",
    )
    parser.add_argument(
        "--task", choices=["add", "detect", "coco"], default="add",
        help="add = bài gốc CE-Loc (sinh box vào vùng trống, đọc samples/ + density). "
             "detect = object detection có điều kiện theo class trên CE-130 "
             "(ground_truth.jpg + all_bboxes, KHÔNG density). "
             "coco = detection đa class trên COCO-minitrain (variant d, có class head).",
    )
    parser.add_argument(
        "--coco_root", type=str, default="../../data",
        help="--task coco: gốc thư mục data/ chứa coco_minitrain/ và coco/",
    )
    parser.add_argument(
        "--no_val", action="store_true",
        help="tắt eval trên split val mỗi epoch (chọn best model theo train loss như "
             "repo gốc). Chỉ dùng khi muốn tái lập đúng hành vi cũ.",
    )
    parser.add_argument(
        "--max_boxes", type=int, default=None,
        help="--task detect: bỏ ảnh có nhiều hơn ngần này box (CE-130 có ảnh tới 1229 box)",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate_loss(model, loader, device, multi_box):
    """Val loss dùng ĐÚNG hàm loss của train, chỉ khác model.eval() + no_grad.

    Lưu ý: loss này vẫn ngẫu nhiên theo timestep t và noise được sample mỗi lần,
    nên giữa 2 epoch nó dao động cả khi model không đổi. Đủ tốt để chọn best
    model và để thấy khoảng cách train/val, nhưng ĐỪNG đọc dao động nhỏ giữa
    các epoch như tín hiệu thật.
    """
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        rgb = batch["pixel_values"].to(device)
        density = (batch["density_map"].to(device) if "density_map" in batch else None)
        text = batch["text"]
        if multi_box:
            loss = model.compute_loss_multibox(
                rgb, density, text,
                batch["boxes"].to(device), batch["box_mask"].to(device),
                gt_labels=batch["labels"].to(device) if "labels" in batch else None)
        else:
            loss = model.compute_loss(rgb, density, text, batch["bbox"].to(device))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Load configs
    train_cfg = load_config("config/default.yaml")
    variant_path = os.path.join("config", "variants", f"{args.variant}.yaml")
    model_cfg = load_config(variant_path)

    if args.epochs is not None:
        train_cfg["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        train_cfg["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        train_cfg["training"]["learning_rate"] = args.lr

    # One result directory per variant so 8 runs don't overwrite each other.
    save_dir = os.path.join("checkpoints", args.variant)
    train_cfg["training"]["save_dir"] = save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Data — variant (c) (noise_net.num_proposals > 1) needs the multi-box
    # target set (all_bboxes from all_phase2_V2) instead of one target_bbox.
    num_proposals = model_cfg["noise_net"].get("num_proposals", 1)
    multi_box = num_proposals > 1
    if args.task == "coco":
        # Variant (d): COCO-minitrain, đa class, có class head, không text.
        train_dataset = CocoDetectionDataset(
            os.path.join(args.coco_root, "coco_minitrain/annotations/instances_minitrain2017.json"),
            os.path.join(args.coco_root, "coco_minitrain/images/train2017"),
            num_proposals=num_proposals, max_boxes=args.max_boxes,
        )
    elif args.task == "detect":
        # Bài detection: ảnh gốc + toàn bộ box cùng class, không density.
        train_dataset = CE130DetectionDataset(
            args.all_phase2_dir, split="train",
            num_proposals=num_proposals, max_boxes=args.max_boxes,
        )
    else:
        data_root = train_cfg["training"]["data"]["train_path"]
        dataset_kwargs = {"use_cache": args.use_cache}
        if multi_box:
            dataset_kwargs.update(multi_box=True, num_proposals=num_proposals,
                                  all_phase2_dir=args.all_phase2_dir)
        train_dataset = ObjectPlacementDataset(data_root, **dataset_kwargs)

    # Split `val` để chọn best model. Repo gốc Count-Editing chỉ có train loss nên
    # `best_model.pth` của nó thực chất là "epoch có train loss thấp nhất" — với
    # dataset nhỏ (1.911 ảnh) thì train loss giảm đều tới cuối, nên "best" gần như
    # luôn là epoch cuối, KHÔNG phải model tổng quát hoá tốt nhất. CE-130 detection
    # có sẵn split val 908 ảnh (overlap với train = 0) nên dùng đúng nó.
    val_dataset = None
    if args.task == "coco" and not args.no_val:
        val_dataset = CocoDetectionDataset(
            os.path.join(args.coco_root, "coco/annotations/instances_val2017.json"),
            os.path.join(args.coco_root, "coco/val2017"),
            num_proposals=num_proposals, max_boxes=args.max_boxes,
        )
    elif args.task == "detect" and not args.no_val:
        val_dataset = CE130DetectionDataset(
            args.all_phase2_dir, split="val",
            num_proposals=num_proposals, max_boxes=args.max_boxes,
        )
    # train.py hardcoded num_workers=4 even though default.yaml carries the key.
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = train_cfg["training"].get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None,
        )
    print(f"Found {len(train_dataset)} training samples "
          f"(cache={'on' if args.use_cache else 'off'}, num_workers={num_workers}, "
          f"multi_box={'on (N=' + str(num_proposals) + ')' if multi_box else 'off'}, "
          f"task={args.task}).")

    # Model
    model = ObjectPlacementPolicy(model_cfg)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=float(train_cfg["training"]["learning_rate"]))

    num_epochs = train_cfg["training"]["num_epochs"]
    num_steps = model_cfg["diffusion"]["num_timesteps"]

    best_loss = float("inf")
    loss_history = []
    val_history = []
    batches_per_epoch = len(train_loader)
    batch_size = train_cfg["training"]["batch_size"]
    train_start = time.monotonic()

    for epoch in range(num_epochs):
        epoch_start = time.monotonic()
        model.train()
        total_loss = 0

        for batch in train_loader:
            rgb = batch["pixel_values"].to(device)
            # Nhánh detect không có density (in_channels=3) -> truyền None.
            density = (batch["density_map"].to(device)
                       if "density_map" in batch else None)
            text = batch["text"]

            optimizer.zero_grad()
            if multi_box:
                gt_boxes = batch["boxes"].to(device)
                box_mask = batch["box_mask"].to(device)
                gt_labels = batch["labels"].to(device) if "labels" in batch else None
                loss = model.compute_loss_multibox(rgb, density, text, gt_boxes, box_mask,
                                                   gt_labels=gt_labels)
            else:
                gt_bbox = batch["bbox"].to(device)
                loss = model.compute_loss(rgb, density, text, gt_bbox)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_loss = evaluate_loss(model, val_loader, device, multi_box) if val_loader else None
        loss_history.append(avg_loss)
        if val_loss is not None:
            val_history.append(val_loss)

        # Tiêu chí chọn best: VAL loss khi có val, train loss khi không.
        # Đây là điểm khác repo gốc (nó chỉ có train loss) — với 1.911 ảnh train
        # thì train loss giảm đều tới cuối nên "best" theo train loss gần như
        # luôn là epoch cuối, không phản ánh khả năng tổng quát hoá.
        score = val_loss if val_loss is not None else avg_loss

        epoch_time = time.monotonic() - epoch_start
        elapsed = time.monotonic() - train_start
        epochs_done = epoch + 1
        avg_epoch_time = elapsed / epochs_done
        eta = avg_epoch_time * (num_epochs - epochs_done)
        is_best = score < best_loss
        # flush=True: nohup buffer stdout theo block, không có nó thì file log
        # đứng im hàng phút rồi mới xả ra một lúc -> tưởng job treo.
        print(
            f"[{args.variant}] Epoch [{epochs_done}/{num_epochs}] "
            f"loss={avg_loss:.4f}"
            + (f" val={val_loss:.4f} gap={val_loss - avg_loss:+.4f}" if val_loss is not None else "")
            + f" best={min(score, best_loss):.4f}"
            f"{' *' if is_best else '  '} "
            f"| lr={optimizer.param_groups[0]['lr']:.2e} "
            f"| {batches_per_epoch} batch x bs={batch_size} "
            f"| epoch {_format_duration(epoch_time)} "
            f"({epoch_time / max(batches_per_epoch, 1) * 1000:.0f} ms/batch) "
            f"| elapsed {_format_duration(elapsed)} "
            f"| ETA {_format_duration(eta)}",
            flush=True,
        )

        save_training_history(save_dir, loss_history)
        if val_history:
            import json as _json
            with open(os.path.join(save_dir, "val_loss_history.json"), "w") as f:
                _json.dump(val_history, f)

        # checkpoint payload always carries the args CE-Loc's own inference.py /
        # test_mul_box.py expect (checkpoint['args']['num_steps']) — the original
        # train.py did not save this, which silently fell back to a default
        # num_timesteps=100 at eval time regardless of what the variant used.
        ckpt_args = {"num_steps": num_steps, "variant": args.variant, "task": args.task}

        if is_best:
            best_loss = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                    "train_loss": avg_loss,
                    "val_loss": val_loss,
                    "args": ckpt_args,
                },
                os.path.join(save_dir, "best_model.pth"),
            )
            print(f"  -> Saved Best Model (Loss: {best_loss:.4f})")

        # Bản gốc chỉ lưu mỗi 10 epoch -> chạy --epochs 205 thì trạng thái cuối
        # (epoch 205) không bao giờ được ghi. Thêm điều kiện epoch cuối cùng.
        if (epoch + 1) % 10 == 0 or epochs_done == num_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": avg_loss,
                    "args": ckpt_args,
                },
                os.path.join(save_dir, "last_model.pth"),
            )

    # README documents checkpoints as shipping alongside model_config_final.yaml
    # and train_config_final.yaml; the original train.py never wrote these out.
    shutil.copy(variant_path, os.path.join(save_dir, "model_config_final.yaml"))
    shutil.copy("config/default.yaml", os.path.join(save_dir, "train_config_final.yaml"))

    total_time = time.monotonic() - train_start
    print(f"Training Complete for variant={args.variant}. Total time: {_format_duration(total_time)}")


if __name__ == "__main__":
    train()
