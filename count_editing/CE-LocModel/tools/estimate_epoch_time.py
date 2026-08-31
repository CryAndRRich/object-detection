"""Ước tính thời gian 1 epoch cho nhánh detection, KHÔNG train thật.

Chạy đúng vòng train (DataLoader -> H2D -> vision -> text -> noise net -> loss ->
backward -> step) trên một số batch nhỏ rồi ngoại suy ra cả epoch, đồng thời tách
bạch thời gian từng phần để biết nút thắt nằm ở đâu (bài học từ nhánh add: 92,6%
thời gian là chờ DataLoader, không phải GPU).

Mặc định đo cả 3 variant detect để so trực tiếp trong một lần chạy.
"""
import argparse
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_detection_dataset import CE130DetectionDataset  # noqa: E402
from data.coco_detection_dataset import CocoDetectionDataset  # noqa: E402
from models.diffusion_module import ObjectPlacementPolicy  # noqa: E402
from train import load_config  # noqa: E402


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure(variant, args, train_cfg):
    cfg_path = os.path.join("config", "variants", f"{variant}.yaml")
    model_cfg = load_config(cfg_path)
    N = model_cfg["noise_net"].get("num_proposals", 1)
    multi_box = N > 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.task == "coco":
        ds = CocoDetectionDataset(
            os.path.join(args.coco_root, "coco_minitrain/annotations/instances_minitrain2017.json"),
            os.path.join(args.coco_root, "coco_minitrain/images/train2017"),
            num_proposals=N, max_boxes=args.max_boxes)
    else:
        ds = CE130DetectionDataset(args.all_phase2_dir, split="train",
                                   num_proposals=N, max_boxes=args.max_boxes)
    bs = args.batch_size or train_cfg["training"]["batch_size"]
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0,
                        prefetch_factor=4 if args.num_workers > 0 else None)

    model = ObjectPlacementPolicy(model_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    n_batches = len(loader)
    it = iter(loader)

    # warmup: loại chi phí khởi động worker / cuDNN autotune ra khỏi số đo
    for _ in range(min(args.warmup, n_batches)):
        b = next(it)
        rgb = b["pixel_values"].to(device)
        if multi_box:
            loss = model.compute_loss_multibox(
                rgb, None, b["text"], b["boxes"].to(device), b["box_mask"].to(device),
                gt_labels=b["labels"].to(device) if "labels" in b else None)
        else:
            loss = model.compute_loss(rgb, None, b["text"], b["bbox"].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    sync()

    t_data = t_h2d = t_fwd = t_bwd = 0.0
    n = 0
    wall0 = time.perf_counter()
    while n < args.batches:
        t0 = time.perf_counter()
        try:
            b = next(it)
        except StopIteration:
            break
        t1 = time.perf_counter(); t_data += t1 - t0

        rgb = b["pixel_values"].to(device, non_blocking=True)
        if multi_box:
            gt = b["boxes"].to(device, non_blocking=True)
            mask = b["box_mask"].to(device, non_blocking=True)
        else:
            gt = b["bbox"].to(device, non_blocking=True)
        sync(); t2 = time.perf_counter(); t_h2d += t2 - t1

        opt.zero_grad()
        if multi_box:
            loss = model.compute_loss_multibox(
                rgb, None, b["text"], gt, mask,
                gt_labels=b["labels"].to(device) if "labels" in b else None)
        else:
            loss = model.compute_loss(rgb, None, b["text"], gt)
        sync(); t3 = time.perf_counter(); t_fwd += t3 - t2

        loss.backward(); opt.step()
        sync(); t4 = time.perf_counter(); t_bwd += t4 - t3
        n += 1

    wall = time.perf_counter() - wall0
    per_batch = wall / max(n, 1)
    epoch_s = per_batch * n_batches

    print(f"\n=== {variant} ===")
    print(f"  dataset={len(ds)} ảnh | batch_size={bs} | batches/epoch={n_batches} "
          f"| num_workers={args.num_workers} | N={N}")
    print(f"  đo {n} batch, tổng {wall:.1f}s -> {per_batch*1000:.0f} ms/batch")
    print(f"  ƯỚC TÍNH 1 EPOCH: {epoch_s/60:.1f} phút")
    for ep in (args.epochs,):
        print(f"  -> {ep} epoch: {epoch_s*ep/3600:.1f} giờ ({epoch_s*ep/86400:.2f} ngày)")
    rows = [("DataLoader (chờ I/O+decode)", t_data), ("Chuyển CPU->GPU", t_h2d),
            ("Forward + loss", t_fwd), ("Backward + optimizer", t_bwd)]
    tot = sum(r[1] for r in rows) or 1.0
    print(f"  {'phần':<30}{'ms/batch':>10}{'%':>8}{'phút/epoch':>13}")
    print("  " + "-" * 59)
    for name, v in sorted(rows, key=lambda r: -r[1]):
        print(f"  {name:<30}{v/max(n,1)*1000:>10.1f}{v/tot*100:>7.1f}%"
              f"{v/max(n,1)*n_batches/60:>13.1f}")
    return epoch_s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None,
                   help="mặc định: đo cả 3 variant detect")
    p.add_argument("--all_phase2_dir", default="../../data/all_phase2_V2")
    p.add_argument("--task", choices=["detect", "coco"], default="detect",
                   help="detect = CE-130 (variant a/b/c). coco = COCO-minitrain (variant d).")
    p.add_argument("--coco_root", default="../../data")
    p.add_argument("--batches", type=int, default=25, help="số batch dùng để đo")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--max_boxes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=200,
                   help="chỉ để quy đổi ra tổng thời gian, không chạy thật")
    args = p.parse_args()

    need = (os.path.join(args.coco_root, "coco_minitrain/annotations/instances_minitrain2017.json")
            if args.task == "coco" else args.all_phase2_dir)
    if not os.path.exists(need):
        print(f"LỖI: không thấy {need}\n"
              f"  Dữ liệu nằm trong data/ nên KHÔNG đi theo git pull — cần copy/giải nén "
              f"lên server rồi trỏ --all_phase2_dir / --coco_root vào đúng chỗ.")
        sys.exit(1)

    train_cfg = load_config("config/default.yaml")
    if args.variant:
        variants = [args.variant]
    elif args.task == "coco":
        variants = ["detect/d_coco_classhead"]
    else:
        variants = ["detect/a_cnn_1box", "detect/b_transformer_1box",
                    "detect/c_transformer_multibox"]
    print(f"Thiết bị: {'cuda' if torch.cuda.is_available() else 'CPU (không có GPU!)'}")
    totals = {}
    for v in variants:
        totals[v] = measure(v, args, train_cfg)

    if len(totals) > 1:
        print("\n=== TỔNG HỢP ===")
        grand = 0.0
        for v, s in totals.items():
            print(f"  {v:<34} {s/60:6.1f} phút/epoch -> {s*args.epochs/3600:7.1f} giờ "
                  f"cho {args.epochs} epoch")
            grand += s * args.epochs
        print(f"  {'CẢ 3 VARIANT':<34} {grand/3600:.1f} giờ ({grand/86400:.1f} ngày)")


if __name__ == "__main__":
    main()
