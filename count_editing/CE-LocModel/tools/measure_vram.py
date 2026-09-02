"""Đo VRAM đỉnh thật của một variant rồi in ra ngưỡng --min-free-mib nên dùng.

Vì sao cần: `run_on_free_gpu.py` trước đây hard-code 15000 MiB cho MỌI job —
con số đó không đo từ đâu cả. Hệ quả thật (2026-08-31): variant (d) và cả 2
job eval bị bỏ qua trong đúng 1 giây vì không GPU nào còn 15000 MiB, dù thực
tế chúng cần ít hơn nhiều.

Đo bằng `torch.cuda.max_memory_allocated` trên đúng vòng train/eval thật
(không phải ước lượng từ số tham số), rồi quy ra ngưỡng:

    ngưỡng = peak_reserved * SAFETY + CUDA_CONTEXT_MIB

- Nhân SAFETY (1.25) cho phần co giãn theo dữ liệu: batch có nhiều box hơn
  trung bình, phân mảnh của caching allocator.
- CỘNG CUDA_CONTEXT_MIB (800) cho phần KHÔNG co giãn: CUDA context + cuDNN
  workspace nằm ngoài bộ đếm của torch (`nvidia-smi` thấy, `max_memory_allocated`
  không thấy). Nhân hệ số vào hằng số này sẽ sai bản chất.

Dùng `reserved` chứ không phải `allocated` vì `nvidia-smi` -- thứ mà
run_on_free_gpu.py đọc -- nhìn thấy đúng phần caching allocator đã giữ chỗ,
không phải phần đang thực sự dùng.
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_detection_dataset import CE130DetectionDataset  # noqa: E402
from data.coco_detection_dataset import CocoDetectionDataset  # noqa: E402
from models.diffusion_module import ObjectPlacementPolicy  # noqa: E402
from train import load_config  # noqa: E402

SAFETY = 1.25
CUDA_CONTEXT_MIB = 800


def mib(x):
    return x / 1024 ** 2


def build_dataset(args, N):
    if args.dataset == "coco":
        return CocoDetectionDataset(
            os.path.join(args.coco_root, "coco_minitrain/annotations/instances_minitrain2017.json"),
            os.path.join(args.coco_root, "coco_minitrain/images/train2017"),
            num_proposals=N, max_boxes=args.max_boxes)
    return CE130DetectionDataset(args.all_phase2_dir, split="train",
                                 num_proposals=N, max_boxes=args.max_boxes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--mode", choices=["train", "eval"], default="train")
    p.add_argument("--dataset", choices=["ce130", "coco"], default="ce130")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--batches", type=int, default=8)
    p.add_argument("--all_phase2_dir", default="../../data/all_phase2_V2")
    p.add_argument("--coco_root", default="../../data")
    p.add_argument("--max_boxes", type=int, default=None)
    p.add_argument("--sampling_steps", type=int, default=None,
                   help="chỉ dùng cho --mode eval; mặc định lấy từ config")
    p.add_argument("--use_ensemble", action="store_true",
                   help="đo kèm use_ensemble (tốn hơn: giữ box của MỌI bước DDIM)")
    p.add_argument("--k", type=int, default=30,
                   help="chỉ nhánh 1-box: số mẫu song song mỗi ảnh, phải khớp --k của "
                        "eval_detection.py vì K mẫu chạy CÙNG LÚC trong 1 batch")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("LỖI: cần GPU để đo VRAM. Chạy trên server.")
        sys.exit(1)

    cfg = load_config(os.path.join("config", "variants", f"{args.variant}.yaml"))
    N = cfg["noise_net"].get("num_proposals", 1)
    multi = N > 1
    device = torch.device("cuda")

    ds = build_dataset(args, N)
    # num_workers=0: worker process không dùng VRAM, loại chúng ra cho phép đo sạch
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    torch.cuda.reset_peak_memory_stats()
    model = ObjectPlacementPolicy(cfg).to(device)
    model_mib = mib(torch.cuda.max_memory_allocated())

    it = iter(loader)
    if args.mode == "train":
        opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        for i in range(args.batches):
            b = next(it)
            rgb = b["pixel_values"].to(device)
            if multi:
                loss = model.compute_loss_multibox(
                    rgb, None, b["text"], b["boxes"].to(device), b["box_mask"].to(device),
                    gt_labels=b["labels"].to(device) if "labels" in b else None)
            else:
                loss = model.compute_loss(rgb, None, b["text"], b["bbox"].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            # Đo SAU vài batch: bước đầu chưa có trạng thái optimizer (AdamW cấp
            # exp_avg/exp_avg_sq ở lần step ĐẦU TIÊN), nên peak của batch 1 thấp giả tạo.
    else:
        from test_mul_box import sample_boxes_multibox, sample_boxes
        model.eval()
        steps = args.sampling_steps or cfg["diffusion"].get("sampling_steps")
        with torch.no_grad():
            for i in range(args.batches):
                b = next(it)
                rgb = b["pixel_values"].to(device)
                # eval chạy TỪNG ẢNH một (đúng như eval_detection.py), không theo batch
                for j in range(rgb.shape[0]):
                    cond = model.encode_condition(rgb[j:j+1], None, [b["text"][j]])
                    if multi:
                        sample_boxes_multibox(model, cond, device, steps,
                                              use_ensemble=args.use_ensemble)
                    else:
                        sample_boxes(model, cond, device, steps, n_samples=args.k)
                    break  # 1 ảnh/batch là đủ: eval không gộp ảnh

    peak_alloc = mib(torch.cuda.max_memory_allocated())
    peak_res = mib(torch.cuda.max_memory_reserved())
    threshold = int(peak_res * SAFETY + CUDA_CONTEXT_MIB)

    print(f"\n=== {args.variant} | mode={args.mode} | dataset={args.dataset} "
          f"| batch_size={args.batch_size} | N={N} ===")
    print(f"  model weights            : {model_mib:8.0f} MiB")
    print(f"  peak ALLOCATED (đang dùng): {peak_alloc:8.0f} MiB")
    print(f"  peak RESERVED  (đã giữ)   : {peak_res:8.0f} MiB   <- nvidia-smi thấy cái này")
    print(f"  + CUDA context ngoài torch: {CUDA_CONTEXT_MIB:8d} MiB")
    print(f"\n  --min-free-mib ĐỀ XUẤT   : {threshold:8d}   "
          f"(= {peak_res:.0f} x {SAFETY} + {CUDA_CONTEXT_MIB})")
    print(f"  (mặc định cũ là 15000 -- {'THỪA' if threshold < 15000 else 'THIẾU'} "
          f"{abs(15000 - threshold)} MiB)")


if __name__ == "__main__":
    main()
