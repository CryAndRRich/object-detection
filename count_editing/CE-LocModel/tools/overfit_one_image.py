"""CHẨN ĐOÁN 2: model có overfit nổi ĐÚNG MỘT ảnh không?

Đây là sanity check rẻ nhất và dứt khoát nhất trong deep learning: cho model
đúng 1 ảnh, train tới khi loss ~ 0. Nếu KHÔNG làm được thì vấn đề không phải
dữ liệu, không phải dung lượng model, không phải số epoch -- mà là một lỗi
trong chính vòng train (loss/matcher/không gian toạ độ/đường điều kiện hoá).

Vì sao cần ngay lúc này: §15.3 cho thấy loss của (c) dừng ở 3,91, đúng mức của
một model đoán hằng số. Hai khả năng:
  (i)  model KHÔNG THỂ làm tốt hơn vì có lỗi -> overfit 1 ảnh sẽ THẤT BẠI
  (ii) model làm được nhưng không tổng quát hoá -> overfit 1 ảnh sẽ THÀNH CÔNG
Hai khả năng này dẫn tới hai hướng sửa HOÀN TOÀN KHÁC NHAU, nên phải phân biệt
trước khi đụng vào cosine schedule hay RoIAlign.

In kèm mốc so sánh "loss của model đoán hằng số" tính TRÊN CHÍNH ảnh đó, để
biết loss đang giảm thật hay chỉ đang trôi về trung bình.
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


def constant_baseline_loss(gt, model):
    """Loss mà một model dự đoán HẰNG SỐ (box trung bình của chính ảnh này) đạt
    được. Đây là sàn mà bất kỳ model nào cũng chạm tới mà không cần nhìn ảnh --
    loss dừng ở đây nghĩa là chưa học được gì có điều kiện."""
    import torch.nn.functional as TF
    from utils.matcher import generalized_box_iou, _cxcywh_to_xyxy
    const = gt.mean(0, keepdim=True).expand_as(gt)
    l1 = TF.l1_loss(const, gt).item()
    giou = torch.diag(generalized_box_iou(_cxcywh_to_xyxy(const), _cxcywh_to_xyxy(gt))).mean().item()
    return model.l1_weight * l1 + model.giou_weight * (1.0 - giou)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--dataset", choices=["ce130", "coco"], default="ce130")
    p.add_argument("--steps", type=int, default=2000,
                   help="số bước gradient trên CÙNG một ảnh")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="cao hơn train thật (5e-5) vì chỉ cần nhớ 1 mẫu")
    p.add_argument("--image_idx", type=int, default=0)
    p.add_argument("--all_phase2_dir", default="../../data/all_phase2_V2")
    p.add_argument("--coco_root", default="../../data")
    p.add_argument("--split", default="train")
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--fixed_t", type=int, default=None,
                   help="ép MỘT timestep duy nhất thay vì lấy ngẫu nhiên. Dùng để tách "
                        "bạch: nếu overfit được với t cố định thấp mà KHÔNG được với t "
                        "ngẫu nhiên thì thủ phạm là noise schedule (§15.1), không phải "
                        "kiến trúc.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(os.path.join("config", "variants", f"{args.variant}.yaml"))
    N = cfg["noise_net"].get("num_proposals", 1)
    multi = N > 1

    if args.dataset == "coco":
        ds = CocoDetectionDataset(
            os.path.join(args.coco_root, "coco_minitrain/annotations/instances_minitrain2017.json"),
            os.path.join(args.coco_root, "coco_minitrain/images/train2017"), num_proposals=N)
    else:
        ds = CE130DetectionDataset(args.all_phase2_dir, split=args.split, num_proposals=N)

    batch = next(iter(DataLoader([ds[args.image_idx]], batch_size=1)))
    rgb = batch["pixel_values"].to(device)
    text = batch["text"]
    n_real = int(batch["box_mask"][0].sum()) if "box_mask" in batch else 1

    model = ObjectPlacementPolicy(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    if multi:
        gt = batch["boxes"].to(device)
        mask = batch["box_mask"].to(device)
        labels = batch["labels"].to(device) if "labels" in batch else None
        floor = constant_baseline_loss(gt[0][mask[0]], model)
    else:
        gt = batch["bbox"].to(device)
        floor = None

    print(f"\n=== OVERFIT 1 ẢNH: {args.variant} ===")
    print(f"ảnh #{args.image_idx} | {n_real} box thật | N={N} | lr={args.lr} "
          f"| t={'CỐ ĐỊNH ' + str(args.fixed_t) if args.fixed_t is not None else 'ngẫu nhiên'}")
    if floor is not None:
        print(f"MỐC: loss của model đoán HẰNG SỐ trên chính ảnh này = {floor:.4f}")
        print("     -> loss dừng quanh mức này = chưa học được gì có điều kiện\n")

    # Ép một timestep duy nhất bằng cách vá torch.randint trong phạm vi vòng lặp.
    real_randint = torch.randint
    if args.fixed_t is not None:
        def fixed_randint(low, high, size, **kw):
            if len(size) == 1 and high == model.num_timesteps:
                return torch.full(size, args.fixed_t, dtype=torch.long,
                                  device=kw.get("device"))
            return real_randint(low, high, size, **kw)
        torch.randint = fixed_randint

    best = float("inf")
    t0 = time.monotonic()
    try:
        for i in range(1, args.steps + 1):
            opt.zero_grad()
            if multi:
                loss = model.compute_loss_multibox(rgb, None, text, gt, mask, gt_labels=labels)
            else:
                loss = model.compute_loss(rgb, None, text, gt)
            loss.backward()
            opt.step()
            best = min(best, loss.item())
            if i % args.log_every == 0 or i == 1:
                el = time.monotonic() - t0
                extra = f" | so với mốc hằng số: {loss.item() / floor * 100:5.1f}%" if floor else ""
                print(f"  bước {i:5d}/{args.steps}  loss={loss.item():8.4f}  "
                      f"best={best:8.4f}  {el:5.0f}s{extra}", flush=True)
    finally:
        torch.randint = real_randint

    print(f"\n--- KẾT LUẬN ---")
    print(f"loss thấp nhất đạt được: {best:.4f}")
    if floor is not None:
        r = best / floor
        if r > 0.85:
            print(f"THẤT BẠI: {r * 100:.0f}% của mốc hằng số — model KHÔNG overfit nổi 1 ảnh.")
            print("  => Còn LỖI trong vòng train (loss/matcher/toạ độ/điều kiện hoá).")
            print("     Chạy lại với --fixed_t 100: nếu khi đó overfit ĐƯỢC thì thủ phạm")
            print("     là noise schedule (§15.1); nếu vẫn không thì lỗi nằm chỗ khác.")
        elif r > 0.4:
            print(f"MỘT PHẦN: {r * 100:.0f}% của mốc hằng số — có học nhưng rất khó.")
        else:
            print(f"THÀNH CÔNG: {r * 100:.0f}% của mốc hằng số — vòng train hoạt động đúng.")
            print("  => Lỗi KHÔNG nằm ở loss/matcher. Vấn đề là tổng quát hoá:")
            print("     điều kiện hoá quá yếu (1 vector cho cả ảnh) hoặc schedule sai.")
    else:
        print(f"(nhánh 1-box: loss là MSE trên epsilon, ~0 nghĩa là overfit thành công)")


if __name__ == "__main__":
    main()
