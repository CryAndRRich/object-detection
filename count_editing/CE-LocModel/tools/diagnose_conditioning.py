"""CHẨN ĐOÁN 1: model có THẬT SỰ điều kiện hoá theo ảnh/text không?

Xuất phát từ §15.3 của docs/ce-loc-detection-results.md: loss của (c) hội tụ
tại 3,91, đúng bằng loss tính được của một model dự đoán HẰNG SỐ (box trung
bình, bỏ qua hoàn toàn đầu vào). Nếu đúng vậy thì mọi cải tiến về diffusion
(schedule, số bước DDIM, ensemble) đều vô nghĩa — phải sửa điều kiện hoá trước.

Script này KHÔNG suy luận từ loss mà đo trực tiếp, bằng cách cố định seed rồi
so box sinh ra khi ĐỔI đầu vào. Cùng seed + cùng nhiễu khởi tạo, nên mọi khác
biệt trong đầu ra CHỈ có thể đến từ điều kiện.

Ba phép đo, mỗi phép trả lời một câu khác nhau:

  A. ĐỔI ẢNH, giữ text     -> box có đổi không? (điều kiện hoá thị giác)
  B. ĐỔI TEXT, giữ ảnh     -> box có đổi không? (điều kiện hoá ngôn ngữ; bỏ
                              qua với variant (d) vì nó tắt hẳn nhánh text)
  C. so với BOX TRUNG BÌNH -> box sinh ra gần GT của ảnh đó, hay gần trung
                              bình toàn tập? Đây là phép đo quyết định.

Cách đọc kết quả (in sẵn ở cuối, không phải tự suy):
  - A ~ 0  => model MÙ với ảnh. Mọi thứ khác phải dừng lại để sửa cái này.
  - C: nếu d(pred, GT_ảnh_này) >= d(pred, box_trung_bình) thì model chỉ đang
    tái tạo thống kê của tập, không nhìn ảnh -> xác nhận §15.3.
"""
import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.ce130_detection_dataset import CE130DetectionDataset  # noqa: E402
from data.coco_detection_dataset import CocoDetectionDataset  # noqa: E402
from models.diffusion_module import ObjectPlacementPolicy  # noqa: E402
from train import load_config  # noqa: E402
from test_mul_box import sample_boxes_multibox, sample_boxes  # noqa: E402
from utils.matcher import box_iou_normalized, _cxcywh_to_xyxy  # noqa: E402


def gen(model, cond, device, steps, seed, multi, k):
    """Sinh box với seed CỐ ĐỊNH -> nhiễu khởi tạo giống hệt nhau giữa các lần
    gọi, nên chênh lệch đầu ra chỉ có thể do `cond`."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    if multi:
        r = sample_boxes_multibox(model, cond, device, steps)
        return (r[0] if isinstance(r, tuple) else r).detach()
    return sample_boxes(model, cond.expand(k, -1), device, None, k).detach()


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--dataset", choices=["ce130", "coco"], default="ce130")
    p.add_argument("--split", default="test")
    p.add_argument("--all_phase2_dir", default="../../data/all_phase2_V2")
    p.add_argument("--coco_root", default="../../data")
    p.add_argument("--n_images", type=int, default=20,
                   help="số ảnh đem so; 20 là đủ vì hiệu ứng nếu có thì rất rõ")
    p.add_argument("--k", type=int, default=30, help="nhánh 1-box: số mẫu mỗi ảnh")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.dirname(args.checkpoint)
    model_cfg = yaml.safe_load(open(os.path.join(ckpt_dir, "model_config_final.yaml")))
    # train_w_args.py lưu checkpoint dạng DICT (epoch/model_state_dict/optimizer/...),
    # không phải state_dict trần — giống eval_detection.py:206-215. Chấp nhận cả hai
    # dạng để còn dùng được với checkpoint cũ nếu có.
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    # num_timesteps phải lấy từ chính lần train đó, nếu không alphas_cumprod sẽ
    # khác kích thước với lúc train (eval_detection.py:207-208 làm y hệt).
    if isinstance(ckpt, dict) and ckpt.get("args", {}).get("num_steps"):
        model_cfg.setdefault("diffusion", {})["num_timesteps"] = ckpt["args"]["num_steps"]
    model = ObjectPlacementPolicy(model_cfg).to(device)
    model.load_state_dict(state)
    model.eval()
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        # val_loss là None khi train với --no_val, nên không format cứng bằng :.4f
        vl = ckpt.get("val_loss")
        print(f"checkpoint: epoch {ckpt['epoch']} | "
              f"train_loss={ckpt.get('train_loss', ckpt.get('loss'))} | "
              f"val_loss={vl if vl is not None else 'không có (--no_val)'}")

    N = model.num_proposals
    multi = N > 1
    steps = model_cfg.get("diffusion", {}).get("sampling_steps", 4)

    if args.dataset == "coco":
        ds = CocoDetectionDataset(
            os.path.join(args.coco_root, "coco/annotations/instances_val2017.json"),
            os.path.join(args.coco_root, "coco/val2017"), num_proposals=N)
    else:
        ds = CE130DetectionDataset(args.all_phase2_dir, split=args.split, num_proposals=N)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    imgs, texts, gts = [], [], []
    for i, b in enumerate(loader):
        if i >= args.n_images:
            break
        imgs.append(b["pixel_values"].to(device))
        texts.append(b["text"])
        rec = ds.samples[i]
        if args.dataset == "coco":
            gts.append(torch.stack([ds._normalize_bbox(x[:4], b["scale"].item())
                                    for x in rec["boxes"]]))
        else:
            gts.append(torch.stack([ds._normalize_bbox(ds._xyxy_to_cxcywh(x), b["scale"].item())
                                    for x in rec["boxes_xyxy"]]))

    print(f"\nModel: {args.variant} | {len(imgs)} ảnh | N={N} | steps={steps} "
          f"| text {'TẮT' if not model.use_text else 'BẬT'}\n")

    # ---- A. đổi ảnh, giữ text -------------------------------------------
    preds = [gen(model, model.encode_condition(imgs[i], None, texts[i]),
                 device, steps, args.seed, multi, args.k) for i in range(len(imgs))]
    base = preds[0]
    diffs = [(preds[i] - base).abs().mean().item() for i in range(1, len(preds))]
    a_mean = sum(diffs) / len(diffs)
    print(f"A. ĐỔI ẢNH (seed cố định): |Δbox| trung bình = {a_mean:.6f}")
    print(f"   min={min(diffs):.6f}  max={max(diffs):.6f}")

    # ---- B. đổi text, giữ ảnh -------------------------------------------
    b_mean = None
    if model.use_text:
        uniq = []
        for t in texts:
            if t[0] not in [u[0] for u in uniq]:
                uniq.append(t)
        if len(uniq) >= 2:
            pb = [gen(model, model.encode_condition(imgs[0], None, u),
                      device, steps, args.seed, multi, args.k) for u in uniq[:5]]
            db = [(pb[i] - pb[0]).abs().mean().item() for i in range(1, len(pb))]
            b_mean = sum(db) / len(db)
            print(f"\nB. ĐỔI TEXT trên CÙNG ảnh: |Δbox| trung bình = {b_mean:.6f}")
            print(f"   text đã thử: {[u[0] for u in uniq[:5]]}")
        else:
            print("\nB. bỏ qua — tập mẫu chỉ có 1 class")
    else:
        print("\nB. bỏ qua — variant này tắt nhánh text")

    # ---- C. so với box trung bình (phép đo quyết định) -------------------
    all_gt = torch.cat(gts, 0)
    mean_box = all_gt.mean(0, keepdim=True)
    print(f"\nC. Box sinh ra gần GT CỦA ẢNH ĐÓ, hay gần BOX TRUNG BÌNH của tập?")
    print(f"   box trung bình toàn tập = {[round(v, 3) for v in mean_box[0].tolist()]}")

    d_gt, d_mean, iou_gt, iou_mean = [], [], [], []
    for i, pr in enumerate(preds):
        g = gts[i].to(device)
        pr_c = pr.to(device)
        # với mỗi box dự đoán: khoảng cách tới GT GẦN NHẤT của chính ảnh này
        d_gt.append(torch.cdist(pr_c, g, p=1).min(dim=1).values.mean().item())
        d_mean.append((pr_c - mean_box.to(device)).abs().sum(-1).mean().item())
        iou = box_iou_normalized(_cxcywh_to_xyxy(pr_c), _cxcywh_to_xyxy(g))
        iou_gt.append(iou.max(dim=1).values.mean().item())
        iou_mean.append(box_iou_normalized(
            _cxcywh_to_xyxy(pr_c), _cxcywh_to_xyxy(mean_box.to(device))).mean().item())

    dg, dm = sum(d_gt) / len(d_gt), sum(d_mean) / len(d_mean)
    ig, im = sum(iou_gt) / len(iou_gt), sum(iou_mean) / len(iou_mean)
    print(f"   L1 tới GT gần nhất của ảnh   : {dg:.4f}")
    print(f"   L1 tới box trung bình của tập: {dm:.4f}")
    print(f"   IoU với GT tốt nhất của ảnh  : {ig:.4f}")
    print(f"   IoU với box trung bình       : {im:.4f}")

    # ---- KẾT LUẬN --------------------------------------------------------
    print("\n" + "=" * 68)
    THR = 1e-4
    if a_mean < THR:
        print(f"[A] MÙ VỚI ẢNH — đổi hẳn ảnh mà box gần như không đổi ({a_mean:.2e}).")
        print("    Đây là lỗi chi phối. Sửa cosine schedule hay tăng bước DDIM đều")
        print("    KHÔNG cứu được, vì đường điều kiện hoá không mang thông tin.")
    elif a_mean < 1e-2:
        print(f"[A] Điều kiện hoá RẤT YẾU ({a_mean:.2e}) — có tín hiệu nhưng nhỏ.")
    else:
        print(f"[A] Có điều kiện hoá theo ảnh ({a_mean:.4f}).")

    if b_mean is not None and b_mean < THR:
        print(f"[B] MÙ VỚI TEXT ({b_mean:.2e}) — nhánh text không đóng góp gì.")

    if dg >= dm:
        print(f"[C] XÁC NHẬN §15.3: box sinh ra KHÔNG gần GT của ảnh hơn là gần box")
        print(f"    trung bình của tập ({dg:.4f} vs {dm:.4f}). Model đang tái tạo")
        print(f"    THỐNG KÊ CỦA TẬP, không phải đọc ảnh.")
    else:
        print(f"[C] Box có bám GT của ảnh hơn box trung bình ({dg:.4f} < {dm:.4f}),")
        print(f"    tỉ lệ {dm / max(dg, 1e-9):.2f}x -> model CÓ học điều kiện hoá thật.")
    print("=" * 68)


if __name__ == "__main__":
    main()
