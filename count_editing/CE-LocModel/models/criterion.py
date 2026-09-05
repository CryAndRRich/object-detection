"""Loss GIỐNG DIFFUSIONDET: 5,0 L1 + 2,0 GIoU + 2,0 Focal.

Trọng số đọc từ `diffusiondet/config.py:34-36,43-44`.

  - L1 + GIoU: CHỈ trên cặp đã match (box không match không có target toạ độ).
  - Focal: trên TOÀN BỘ N slot (slot không match có target rõ ràng: score = 0).
  - Chuẩn hoá / num_boxes = số cặp matched, clamp(min=1) cho ảnh 0 GT.

KHÔNG có loss epsilon (vô nghĩa dưới set-matching: matcher hoán vị nên không tồn
tại epsilon nào vừa thuộc prediction p vừa ứng với GT g -> train theo nhiễu của
box padding). KHÔNG có deep supervision (kiến trúc 1-forward, không có stage
trung gian để supervise).

VÌ SAO CẦN CẢ L1 LẪN GIoU: L1 phạt sai lệch TUYỆT ĐỐI nên với vật CE-130 (median
0,41 % diện tích ảnh) gần như bỏ qua vật nhỏ; GIoU phạt theo TỈ LỆ chồng lấn và
có gradient cả khi hai box rời nhau.

CẠM BẪY SCORE HEAD: focal alpha=0,25 với head không phân biệt được sẽ hội tụ về
một HẰNG SỐ. Vòng 1 ra 0,263 (16 % dương) < ngưỡng 0,5 -> mọi box bị lọc ->
fallback argmax giữ đúng 1 box ("viz chỉ vẽ 1 box", từng bị chẩn đoán nhầm là lỗi
công cụ). Nên lúc inference dùng TOP-K, không dùng ngưỡng tuyệt đối.
"""

import torch
import torch.nn.functional as F

from utils.box_ops import box_iou, cxcywh_to_xyxy, generalized_box_iou, sanitize_boxes
from utils.matcher import match

__all__ = ["SetCriterion"]

W_L1, W_GIOU, W_CLASS = 5.0, 2.0, 2.0
ALPHA, GAMMA = 0.25, 2.0


class SetCriterion:
    def __init__(self, matcher_method="hungarian", **matcher_kw):
        self.method = matcher_method
        self.matcher_kw = matcher_kw

    def __call__(self, pred_boxes, pred_logits, targets):
        """
        pred_boxes  : [B, N, 4] cxcywh [0,1]
        pred_logits : [B, N]
        targets     : list[B] tensor [M_i, 4] cxcywh [0,1]
        -> (loss tổng, dict thành phần, danh sách chỉ số ghép cặp)
        """
        dev = pred_boxes.device
        l1_all, giou_all, iou_all = [], [], []
        tgt_score = torch.zeros_like(pred_logits)
        indices, n_matched = [], 0

        for i, gt in enumerate(targets):
            if gt.numel() == 0:
                indices.append((torch.zeros(0, dtype=torch.long, device=dev),) * 2)
                continue

            pi, gi = match(pred_boxes[i].detach(), gt, pred_logits[i].detach(),
                           method=self.method, **self.matcher_kw)
            indices.append((pi, gi))
            if len(pi) == 0:
                continue
            n_matched += len(pi)
            tgt_score[i, pi] = 1.0

            p, g = pred_boxes[i][pi], gt[gi]
            l1_all.append(F.l1_loss(p, g, reduction="none").sum(-1))

            p_xyxy = sanitize_boxes(cxcywh_to_xyxy(p))
            g_xyxy = cxcywh_to_xyxy(g)
            giou = torch.diagonal(generalized_box_iou(p_xyxy, g_xyxy))
            giou_all.append(1.0 - giou)
            # IoU THẬT (>= 0) để báo cáo, KHÔNG phải GIoU (có thể âm khi rời nhau).
            # Chỉ số theo dõi mà sai thì vô dụng — đây là thứ dùng để biết toạ độ
            # có tốt lên không, tách khỏi loss.
            iou_all.append(torch.diagonal(box_iou(p_xyxy, g_xyxy)[0]).detach())

        den = max(n_matched, 1)
        loss_l1 = torch.cat(l1_all).sum() / den if l1_all else pred_boxes.sum() * 0.0
        loss_giou = torch.cat(giou_all).sum() / den if giou_all else pred_boxes.sum() * 0.0

        loss_ce = sigmoid_focal_loss(pred_logits, tgt_score).sum() / den

        total = W_L1 * loss_l1 + W_GIOU * loss_giou + W_CLASS * loss_ce
        stats = {
            "loss": float(total),
            "loss_l1": float(loss_l1),
            "loss_giou": float(loss_giou),
            "loss_ce": float(loss_ce),
            "n_matched": n_matched,
            "iou_matched": float(torch.cat(iou_all).mean()) if iou_all else 0.0,
        }
        return total, stats, indices


def sigmoid_focal_loss(logits, targets, alpha=ALPHA, gamma=GAMMA):
    """Focal loss dạng sigmoid — giống DiffusionDet (`use_focal=True`).

    1 chiều + sigmoid TƯƠNG ĐƯƠNG 2 chiều + softmax về mặt toán học (softmax chỉ
    phụ thuộc HIỆU hai logit -> một chiều tự do thừa). DiffusionDet với focal cũng
    dùng 80 chiều chứ KHÔNG phải 81 — "nền" = mọi logit đều thấp.
    """
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        loss = (alpha * targets + (1 - alpha) * (1 - targets)) * loss
    return loss
