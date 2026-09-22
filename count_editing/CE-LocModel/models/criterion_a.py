"""Loss cho EXPERIMENT A — giám sát sâu ở MỌI tầng, matcher chạy lại từng tầng.

TÁI DÙNG, KHÔNG CHÉP LẠI: phần tính loss cho MỘT tầng (`SetCriterion._forward_one`)
giữ nguyên từ vòng 1 — cùng trọng số 5,0 L1 + 2,0 GIoU + 2,0 Focal của DiffusionDet,
cùng cách chuẩn hoá `/ num_matched`, cùng `sanitize_boxes` trước GIoU. Chép lại chỉ tạo
ra hai bản dễ lệch nhau.

KHÁC VÒNG 1 ĐÚNG MỘT ĐIỂM: CỘNG loss các tầng thay vì lấy TRUNG BÌNH.

`SetCriterion._forward_rounds` của vòng 1 trả `total / n`. DiffusionDet, DETR và V-DETR
đều CỘNG — mỗi tầng phụ nhận CÙNG trọng số với tầng cuối:

    # refs/repos/DiffusionDet/diffusiondet/detector.py:148-151
    for i in range(self.num_heads - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})   # v GIỮ NGUYÊN
    weight_dict.update(aux_weight_dict)

    # refs/repos/V-DETR/criterion.py:703
    loss += interm_loss

Chia 6 làm gradient tới MỖI tầng bị nhân 1/6, tương đương train nhánh box ở learning
rate thấp hơn 6 lần. Lý do ban đầu của việc chia có lẽ chỉ để con số `loss` in ra so
sánh được giữa cấu hình 1 vòng và 6 vòng — đó là chuyện log, không phải chuyện tối ưu;
`stats["loss_mean"]` ở đây giữ lại tiện ích ấy mà không đụng vào gradient.

CẢNH BÁO KHI CHẠY: gradient lớn hơn 6 lần so với vòng 1. Giữ `lr = 1e-4` trước; nếu
loss phân kỳ thì hạ xuống 5e-5 — đừng quay lại chia trung bình.
"""

import torch

from models.criterion import SetCriterion

__all__ = ["DeepSetCriterion", "loss_from_layers"]


class DeepSetCriterion(SetCriterion):
    """Nhận DANH SÁCH `(boxes, logits)` theo tầng, cũ trước.

    Không có nhánh dispatch theo kiểu dữ liệu như vòng 1 (`crit(boxes, logits, ...)` hay
    `crit(rounds, ...)` tuỳ kiểu đối số đầu): EXPERIMENT A luôn có nhiều tầng, nên chữ
    ký chỉ có một dạng duy nhất.
    """

    def __call__(self, layers, targets, labels=None):
        """
        layers  : list[(boxes [B,N,4], logits [B,N] hoặc [B,N,C])], cũ trước
        targets : list[B] tensor [M_i, 4] cxcywh trong [0,1]
        labels  : list[B] tensor [M_i] long — BẮT BUỘC khi logits là 3 chiều
        -> (loss, stats, indices của tầng cuối)

        Matcher CHẠY LẠI ở mỗi tầng, vì box của tầng 2 khác hẳn box của tầng 6 — ép
        chúng khớp cùng một GT là sai bài toán. Đây là chuẩn của cả 8/8 bài trong khảo
        sát; DiffusionDet gọi lại matcher trong đúng vòng lặp aux
        (`refs/repos/DiffusionDet/diffusiondet/loss.py:253-255`).

        Chi phí: matcher chạy `n_layer` lần. Ở vòng 1 với N=100 matcher đã chiếm 24 %
        một step, nên N=30 của vòng 2 là khoản tiết kiệm quyết định (ma trận 30 x n_gt
        thay vì 100 x n_gt).
        """
        if not isinstance(layers, list) or not layers:
            raise TypeError("DeepSetCriterion cần một list (boxes, logits) khác rỗng")

        total, per_layer, indices = 0.0, [], None
        for boxes, logits in layers:
            loss, st, idx = self._forward_one(boxes, logits, targets, labels)
            total = total + loss                      # CỘNG, không chia
            per_layer.append(st)
            indices = idx                             # giữ assignment của tầng CUỐI

        n = len(per_layer)
        stats = {k: sum(st[k] for st in per_layer) / n for k in per_layer[0]}
        stats["n_layers"] = n
        # `loss` là con số THẬT đi vào backward; `loss_mean` chỉ để so với vòng 1.
        stats["loss"] = float(total)
        stats["loss_mean"] = float(total) / n

        # Đường cong theo tầng là chỉ số CHÍNH để đọc EXPERIMENT A: nếu nó phẳng thì
        # việc cộng dồn không mang lại gì, và kết luận đó phải nhìn thấy được trước khi
        # xây thêm bất cứ thứ gì lên trên.
        for k in ("loss", "iou_matched", "n_matched"):
            stats[f"{k}_per_layer"] = [st[k] for st in per_layer]
        # Các con số tiêu đề mô tả tầng mà suy luận thực sự dùng (`layers[-1]`), để log
        # khớp với thứ `eval.py` báo cáo.
        for k in ("loss", "iou_matched", "n_matched", "loss_l1", "loss_giou", "loss_ce"):
            stats[f"{k}_final"] = per_layer[-1][k]
        return total, stats, indices


def loss_from_layers(crit, layers, targets, labels=None):
    """Gọi criterion trên thứ `model.forward` trả về.

    Tồn tại để mọi công cụ (train / eval / visualise / overfit) đi qua CÙNG một chỗ.
    Vòng 1 có bốn công cụ tự giải nén tuple, và chúng chỉ ném lỗi khi gặp cấu hình
    nhiều vòng — tức là trên server, giữa một lần train dài.

    -> (loss, stats, indices, logits của tầng mà eval dùng)
    """
    loss, st, idx = crit(layers, targets, labels=labels)
    return loss, st, idx, layers[-1][1]
