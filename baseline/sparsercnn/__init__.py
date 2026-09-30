# Sparse R-CNN (Peize Sun et al., CVPR 2021), vendored từ github.com/PeizeSun/SparseR-CNN
# @ 0e5028da8edd4d0ab2a31b85e511a7af196529d2, thư mục projects/SparseRCNN/sparsercnn/ (MIT, xem LICENSE).
# config.py / detector.py / head.py / loss.py giữ nguyên, trừ các dòng import `util` đánh dấu
# `[baseline]`: util/ của Sparse R-CNN trùng hệt util/ của DiffusionDet (DiffusionDet được viết từ
# Sparse R-CNN), bản của DiffusionDet đã sửa nhánh torchvision < 0.7 (đọc "0.21" thành 0,2 nên luôn
# đúng rồi import symbol đã bị xoá) -> dùng chung `baseline.diffusiondet.util`.
# Sửa 1 lỗi của bản gốc (loss.py, `loss_boxes`, đánh dấu `[baseline]`): ảnh có nhiều GT hơn số proposal làm
# `image_size` lệch số cặp đã ghép -> RuntimeError (COCO không có ca này; CE-130 train có ảnh > 300 vật).
# Không chép: dataset_mapper (mọi baseline dùng DiffusionDetDatasetMapper), TTA, util/ vẽ hình.
from .config import add_sparsercnn_config
from .detector import SparseRCNN
