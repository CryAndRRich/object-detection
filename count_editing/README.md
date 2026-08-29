# object-detection/count_editing — CE-Loc/CE-Gen, bản sửa của dự án `multi_condition`

Sub-project thứ 3 của `object-detection/`, ngang hàng [`../diffusiondet/`](../diffusiondet/README.md)
và [`../diffu_grounding_dino/`](../diffu_grounding_dino/README.md). Đây là bản
**Add One, Take One — Count Editing with Intra-Category Coherence** (NeurIPS 2026,
`../../refs/Add_One_Take_One.md`) — **bài của chính người dùng dự án**, nên thư mục này là nơi
sửa/mở rộng, không phải bản gốc read-only. Bản 100% read-only để đối chiếu nằm ở
`../../refs/repos/Count-Editing/` (đừng nhầm hai thư mục).

**Trọng tâm là CE-LocModel** (chọn box để thêm/bớt vật, không phải sinh ảnh) — xem
`../../CLAUDE.md` để biết vì sao và lộ trình cải tiến đầy đủ. `CE-GenModel/` được giữ nguyên đi
kèm (cùng 1 zip gốc từ tác giả) nhưng dự án hiện không chạm tới.

## Vì sao nằm ở đây thay vì thư mục riêng

Trước đây thư mục này ở `multi_condition/count_editing/` (ngoài `object-detection/`, không phải
git repo). Đã chuyển vào đây để đi chung git repo `object-detection` với DiffusionDet và
DiffuGroundingDINO — trên server chỉ cần `git pull`, không phải zip/unzip thủ công riêng cho
CE-Loc nữa. `../../CLAUDE.md` (rule "mọi thứ lên server qua zip") vẫn đúng cho `data/`/`weights/`
(không push git) — chỉ phần **code** CE-Loc chuyển sang git.

`README_upstream.md` là README gốc của tác giả Count-Editing (không sửa) — đọc `CE-LocModel/README.md`
để biết cách chạy thật (đã viết lại đầy đủ cho bản sửa này).

## Dữ liệu

`../data/samples/` (train/test, dataset train gốc CE-Loc) và `../data/all_phase2_V2/` (dump CE-130
đầy đủ, dùng tính C-NLL) — dùng chung layout `data/` với DiffusionDet/DiffuGroundingDINO
(`../data/README.md`), không nằm trong `count_editing/` nữa. `CE-LocModel/config/default.yaml` và
`test_mul_box.py --all_phase2_dir` trỏ đường dẫn tương đối `../../data/...` — vẫn đúng nguyên vì
`count_editing/` và `data/` chuyển vào `object-detection/` **cùng lúc, cùng cấp**, độ sâu tương đối
không đổi. Đổi vị trí `count_editing/` hoặc `data/` riêng lẻ trong tương lai thì phải sửa lại các
đường dẫn này.

## Trạng thái

Xem `CE-LocModel/README.md` (mục "Ablation kiến trúc" + 3 variant chính `a_repro`/`b_transformer`/
`c_multibox` dưới `config/variants/main/`) và `../../CLAUDE.md` (mục "Việc dở / bước tiếp") để biết
tiến độ chi tiết — **chưa train thật trên server**, mới verify code chạy được cục bộ.
