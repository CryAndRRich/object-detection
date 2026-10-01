"""The grid of one-hot point prompts that seeds propagation.

This replaces the human in M2N2. M2N2 is INTERACTIVE: a person clicks, and the
four scoring functions that pick its cut threshold (s_prior, s_edge, s_pos,
s_neg) read those clicks back. Take the human away and two of the four become
meaningless, which is why Diffuse2Seg drops that whole mechanism and lets
clusters compete by argmax instead. What survives is the seeding idea, now on a
regular grid -- exactly what SAM's automatic mask generator does.

                    WHY THE STRIDE IS 3 AND NOT THE PAPER'S 6

Measured on 200 CE-130 val images (canvas 512, r=64, so one cell is 8 px):

    stride  prompts  in padding  boxes with >=1 prompt
      2       1024      20 %            99.0 %
      3        441      22 %            96.6 %     <- chosen
      4        256      19 %            91.5 %

The median box short side is 4.65 cells and the median SMALLEST box per image
is 2.24 cells, so the paper's stride of 6 steps clean over small objects. A
prompt that never lands inside an object cannot produce a mask for it, and no
downstream stage recovers that.

                          WHY PADDING MUST BE DROPPED

Every CE-130 image is exactly 384 px tall and 384-1918 wide, so the
aspect-preserving resize always leaves padding at the BOTTOM -- about 29 % of
the canvas at the median aspect ratio of 1.41. A prompt seeded on that flat
CLIP-mean grey propagates freely across it and yields one huge mask covering the
padding. `valid_h` from scale_to_canvas is the boundary, and it is the only
thing standing between the pipeline and a pile of garbage boxes.
"""

import numpy as np
import torch

__all__ = ["build_prompt_grid", "f0_onehot", "cells_to_canvas_xy",
           "points_to_cells"]


def build_prompt_grid(grid_r, stride_cells, valid_h=1.0, min_valid_frac=1.0,
                     valid_w=1.0):
    """Regular grid of seed cells, with padding cells dropped.

    Args:
        grid_r:         latent grid side (64 for a 512 canvas).
        stride_cells:   spacing in cells.
        valid_h:        fraction of canvas height that is real image.
        min_valid_frac: how much of a cell must be inside the real image.
                        1.0 = the whole cell.
        valid_w:        fraction of canvas WIDTH that is real image. Defaults to
                        1.0, which is exactly right for CE-130: every image
                        there is 384 tall and at least that wide, so W >= H and
                        padding only ever lands at the bottom. COCO has portrait
                        images too (427x640 is common), where padding lands on
                        the RIGHT instead -- without this, seeds would be
                        planted on flat grey and propagate into one huge mask.

    Returns:
        (K, 2) int array of (row, col) cell indices.

    The grid is offset by stride//2 so seeds sit near cell centres rather than
    hugging the top-left edge.
    """
    assert 1 <= stride_cells < grid_r, f"stride {stride_cells} out of range for r={grid_r}"
    off = stride_cells // 2
    rows = np.arange(off, grid_r, stride_cells)
    cols = np.arange(off, grid_r, stride_cells)

    # A cell spans rows [r, r+1) in cell units; require enough of it inside.
    keep_r = (rows + min_valid_frac) <= valid_h * grid_r + 1e-9
    keep_c = (cols + min_valid_frac) <= valid_w * grid_r + 1e-9
    rows, cols = rows[keep_r], cols[keep_c]

    if len(rows) == 0:                       # degenerate aspect ratio
        rows = np.array([0])
    if len(cols) == 0:
        cols = np.array([0])

    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    return np.stack([rr.ravel(), cc.ravel()], axis=1).astype(np.int64)


def f0_onehot(cells, grid_r, device="cpu", dtype=torch.float32):
    """(K, N) one-hot seeds -- exactly one 1.0 per row.

    DELIBERATELY NOT M2N2's create_single_point_heatmap. That function splits a
    point bilinearly over four cells and then normalises by MAX, so a row sums
    to anywhere between 1.0 and 4.0 depending on where the click fell inside a
    cell. Here f0 is multiplied by `lam` as the anchor term, so a varying row
    sum would silently make the anchor strength per-prompt -- lam would no
    longer mean one thing. Our prompts sit on cell indices by construction, so
    interpolation would buy nothing anyway.
    """
    cells = np.asarray(cells, dtype=np.int64).reshape(-1, 2)
    K, N = len(cells), grid_r * grid_r
    flat = cells[:, 0] * grid_r + cells[:, 1]
    assert flat.min() >= 0 and flat.max() < N, "prompt cell outside the grid"

    f0 = torch.zeros(K, N, device=device, dtype=dtype)
    f0[torch.arange(K), torch.from_numpy(flat)] = 1.0
    return f0


def cells_to_canvas_xy(cells, grid_r, canvas):
    """Cell (row, col) -> canvas pixel (x, y) at the cell centre. For plots."""
    cell_px = canvas / float(grid_r)
    cells = np.asarray(cells, dtype=np.float64).reshape(-1, 2)
    return np.stack([(cells[:, 1] + 0.5) * cell_px,
                     (cells[:, 0] + 0.5) * cell_px], axis=1)


def points_to_cells(points_xy, W, H, grid_r, canvas, valid_w=1.0, valid_h=1.0,
                    dedup=True):
    """Điểm pixel trên ẢNH GỐC -> ô lưới (row, col). Nghịch đảo của letterbox.

    Dùng khi prompt đến từ nguồn ngoài thay cho lưới đều — ví dụ tâm vật do
    CountGD sinh ra (`data/density_points.json` của ce_localization).

    `points_xy`: (K, 2) toạ độ (x, y) pixel liên tục trên ảnh gốc W×H.
    Trả (M, 2) int64, M <= K: điểm ngoài ảnh bị loại, và nếu `dedup` thì nhiều
    điểm rơi cùng một ô gộp làm một.

    ⚠️ HAI CHỖ DỄ SAI, cả hai đều âm thầm:

    1. Ảnh được resize GIỮ TỈ LỆ rồi dán góc trên-trái, nên vùng thật chỉ chiếm
       `valid_w × valid_h` của canvas. Chia cho `canvas` thay vì cho phần hợp lệ
       sẽ nén toạ độ: trên ảnh dọc 478×640 (valid_w = 0,746) điểm ở mép phải
       lệch 26 %. Ảnh CE-130 luôn W >= H nên `valid_w = 1` và lỗi KHÔNG lộ —
       chỉ lộ trên PACO/COCO.
    2. Trùng ô là chuyện thường: density cho ~20 điểm trên lưới 64×64, nhưng hai
       vật sát nhau có thể rơi cùng một ô 8 px. Không gộp thì `f0_onehot` tạo
       hai prompt giống hệt nhau -> hai soft map trùng -> KL = 0 -> cụm thừa.
    """
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    # pixel ảnh gốc -> pixel canvas -> ô lưới
    sx = valid_w * canvas / float(W)
    sy = valid_h * canvas / float(H)
    cell_px = canvas / float(grid_r)
    col = np.floor(pts[:, 0] * sx / cell_px).astype(np.int64)
    row = np.floor(pts[:, 1] * sy / cell_px).astype(np.int64)

    # Loại điểm ngoài VÙNG THẬT (không chỉ ngoài lưới): điểm rơi vào pad sẽ lan
    # tự do trên nền xám phẳng và sinh một mask khổng lồ.
    #
    # ⚠️ CEIL chứ không FLOOR. Vùng thật hiếm khi kết thúc đúng mép ô: ảnh
    # 640×384 cho valid_h·r = 38,375, nên ô hàng 38 chứa 37,5 % ảnh thật và
    # 62,5 % pad. FLOOR vứt cả hàng đó -> mọi điểm ở DẢI ĐÁY ảnh bị loại âm
    # thầm (đo: điểm (639, 383) biến mất). CE-130 có vật sát đáy nên mất thật.
    # Điểm đã nằm trong ảnh gốc thì ô chứa nó hợp lệ theo định nghĩa — khác
    # `build_prompt_grid`, nơi `min_valid_frac` đòi ô NẰM TRỌN trong ảnh vì ở
    # đó ô được sinh ra mù, không có điểm nào bảo chứng.
    nx = min(int(np.ceil(valid_w * grid_r)), grid_r)
    ny = min(int(np.ceil(valid_h * grid_r)), grid_r)
    keep = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    cells = np.stack([row[keep], col[keep]], axis=1)

    if dedup and len(cells):
        cells = np.unique(cells, axis=0)
    return cells
