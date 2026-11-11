"""Vẽ dùng chung cho các tool hình (plot_cnll_worst, plot_review_filter, vlm/plot_compare)."""


def draw_boxes(ax, boxes, color, lw=1.5, ls="-", alpha=1.0):
    """Viền chữ nhật không tô cho từng box xyxy (toạ độ ảnh của `ax`)."""
    from matplotlib.patches import Rectangle
    for b in boxes:
        ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, edgecolor=color, linewidth=lw, linestyle=ls,
                               alpha=alpha))
