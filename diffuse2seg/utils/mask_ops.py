"""Mask geometry: latent grid -> original image, and mask IoU.

Diffuse2Seg predicts on a `grid_r x grid_r` latent grid over a square padded
canvas; ground truth lives at the original image resolution. AR is computed
where the ground truth is, so predictions come to it -- never the reverse.
Downsampling GT to 140x140 would quietly delete the small objects, and 68 % of
PACO's targets are small.

Upsampling is NEAREST, on purpose. The mask is the solution of a discrete
problem on the graph; bilinear would invent boundary values that no propagation
produced and make the result depend on the interpolation rather than on the
method. The consequence is honest and must be stated when reading numbers: a
mask edge is quantised to one latent cell, which at canvas 1120 / grid 140 is
8 canvas px, i.e. about 4.6 original px on a typical 640-wide PACO image.
"""

import numpy as np

__all__ = ["masks_to_original", "mask_iou_matrix"]


def masks_to_original(masks, valid_w, valid_h, W, H):
    """(M, r, r) bool on the padded canvas -> (M, H, W) bool on the original.

    The canvas holds the image in its top-left [0, valid_w] x [0, valid_h]
    fraction; everything outside is padding and is dropped, not stretched.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.ndim != 3 or len(masks) == 0:
        return np.zeros((0, H, W), dtype=bool)

    r = masks.shape[1]
    # For each original pixel, which latent cell covers it. Pixel centres, so
    # the sampling is symmetric rather than biased toward the top-left.
    ys = (((np.arange(H) + 0.5) / H) * valid_h * r).astype(np.int64)
    xs = (((np.arange(W) + 0.5) / W) * valid_w * r).astype(np.int64)
    np.clip(ys, 0, r - 1, out=ys)
    np.clip(xs, 0, r - 1, out=xs)
    return masks[:, ys][:, :, xs]


def mask_iou_matrix(pred, gt, chunk=64):
    """(P, H, W) x (G, H, W) bool -> (P, G) IoU.

    Computed in chunks of predictions: the boolean intersection of every pair at
    once would be P*G*H*W bytes -- at P=529, G=279, H*W=640*480 that is 45 TB.
    Flattening to uint8 and using matmul keeps it to one (P, G) result per
    chunk, at the cost of one (chunk, H*W) float buffer.
    """
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    P, G = len(pred), len(gt)
    # Thoát TRƯỚC khi reshape: `np.zeros((0,H,W)).reshape(0, -1)` ném
    # "cannot reshape array of size 0 into shape (0,newaxis)". Ảnh không có mask
    # nào là chuyện CÓ THẬT trên dữ liệu thật (mọi cụm đều dưới a_min=100 px),
    # nên đường này là đường sống, không phải phòng xa.
    if P == 0 or G == 0:
        return np.zeros((P, G), dtype=np.float64)
    pred = pred.reshape(P, -1)
    gt = gt.reshape(G, -1)

    a_pred = pred.sum(axis=1).astype(np.float64)
    a_gt = gt.sum(axis=1).astype(np.float64)
    g = gt.astype(np.float32)

    out = np.zeros((P, G), dtype=np.float64)
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        inter = pred[s:e].astype(np.float32) @ g.T          # (chunk, G)
        union = a_pred[s:e, None] + a_gt[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            out[s:e] = np.where(union > 0, inter / union, 0.0)
    return out
