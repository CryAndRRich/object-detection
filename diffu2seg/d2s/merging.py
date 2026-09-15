r"""Algorithm 2 — mask merging: soft object maps -> multi-granularity instances.

A LINE-BY-LINE IMPLEMENTATION of the paper's Algorithm 2 (arXiv 2609.06491,
§3.4 + §A.1). The pseudo-code, verbatim:

     1: p_k = f_k / sum_i f_k,i                       for all k
     2: d_KL = 0.5 * (KL(p_k || p_k') + KL(p_k' || p_k))
     3: P <- {}
     4: for h in H:
     5:     {C_1^h, ..., C_Kh^h} <- AgglomCluster(D, h)
     6:     pbar_c^h = (1/|C_c^h|) * sum_{k in C_c^h} p_k
     7:     ptilde_c^h <- Upsample(pbar_c^h, H x W)
     8:     M^h(x) <- argmax_c ptilde_c^h(x)
     9:     for each label c in M^h:
    10:         r <- ConnectedComponents(M^h)
    11:         if area(r) >= a_min: P <- P + {r}
    12: sort P by descending area; M <- {}
    13: for p in P:
    14:     if |M| < N_max and max_{q in M} IoU(p, q) <= tau_IoU: M <- M + {p}

                 THREE THINGS STAGE 1 GOT STRUCTURALLY WRONG

This is not a refinement of stage 1's readout -- it is a different operation,
and the differences are exactly where a "close enough" implementation fails:

1. NO PER-MAP THRESHOLD. Stage 1 thresholded each propagated map at a quantile
   and took the component containing its seed. Algorithm 2 never thresholds a
   map: it AVERAGES the maps within a cluster, upsamples, and takes an ARGMAX
   ACROSS CLUSTERS. Every pixel therefore belongs to exactly one cluster -- the
   result is a PARTITION of the image, not a set of independent blobs. That is
   why no `quantile` or `rel_floor` appears anywhere below.

2. UPSAMPLE BEFORE ARGMAX (line 7 before line 8). Doing it the other way round
   would quantise every boundary to one latent cell (8 canvas px). Upsampling
   the continuous maps first lets the argmax boundary fall between cells.

3. NMS IS AREA-DESCENDING WITH tau_IoU = 0.9, not a dedup at 0.7. Large masks
   are kept first and the threshold is deliberately permissive, because adjacent
   granularity levels legitimately produce NESTED masks -- a chair at h=2.99 and
   its seat at h=0.186 are both wanted. The paper measured that tightening this
   to 0.5 "reduces recall significantly by 3.1 p.p. in mAR".

                       WHAT THE PAPER DOES NOT SAY

`kl_eps`: a propagated map is zero over most of the image, and KL takes
log(p/q), so log(0) = -inf poisons the entire distance matrix. Same class of
omission as `g_eps` in the propagation step. Clamping before the log is the
only way the formula is computable at all.
"""

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import label as cc_label
from scipy.spatial.distance import squareform

__all__ = ["normalise_maps", "symmetric_kl_matrix", "cluster_at_heights",
           "masks_from_clusters", "area_descending_nms", "merge_maps_to_masks"]


def normalise_maps(f, eps=1e-12):
    """Line 1: p_k = f_k / sum_i f_k,i. (K, N) -> (K, N), each row sums to 1.

    A row that is entirely zero (every prompt that propagated nowhere) would
    divide by zero; those rows come back as zeros and are dropped by the caller
    rather than turned into a uniform distribution, which would claim the prompt
    saw the whole image equally.
    """
    f = np.asarray(f, dtype=np.float64)
    f = np.clip(f, 0.0, None)
    s = f.sum(axis=1, keepdims=True)
    out = np.zeros_like(f)
    ok = s[:, 0] > eps
    out[ok] = f[ok] / s[ok]
    return out, ok


def symmetric_kl_matrix(p, eps=1e-12, chunk=64):
    """Line 2: the (K, K) symmetric KL distance matrix.

        d(k, k') = 0.5 * ( KL(p_k || p_k') + KL(p_k' || p_k) )

    Expanded so it costs two matmuls instead of a K x K x N tensor. With
    q = clip(p, eps):

        KL(p_a || p_b) = sum_i p_a,i log p_a,i - sum_i p_a,i log p_b,i
                       = -H(p_a)  -  (p_a @ log q_b)

    The first term depends on `a` only; the second is a single (K, K) matmul.
    The literal form builds K*K*N doubles -- at K=529, N=19600 that is 43 GB.

    Note `d(k,k) = 0` exactly and `d >= 0`, both of which scipy's linkage
    requires of a condensed distance vector; small negatives from rounding are
    clamped.
    """
    p = np.asarray(p, dtype=np.float64)
    K = len(p)
    if K == 0:
        return np.zeros((0, 0))

    logq = np.log(np.clip(p, eps, None))
    # -H(p_a) = sum_i p_a,i log p_a,i, one value per row.
    neg_ent = np.einsum("ij,ij->i", p, logq)

    cross = np.empty((K, K), dtype=np.float64)
    for s in range(0, K, chunk):
        e = min(s + chunk, K)
        cross[s:e] = p[s:e] @ logq.T          # cross[a, b] = sum_i p_a,i log q_b,i

    kl = neg_ent[:, None] - cross             # KL(p_a || p_b)
    d = 0.5 * (kl + kl.T)
    np.fill_diagonal(d, 0.0)
    return np.clip(d, 0.0, None)


def cluster_at_heights(d, heights):
    """Lines 5-6: average-linkage agglomerative clustering, cut at each height.

    Returns a list of (K,) integer label arrays, one per height, labels from 0.

    `average` linkage is the paper's choice ("agglomerative average linkage
    clustering on D"), and `fcluster(..., 'distance')` is the dendrogram cut at
    height h. scipy wants the distance matrix in condensed form, which also
    checks the symmetry and zero diagonal that `symmetric_kl_matrix` guarantees.
    """
    d = np.asarray(d, dtype=np.float64)
    K = len(d)
    if K == 0:
        return [np.zeros(0, dtype=np.int64) for _ in heights]
    if K == 1:
        return [np.zeros(1, dtype=np.int64) for _ in heights]

    Z = linkage(squareform(d, checks=False), method="average")
    return [fcluster(Z, t=float(h), criterion="distance").astype(np.int64) - 1
            for h in heights]


def _upsample_nearest(maps, grid_r, H, W, valid_w=1.0, valid_h=1.0):
    """(C, N) on the latent grid -> (C, H, W) at original resolution.

    NEAREST, not bilinear. The soft maps are the solution of a discrete problem
    on a graph; bilinear would invent values between tokens and make the argmax
    boundary depend on the interpolation rather than on the propagation. The
    paper says "Upsample" without specifying, and nearest is the choice that
    adds nothing the method did not produce.

    `valid_w`/`valid_h` cut away the padded part of the canvas: the image
    occupies only the top-left fraction, and stretching the padding into the
    result would map grey canvas onto real pixels.
    """
    C = len(maps)
    # float32: kết quả chỉ dùng cho argmax và so > 0, nên độ chính xác float64
    # không mua được gì và tốn gấp đôi. Đo ở kịch bản xấu nhất (529 cụm,
    # 640x480): 1,30 GB float64 -> 0,65 GB float32, và việc này lặp cho MỖI
    # trong 6 mức.
    m = maps.astype(np.float32, copy=False).reshape(C, grid_r, grid_r)
    ys = (((np.arange(H) + 0.5) / H) * valid_h * grid_r).astype(np.int64)
    xs = (((np.arange(W) + 0.5) / W) * valid_w * grid_r).astype(np.int64)
    np.clip(ys, 0, grid_r - 1, out=ys)
    np.clip(xs, 0, grid_r - 1, out=xs)
    return m[:, ys][:, :, xs]


def masks_from_clusters(p, labels, grid_r, H, W, min_area_px=100,
                        valid_w=1.0, valid_h=1.0, connectivity=1):
    """Lines 6-11 for ONE height: average, upsample, argmax, components.

    Returns a list of (H, W) bool masks whose area is at least `min_area_px`.

    The argmax is over CLUSTERS, so every pixel is assigned to exactly one and
    the result partitions the image. Connected components then split a cluster
    that covers two separate regions (two chairs merged by similarity) into two
    instances -- this is the step that makes them INSTANCE masks rather than
    semantic regions.
    """
    if len(p) == 0:
        return []

    n_cl = int(labels.max()) + 1 if len(labels) else 0
    if n_cl == 0:
        return []

    # Line 6: mean of the maps in each cluster.
    bar = np.zeros((n_cl, p.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=n_cl).astype(np.float64)
    np.add.at(bar, labels, p)
    bar /= np.maximum(counts, 1.0)[:, None]

    # Lines 7-8: upsample THEN argmax.
    up = _upsample_nearest(bar, grid_r, H, W, valid_w, valid_h)
    seg = up.argmax(axis=0)

    # A cluster that wins nowhere contributes nothing; a pixel where every
    # cluster is 0 would otherwise be handed to cluster 0 by argmax's tie rule.
    nothing = up.max(axis=0) <= 0.0

    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) if connectivity == 1 \
        else np.ones((3, 3))

    out = []
    for c in range(n_cl):
        region = (seg == c) & ~nothing
        if not region.any():
            continue
        lab, n = cc_label(region, structure=structure)
        for j in range(1, n + 1):
            comp = lab == j
            if comp.sum() >= min_area_px:
                out.append(comp)
    return out


def area_descending_nms(masks, iou_thr=0.9, max_masks=1000):
    """Lines 12-14: keep large masks first, reject anything overlapping > tau.

    DELIBERATELY PERMISSIVE at 0.9. Adjacent granularity levels produce nested
    masks on purpose -- a chair and its seat are both wanted -- and the paper
    measured that 0.5 costs 3.1 p.p. of mAR.

    Returns (kept list, info dict).
    """
    if not masks:
        return [], {"n_in": 0, "n_kept": 0, "n_suppressed": 0, "n_over_cap": 0}

    flat = np.stack([m.ravel() for m in masks]).astype(np.float32)
    areas = flat.sum(axis=1).astype(np.float64)
    order = np.argsort(-areas)

    # PRE-ALLOCATED BUFFER, không np.stack trong vòng lặp.
    #
    # Bản đầu dựng `np.stack(kept_flat)` ở MỖI bước, tức cấp phát lại cả mảng
    # (n_kept, H*W) float32 mỗi lần. Đo thật: 300 mask 480x640 mất 88 giây, và
    # NMS là O(n^2) nên 3000 mask (6 mức x ~500 cụm) sẽ là ~2,4 GIỜ MỖI ẢNH —
    # không chạy nổi. Ghi lại vì con số này không đoán ra được, phải đo.
    #
    # Ở đây buffer cấp phát MỘT lần và chỉ ghi thêm hàng; matmul chạy trên
    # `view` của phần đã dùng. Bộ nhớ: max_masks x H*W float32 = 1,2 GB ở
    # 1000 x 480x640 -- nên buffer chỉ lớn tới số mask THẬT SỰ giữ, không tới
    # max_masks.
    #
    # BOUNDING-BOX PREFILTER. Hai mask có hộp bao rời nhau thì IoU = 0, nên
    # không cần matmul nào cả. Đo trên mask ngẫu nhiên (trường hợp xấu nhất,
    # không mask nào bị chặn): 1000 mask mất 51 s chỉ vì matmul lặp lại
    # (n_kept, H*W) @ (H*W,). Hộp bao đổi phép đó lấy 4 phép so sánh số nguyên.
    # Trên dữ liệu thật phần lớn cặp mask rời nhau, nên đây là lọc chính.
    boxes = np.zeros((n_in := len(masks), 4), dtype=np.int64)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        boxes[i] = (xs.min(), ys.min(), xs.max(), ys.max()) if len(ys) else (0, 0, -1, -1)

    cap = min(max_masks, n_in)
    buf = np.empty((cap, flat.shape[1]), dtype=np.float32)
    kept_idx = []
    kept_boxes = np.zeros((cap, 4), dtype=np.int64)
    n_cap = 0
    for i in order:
        n_kept = len(kept_idx)
        if n_kept >= max_masks:
            n_cap += 1
            continue
        if n_kept:
            b = boxes[i]
            kb = kept_boxes[:n_kept]
            # Hộp bao GIAO nhau: điều kiện cần để IoU > 0.
            cand = np.flatnonzero((kb[:, 0] <= b[2]) & (kb[:, 2] >= b[0]) &
                                  (kb[:, 1] <= b[3]) & (kb[:, 3] >= b[1]))
            if len(cand):
                inter = buf[cand] @ flat[i]
                union = areas[np.asarray(kept_idx)[cand]] + areas[i] - inter
                with np.errstate(divide="ignore", invalid="ignore"):
                    iou = np.where(union > 0, inter / union, 0.0)
                if iou.max() > iou_thr:
                    continue
        buf[n_kept] = flat[i]
        kept_boxes[n_kept] = boxes[i]
        kept_idx.append(int(i))

    info = {"n_in": n_in, "n_kept": len(kept_idx),
            "n_suppressed": n_in - len(kept_idx) - n_cap,
            "n_over_cap": n_cap}
    return [masks[i] for i in kept_idx], info


def merge_maps_to_masks(f, cfg, H, W, valid_w=1.0, valid_h=1.0):
    """The whole of Algorithm 2. (K, N) soft maps -> list of (H, W) bool masks.

    Returns (masks, info) where info carries the per-level counts needed to see
    whether the granularity levels are doing anything at all.
    """
    p, ok = normalise_maps(f)
    p = p[ok]
    if len(p) == 0:
        return [], {"n_prompts_used": 0, "n_levels": 0, "per_level": [],
                    "nms": {"n_in": 0, "n_kept": 0, "n_suppressed": 0,
                            "n_over_cap": 0}}

    d = symmetric_kl_matrix(p, eps=cfg.kl_eps)
    heights = np.geomspace(cfg.kl_h_min, cfg.kl_h_max, cfg.n_levels)
    all_labels = cluster_at_heights(d, heights)

    pool, per_level = [], []
    for h, labels in zip(heights, all_labels):
        masks = masks_from_clusters(
            p, labels, cfg.grid_r, H, W, min_area_px=cfg.min_area_px,
            valid_w=valid_w, valid_h=valid_h, connectivity=cfg.connectivity)
        per_level.append({"h": float(h),
                          "n_clusters": int(labels.max()) + 1 if len(labels) else 0,
                          "n_masks": len(masks)})
        pool.extend(masks)

    kept, nms_info = area_descending_nms(pool, iou_thr=cfg.nms_iou,
                                         max_masks=cfg.max_masks)
    return kept, {"n_prompts_used": int(ok.sum()), "n_levels": cfg.n_levels,
                  "heights": [float(h) for h in heights],
                  "per_level": per_level, "nms": nms_info}
