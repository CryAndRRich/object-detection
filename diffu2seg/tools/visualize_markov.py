"""Minh hoạ chuỗi Markov của M2N2 lan truyền từ MỘT prompt point.

⚠️ ĐÂY LÀ M2N2, KHÔNG PHẢI Diffuse2Seg. Diffuse2Seg thay toàn bộ cơ chế này
bằng p-Laplacian. Tool này chỉ để ĐỌC HIỂU cơ chế, không sinh ra số nào đi vào
bảng kết quả.

Vẽ một hàng ảnh:
  1. ảnh gốc + prompt point
  2. hàng attention THÔ của ô prompt (A[seed]) — tức "một bước, chưa lan"
  3..N. p_t sau từng bước chuỗi Markov (đã chia max)
  cuối. Markov-map m[k] = thời gian đến ngưỡng (THẤP = gần về ngữ nghĩa)

VÍ DỤ
  python tools/visualize_markov.py --dataset paco --image-id 5142 \
      --point 0.42 0.45 --canvas 512 \
      --out /mnt/disk1/aiotlab/haitn/output/markov_5142.png

--point là toạ độ TƯƠNG ĐỐI (x, y) trong [0,1] trên ẢNH GỐC, để không phải
tra pixel. Mặc định (0.5, 0.5).
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base import Diffu2SegConfig                  # noqa: E402
from d2s.affinity import change_temperature              # noqa: E402
from d2s.markov import matrix_ipf, markov_map_from_prompt  # noqa: E402
from d2s.pipeline import build_affinity                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_best import build_dataset                 # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="paco", choices=["paco", "coco", "ce130"])
    ap.add_argument("--split", default="val")
    ap.add_argument("--image-id", type=int, default=None,
                    help="id ảnh trong json; bỏ trống thì lấy --index")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--point", type=float, nargs=2, default=[0.5, 0.5],
                    metavar=("X", "Y"), help="toạ độ tương đối trong [0,1]")
    ap.add_argument("--canvas", type=int, default=512,
                    help="512 nhẹ và đủ để nhìn cơ chế; 1120 là của paper")
    ap.add_argument("--steps", type=int, nargs="+",
                    default=[1, 2, 3, 5, 10, 25, 50],
                    help="các bước chuỗi Markov muốn vẽ")
    ap.add_argument("--temperature", type=float, default=0.65, help="M2N2: 0.65")
    ap.add_argument("--tau", type=float, default=0.3, help="M2N2: 0.3")
    ap.add_argument("--ipf-iters", type=int, default=200,
                    help="M2N2 truyền 200 ở cả hai call-site (mặc định 15 KHÔNG đủ)")
    ap.add_argument("--no-ipf", action="store_true",
                    help="[ĐỐI CHỨNG] bỏ IPF để thấy vì sao paper cần nó")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--underlay", type=float, default=0.0,
                    help="độ đậm ảnh gốc dưới Markov-map. MẶC ĐỊNH 0 = tắt, "
                         "giống paper. Bật (vd 0.3) thì dễ đối chiếu vùng "
                         "trắng với vật, nhưng làm giảm tương phản của map.")
    ap.add_argument("--show-attention", action="store_true",
                    help="thêm panel attention THÔ A[seed] để so với Markov-map "
                         "(đây là đại lượng KHÁC: xác suất, cao = gần)")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Diffu2SegConfig(canvas=args.canvas).validate()
    root = args.data_root or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "data")
    ds = build_dataset(args.dataset, root, cfg.canvas, split=args.split)

    idx = args.index
    if args.image_id is not None:
        hits = [i for i in range(len(ds)) if ds.images[i]["id"] == args.image_id]
        if not hits:
            raise SystemExit(f"không thấy image_id={args.image_id} trong {args.dataset}")
        idx = hits[0]
    s = ds[idx]

    r = cfg.grid_r
    vh = int(s["valid_h"] * cfg.canvas)
    vw = int(s["valid_w"] * cfg.canvas)

    # prompt: toạ độ tương đối -> ô lưới. Dùng vùng HỢP LỆ (bỏ pad).
    px, py = args.point
    cx = min(int(px * vw / cfg.canvas * r), r - 1)
    cy = min(int(py * vh / cfg.canvas * r), r - 1)
    seed = cy * r + cx

    print(f"ảnh {s['file_name']} ({s['W']}x{s['H']})  canvas={cfg.canvas} r={r}")
    print(f"prompt tương đối ({px:.2f}, {py:.2f}) -> ô lưới ({cx}, {cy}) = index {seed}")

    print("chạy SD, trích self-attention ...", flush=True)
    from d2s.attention import StableDiffusion2AttentionAggregator
    agg = StableDiffusion2AttentionAggregator(
        timestep=cfg.timesteps[0], attention_resolution=r,
        weight_down_block_0=cfg.w_down_0, weight_down_block_1=cfg.w_down_1,
        weight_up_block_0=cfg.w_up_0, weight_up_block_1=cfg.w_up_1,
        weight_up_block_2=cfg.w_up_2, hugging_face_model_id=cfg.model_source,
        prompt_text=cfg.prompt_text, device=args.device, torch_dtype=torch.float16)
    A = build_affinity(s["image"], cfg, agg)          # (N, N) row-stochastic

    raw_row = A[seed].detach().clone()                # hàng attention THÔ

    # --- dựng transition matrix theo M2N2: nhiệt độ -> IPF ---
    A = change_temperature(A, args.temperature, dim=-1)
    if not args.no_ipf:
        print(f"IPF {args.ipf_iters} vòng ...", flush=True)
        A = matrix_ipf(A, iterations=args.ipf_iters)
    rs, cs = A.sum(1), A.sum(0)
    print(f"  hàng tổng [{rs.min():.6f}, {rs.max():.6f}]  "
          f"cột tổng [{cs.min():.6f}, {cs.max():.6f}]"
          + ("   <-- chưa doubly stochastic (--no-ipf)" if args.no_ipf else ""))

    print("lan truyền chuỗi Markov ...", flush=True)
    m, snaps = markov_map_from_prompt(
        A, seed, tau=args.tau, max_iterations=args.max_iter,
        snapshot_steps=args.steps)

    n_reached = int((m < args.max_iter).sum())
    print(f"  {n_reached}/{len(m)} ô vượt tau={args.tau} trong {args.max_iter} bước")
    print(f"  m: min {m.min():.2f}, trung vị {m.median():.2f}, max {m.max():.2f}")

    # ---------------- vẽ ----------------
    bg = np.asarray(Image.fromarray(s["image"][:vh, :vw])
                    .resize((s["W"], s["H"]), Image.BILINEAR))
    # ⚠️ Ô lưới chỉ trải trên VÙNG HỢP LỆ, không phải cả r ô: ảnh dọc
    # 478x640 chiếm 382/512 canvas = 47,8 ô ngang, 16,2 ô còn lại là PAD.
    # Chia cho r sẽ đặt chấm lệch trái tới 26 % (đo: px=1.0 ra pixel 355 thay
    # vì 478). Chia cho SỐ Ô HỢP LỆ mới đúng. Ảnh ngang có valid_w=1 nên hai
    # cách trùng nhau — đó là lý do lỗi không lộ trên ảnh con mèo.
    gx = (cx + 0.5) / max(vw / cfg.canvas * r, 1e-9) * s["W"]
    gy = (cy + 0.5) / max(vh / cfg.canvas * r, 1e-9) * s["H"]

    # Số ô thực sự phủ ảnh; phần còn lại của lưới nằm trên PAD.
    nx = max(int(round(vw / cfg.canvas * r)), 1)
    ny = max(int(round(vh / cfg.canvas * r)), 1)

    def to_img(vec):
        """(N,) trên lưới r x r -> (H, W) ở kích thước ảnh gốc.

        ⚠️ CẮT BỎ PAD trước khi resize. Ảnh dọc 478x640 chỉ chiếm 48/64 cột
        lưới; kéo cả 64 cột ra chiều rộng ảnh sẽ NÉN map lại và lệch khỏi vật.
        Ảnh ngang có valid_w=1 nên không lộ — đó là lý do lỗi sống sót qua ảnh
        con mèo.
        """
        a = vec.detach().float().cpu().numpy().reshape(r, r)[:ny, :nx]
        return np.asarray(Image.fromarray(a).resize((s["W"], s["H"]), Image.BILINEAR))

    # MÀU: theo đúng M2N2. Hình 3 của paper ghi rõ "Markov-maps are INVERTED
    # such that the LOWEST value is WHITE and the highest value is black", và
    # code của họ vẽ `imshow(-markov_map, cmap='gray')`. Markov-map là THỜI
    # GIAN ĐẾN, thấp = gần về ngữ nghĩa, nên đảo dấu làm vùng cùng vật thành
    # TRẮNG. Ta theo y hệt để hình đọc được cạnh hình trong paper.
    # ⚠️ THANG MÀU. `m` có dải [0, max_iter]: ô ĐÃ tới nằm quanh 0..50, ô CHƯA
    # tới mang đúng max_iter (200). Để imshow tự scale theo min/max thì ~75 %
    # dải màu bị phí cho khoảng trống giữa hai nhóm, và mọi sắc độ trong vùng
    # đang nhìn bị nén vào một góc -> nền đen kịt, mất hết chi tiết
    # (đo 2026-09-18 trên chính ảnh con mèo).
    #
    # Paper vẽ `m` ở trạng thái HỘI TỤ, nơi gần như mọi ô đều đã tới nên không
    # gặp vấn đề này. Hình theo bước thì có cả hai loại trong một ảnh.
    #
    # Sửa: chốt vmin/vmax theo PHÂN VỊ của vùng ĐÃ TỚI, và dùng CHUNG cho mọi
    # panel để các bước so được với nhau. Ô chưa tới vượt vmin sẽ bị kẹp thành
    # đen — đúng ý nghĩa "chưa tới".
    reached = m[m < args.max_iter]
    if len(reached) > 8:
        lo = float(torch.quantile(reached.float(), 0.02))
        hi = float(torch.quantile(reached.float(), 0.98))
    else:                       # gần như không ô nào tới: quay về dải thô
        lo, hi = 0.0, float(args.max_iter)
    if hi - lo < 1e-6:
        hi = lo + 1.0
    vmin, vmax = -hi, -lo       # vì ta vẽ -m
    print(f"  thang màu: m in [{lo:.2f}, {hi:.2f}] (phân vị 2-98 % của vùng đã tới)")

    steps = [t for t in args.steps if t in snaps]
    panels = [("Input image + prompt", bg, None, None)]
    if args.show_attention:
        # Attention thô KHÔNG phải Markov-map: nó là xác suất, cao = gần. Không
        # đảo dấu, và để cmap khác để không bị đọc nhầm là cùng một đại lượng.
        panels.append(("Raw attention A[seed]", to_img(raw_row), "magma", None))
    for t in steps:
        _, m_t = snaps[t]
        panels.append((f"Markov-map @ t={t}", -to_img(m_t), "gray", (t, m_t)))
    panels.append((f"Markov-map final (tau={args.tau})", -to_img(m), "gray",
                   (args.max_iter, m)))

    n = len(panels)
    n_top = (n + 1) // 2
    n_bot = n - n_top
    ncols = 2 * n_top
    aspect = s["H"] / max(s["W"], 1)
    nrows = 1 if n_bot == 0 else 2
    fig = plt.figure(figsize=(4.2 * n_top, 4.2 * aspect * nrows + 1.0))
    gs = fig.add_gridspec(nrows, ncols)

    for i, (title, img, cmap, _extra) in enumerate(panels):
        if i < n_top:
            row, c0 = 0, i * 2
        else:
            row = 1
            c0 = (ncols - 2 * n_bot) // 2 + (i - n_top) * 2
        ax = fig.add_subplot(gs[row, c0:c0 + 2])
        if cmap is None:
            ax.imshow(img)
        elif cmap == "gray":
            # Ảnh gốc mờ bên dưới để đối chiếu vùng trắng với vật thật; không
            # có nó thì không biết mask có bám đúng con mèo hay không.
            if args.underlay > 0:
                ax.imshow(bg, alpha=args.underlay)
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax,
                          alpha=1.0 - args.underlay * 0.45)
            else:
                ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            ax.imshow(img, cmap=cmap)
        ax.plot([gx], [gy], marker="o", ms=7, mfc="none", mec="lime", mew=2)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"M2N2 Markov propagation — {s['file_name']}  |  "
        f"T={args.temperature} tau={args.tau} "
        f"{'NO IPF' if args.no_ipf else f'IPF {args.ipf_iters}'}  |  "
        f"canvas={cfg.canvas} r={r} t={cfg.timesteps[0]}", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
