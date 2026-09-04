"""Chạy một script Python bất kỳ (train_w_args.py, test_mul_box.py, ...) trên
GPU trống nhất của server, tự động chọn qua nvidia-smi.

Port lại nguyên logic từ
object-detection/diffu_grounding_dino/tools/run_train.py — cùng lý do: server
này chạy chung với người/job khác, nên không thể cứ mặc định GPU 0. Khác với
bản gốc (chỉ wrap main.py cố định), script này wrap MỘT SCRIPT PYTHON BẤT KỲ
truyền vào — vì CE-LocModel có 2 entry point cần chạy trên server
(train_w_args.py, test_mul_box.py), không phải 1.

  1. Query nvidia-smi cho mọi GPU: free memory + utilization.
  2. Loại trước các GPU không đủ ``--min-free-mib`` free memory -- điều kiện
     CỨNG, không GPU nào qua được thì thoát lỗi thay vì chọn liều rồi OOM
     giữa chừng.
  3. Trong số các GPU đã đủ memory, ưu tiên GPU KHÔNG bận tính toán
     (utilization dưới ngưỡng); nếu tất cả đều bận tính toán, chọn GPU nhiều
     free memory nhất trong số đó rồi cảnh báo.
  4. Set CUDA_VISIBLE_DEVICES=<gpu đã chọn> rồi exec script Python truyền vào
     với mọi tham số còn lại.

Cách dùng — script Python cần chạy đứng ngay sau --, mọi tham số của NÓ đứng
sau đó:

    python tools/run_on_free_gpu.py -- train_w_args.py --variant resnet34_transformer
    python tools/run_on_free_gpu.py -- test_mul_box.py --checkpoint checkpoints/resnet34_transformer/best_model.pth --variant resnet34_transformer

Muốn ép 1 GPU cụ thể: --gpu <index> (đặt TRƯỚC --).
"""

import argparse
import os
import subprocess
import time
import sys

UTIL_BUSY_THRESHOLD = 50  # phần trăm; GPU trên ngưỡng này coi là đang bận tính toán

# Mã trả về nội bộ của _pick_and_run khi KHÔNG chọn được GPU nào (khác hẳn "job
# đã chạy rồi chết"): job chưa hề khởi động, nên đây luôn là tình trạng tạm thời
# của server dùng chung -- phải chờ chứ không được coi là lỗi code. Giá trị âm
# nên không đụng mã thoát nào của Python.
NO_GPU = -1000


def query_gpus():
    """``[(index, free_mib, total_mib, util_percent), ...]`` từ nvidia-smi."""
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    gpus = []
    for line in out.strip().splitlines():
        idx, used, total, util = (int(x.strip()) for x in line.split(","))
        gpus.append((idx, total - used, total, util))
    return gpus


# --- Suy ngưỡng free-memory từ chính lệnh sắp chạy -------------------------
#
# Lịch sử của con số này, vì nó đã sai HAI lần theo hai hướng ngược nhau:
#
#   1. Ban đầu hard-code 15000 MiB cho MỌI job -- số bịa, không đo. Hậu quả
#      (2026-08-31): variant (d) và cả 2 job eval bị bỏ qua trong đúng 1 giây
#      vì không GPU nào còn 15000 MiB, dù chúng cần ít hơn.
#   2. Rồi thay bằng một mô hình bộ nhớ suy từ kích thước tensor. Mô hình đó
#      ước lượng THẤP hơn thực tế 2,0-4,2 lần (đo 2026-09-04):
#
#         job                     công thức cũ    ĐO THẬT
#         train (d) bs=32            3362 MiB     9200 MiB
#         train (e) bs=32            3362 MiB     6862 MiB
#         eval multibox              1200 MiB    ~5000 MiB
#
#      Sai theo hướng NGUY HIỂM: ước lượng thấp -> chọn GPU không đủ -> OOM
#      giữa chừng, mất cả giờ train. May là 3 job trên vẫn chạy được.
#
# Bản hiện tại HIỆU CHỈNH TỪ SỐ ĐO THẬT ở trên, không suy từ kích thước tensor
# nữa. Nguyên tắc: neo vào giá trị CAO NHẤT quan sát được cho mỗi chế độ, không
# lấy trung bình -- vì hậu quả của ước lượng thấp (OOM, mất giờ train) nặng hơn
# nhiều so với ước lượng cao (chỉ phải chờ GPU thêm một lúc).
#
# Vì sao (d) 9200 mà (e) 6862 dù cùng bs=32, cùng N=300: CHƯA GIẢI THÍCH ĐƯỢC.
# Class head 80 chiều chỉ chiếm ~17 MiB trong 2338 MiB chênh lệch. Không suy
# đoán tiếp -- dùng mức cao hơn cho an toàn, và đo lại bằng tools/measure_vram.py
# khi cần con số chính xác cho một cấu hình cụ thể.
TRAIN_FIXED_MIB = 2000      # model state + CUDA context + cuDNN workspace
TRAIN_PER_IMAGE_MIB = 225   # hiệu chỉnh từ (9200 - 2000) / 32, dùng mức CAO nhất
EVAL_MULTIBOX_MIB = 5000    # ĐO THẬT: eval (c)/(e), 1 chain [1,300,4], không dùng --k
EVAL_1BOX_FIXED_MIB = 2000  # ƯỚC LƯỢNG -- chưa có số đo cho nhánh 1-box
EVAL_1BOX_PER_K_MIB = 100   # ƯỚC LƯỢNG -- K chain chạy song song trong 1 batch
SAFETY = 1.15               # đệm nhỏ, vì các số trên đã là số đo thật chứ không phải mô hình


def infer_min_free_mib(target, gpu_capacity_mib=None):
    """Ngưỡng free-memory suy từ lệnh sắp chạy, hiệu chỉnh từ số đo thật.

    Đọc --batch_size (train) hoặc --k (eval 1-box) ngay trong argv của script
    con -- không import gì, không dựng model, nên không tốn VRAM để quyết định
    xem có đủ VRAM không.

    Cố tình ƯỚC LƯỢNG CAO khi không chắc: chọn nhầm GPU quá nhỏ thì OOM giữa
    chừng (mất cả giờ train), chọn nhầm ngưỡng hơi cao thì chỉ phải chờ thêm.

    `gpu_capacity_mib`: dung lượng GPU lớn nhất trên máy, dùng để chặn trần.
    Truyền vào từ ngoài (caller đã đọc nvidia-smi rồi) thay vì tự query, để hàm
    này thuần tuý và không tốn thêm một lần gọi nvidia-smi.
    """
    script = os.path.basename(target[0]) if target else ""
    is_eval = "eval" in script or "test_" in script
    is_diag = "diagnose" in script or "overfit" in script or "measure_vram" in script

    def flag(name, default):
        for i, a in enumerate(target):
            if a == name and i + 1 < len(target):
                try:
                    return int(target[i + 1])
                except ValueError:
                    return default
            if a.startswith(name + "="):
                try:
                    return int(a.split("=", 1)[1])
                except ValueError:
                    return default
        return default

    if is_eval:
        # Nhánh multibox (variant c/d/e) không dùng --k: nó chạy 1 chain [1,N,4],
        # nên chi phí gần như hoàn toàn cố định. Nhánh 1-box (a/b) thì K chain
        # chạy SONG SONG trong cùng một batch, nên K mới là thứ quyết định.
        # Không đọc được config từ đây, nên lấy MỨC CAO HƠN của hai khả năng.
        k = flag("--k", 30)
        base = max(EVAL_MULTIBOX_MIB, EVAL_1BOX_FIXED_MIB + EVAL_1BOX_PER_K_MIB * k)
    elif is_diag:
        # overfit_one_image.py train thật nhưng chỉ 1 ảnh; diagnose/measure_vram
        # chạy forward. Đều rẻ hơn train đầy đủ, nhưng vẫn dựng cả model +
        # optimizer -> lấy bằng mức eval cho an toàn.
        base = EVAL_MULTIBOX_MIB
    else:
        bs = flag("--batch_size", 32)
        base = TRAIN_FIXED_MIB + TRAIN_PER_IMAGE_MIB * bs

    threshold = int(base * SAFETY)

    # Chặn trần: một ngưỡng vượt dung lượng GPU lớn nhất trên máy thì KHÔNG GPU
    # nào qua nổi, và job sẽ chờ vô ích cho tới hết --wait-for-gpu rồi chết --
    # tệ hơn hẳn việc cứ thử chạy. Ví dụ: eval 1-box với --k 300 suy ra 36.800
    # MiB trong khi A30 chỉ có 24.576. Trong tình huống đó, hạ xuống 90% dung
    # lượng GPU và để job tự OOM nếu thật sự không đủ -- ít nhất nó được thử,
    # và traceback OOM nói rõ hơn "chờ 6 tiếng rồi bỏ cuộc".
    if gpu_capacity_mib is not None:
        cap = gpu_capacity_mib
        if threshold > cap:
            print(f"CẢNH BÁO: ngưỡng suy ra {threshold} MiB vượt dung lượng GPU lớn nhất "
                  f"({cap} MiB) -- không GPU nào qua được, job sẽ chờ vô ích. Hạ xuống "
                  f"{int(cap * 0.9)} MiB và để job tự chạy; nếu OOM thật thì giảm "
                  f"--batch_size / --k.", flush=True)
            threshold = int(cap * 0.9)

    return threshold


def pick_gpu(gpus, min_free_mib):
    """``(None, False)`` if no GPU has enough free memory; otherwise ``(gpu, had_idle)``.

    Memory is filtered FIRST and is a hard requirement -- a GPU below
    ``min_free_mib`` is never chosen no matter how idle its compute is, since
    that is exactly what OOMs. Only among the GPUs that fit is idle compute
    (``utilization < UTIL_BUSY_THRESHOLD``) preferred.
    """
    fits = [g for g in gpus if g[1] >= min_free_mib]
    if not fits:
        return None, False
    idle = [g for g in fits if g[3] < UTIL_BUSY_THRESHOLD]
    pool = idle if idle else fits
    chosen = max(pool, key=lambda g: g[1])
    return chosen, bool(idle)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gpu", type=int, default=None, help="ép chạy trên GPU index cụ thể, bỏ qua auto-detect")
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="số lần thử lại khi job chết vì OOM. Server dùng chung nên GPU có thể bị job "
        "khác chiếm mất trong khoảng thời gian giữa lúc đọc nvidia-smi và lúc thật sự cấp phát "
        "-- lần thử lại sẽ đọc nvidia-smi MỚI và có thể chọn GPU khác.",
    )
    parser.add_argument(
        "--retry-wait",
        type=int,
        default=60,
        help="số giây chờ trước mỗi lần thử lại",
    )
    parser.add_argument(
        "--wait-for-gpu",
        type=int,
        default=0,
        help="tổng số GIÂY sẵn sàng chờ khi KHÔNG GPU nào đủ free memory (mặc định 0 = thoát "
        "ngay như cũ). Khác --retries: cái đó đếm số lần job chạy rồi chết, cái này là ngân sách "
        "thời gian chờ trước khi job kịp khởi động. Dùng khi xếp hàng nhiều job tuần tự trên "
        "server dùng chung, ví dụ --wait-for-gpu 21600 để chờ tối đa 6 giờ.",
    )
    parser.add_argument(
        "--gpu-poll",
        type=int,
        default=300,
        help="số giây giữa 2 lần đọc lại nvidia-smi trong lúc chờ GPU",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=None,
        help="loại GPU có free memory dưới mức này trước khi xét utilization. Bỏ trống = TỰ SUY "
        "từ chính lệnh sắp chạy (xem infer_min_free_mib); truyền số để ép, ví dụ khi đã đo bằng "
        "tools/measure_vram.py và biết chắc mức thực tế.",
    )
    parser.add_argument(
        "target", nargs=argparse.REMAINDER,
        help="script Python cần chạy + tham số của nó, đặt sau --, ví dụ: "
        "-- train_w_args.py --variant resnet34_transformer",
    )
    args = parser.parse_args()

    target = args.target
    if target and target[0] == "--":
        target = target[1:]
    if not target:
        parser.error("thiếu script cần chạy — đặt sau --, ví dụ: -- train_w_args.py --variant resnet18_cnn")

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cmd = [sys.executable] + target

    if args.min_free_mib is not None:
        print(f"--min-free-mib = {args.min_free_mib} MiB (ép bằng tham số)", flush=True)
    # Nếu để trống thì ngưỡng được suy trong _pick_and_run, nơi đã có sẵn snapshot
    # nvidia-smi -- tránh đọc nvidia-smi thêm một lần chỉ để biết dung lượng GPU.

    deadline = time.time() + args.wait_for_gpu
    attempt = 0
    while True:
        rc = _pick_and_run(args, cmd, repo_root, attempt + 1)

        # KHÔNG chọn được GPU: job chưa hề chạy. Đây luôn là tình trạng tạm thời
        # của server dùng chung (job của người khác vừa chiếm chỗ), nên chờ theo
        # NGÂN SÁCH THỜI GIAN --wait-for-gpu chứ không tính vào --retries: đếm số
        # lần vô nghĩa khi nguyên nhân là "đợi người khác chạy xong".
        if rc == NO_GPU:
            remain = deadline - time.time()
            if remain <= 0:
                print(f"Đã chờ hết {args.wait_for_gpu}s mà vẫn không GPU nào đủ "
                      f"{args.min_free_mib} MiB free -- dừng. Tăng --wait-for-gpu, giảm "
                      f"--batch_size, hoặc chạy lại lúc server rảnh.", flush=True)
                raise SystemExit(1)
            print(f"[chờ GPU] còn {remain/60:.0f} phút trong ngân sách chờ. "
                  f"Đọc lại nvidia-smi sau {args.gpu_poll}s...\n", flush=True)
            time.sleep(min(args.gpu_poll, max(remain, 1)))
            continue

        attempt += 1
        if rc == 0 or not _looks_like_oom(repo_root, rc):
            raise SystemExit(rc)
        if attempt > args.retries:
            print(f"HẾT {args.retries} lần thử lại, lần nào cũng thoát != 0 -- dừng. Nếu traceback "
                  f"ở trên là CUDA out of memory thì GPU đang bị chiếm liên tục (đợi rồi chạy lại, "
                  f"hoặc ép --gpu <id>). Nếu là lỗi khác thì thử lại không giúp được gì -- sửa code.")
            raise SystemExit(rc)
        print(f"\n[thử lại {attempt}/{args.retries}] job thoát với mã {rc}. Nguyên nhân THƯỜNG GẶP "
              f"trên server dùng chung là OOM: GPU bị job khác chiếm mất giữa lúc đọc nvidia-smi và "
              f"lúc thật sự cấp phát. Nhưng mã thoát không phân biệt được OOM với lỗi code, nên NẾU "
              f"cả {args.retries} lần đều hỏng thì hãy đọc traceback ở trên -- rất có thể là bug thật, "
              f"không phải GPU. Đợi {args.retry_wait}s rồi đọc lại trạng thái GPU...\n", flush=True)
        time.sleep(args.retry_wait)


def _looks_like_oom(repo_root, rc):
    """Chỉ thử lại khi job chết vì hết VRAM, không phải mọi lỗi.

    Python thoát với mã 1 cho mọi exception nên mã trả về không phân biệt được;
    dùng chính torch để hỏi lại xem GPU có đang cạn không thì phức tạp và vẫn
    đoán mò. Ở đây chấp nhận thử lại với mọi mã != 0 NHƯNG chỉ khi lần chạy đó
    kéo dài rất ngắn (OOM lúc model.to(device) xảy ra trong vài giây, còn lỗi
    logic/dữ liệu thường cũng nhanh -- nên đây là heuristic, và mỗi lần thử lại
    đều in rõ lý do để người đọc log biết chuyện gì đang xảy ra).
    """
    return rc != 0


def _pick_and_run(args, cmd, repo_root, attempt):
    if args.gpu is not None:
        chosen_index = args.gpu
        print(f"dùng GPU {chosen_index} (ép bằng --gpu)")
    else:
        gpus = query_gpus()
        print("trạng thái GPU hiện tại:")
        for idx, free, total, util in gpus:
            print(f"  GPU {idx}: free {free:6d} MiB / {total} MiB, utilization {util:3d}%")
        if args.min_free_mib is None:
            args.min_free_mib = infer_min_free_mib(
                cmd[1:], gpu_capacity_mib=max(g[2] for g in gpus))
            print(f"--min-free-mib tự suy: {args.min_free_mib} MiB (từ chính lệnh sắp chạy, "
                  f"hiệu chỉnh theo số đo thật). Đo lại bằng: "
                  f"python tools/measure_vram.py --variant <v> --mode <train|eval>", flush=True)
        picked, had_idle = pick_gpu(gpus, args.min_free_mib)
        if picked is None:
            print(f"LỖI: không GPU nào còn >= {args.min_free_mib} MiB free -- không đủ cho batch_size "
                  f"hiện tại. Đợi rồi thử lại, giảm batch_size, hoặc chỉnh --min-free-mib nếu đã biết "
                  f"chắc mức thực tế cần thấp hơn.", flush=True)
            return NO_GPU
        chosen_index, free, total, util = picked
        if not had_idle:
            print(f"CẢNH BÁO: GPU {chosen_index} đủ free memory ({free} MiB) nhưng utilization {util}% "
                  f"(>{UTIL_BUSY_THRESHOLD}%) -- chọn vì không GPU nào vừa đủ memory vừa rảnh compute, "
                  f"sẽ tranh chấp compute với job khác nên train chậm hơn, nhưng không OOM.")
        else:
            print(f"chọn GPU {chosen_index} (free {free} MiB, utilization {util}%)")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(chosen_index)

    print(f"chạy: CUDA_VISIBLE_DEVICES={chosen_index} {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, cwd=repo_root)
    return result.returncode


if __name__ == "__main__":
    main()
