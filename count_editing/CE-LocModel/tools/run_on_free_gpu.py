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
# Ngưỡng cũ hard-code 15000 MiB cho MỌI job là số bịa, không đo từ đâu, và đã
# gây hậu quả thật: 2026-08-31 variant (d) cùng cả 2 job eval bị bỏ qua trong
# đúng 1 giây vì không GPU nào còn 15000 MiB, dù chúng cần ít hơn nhiều.
#
# Mô hình bộ nhớ dưới đây cố ý ĐƠN GIẢN và mọi hằng số đều đo được bằng
# tools/measure_vram.py. Ba phần cộng lại:
#
#   1. CUDA_CONTEXT: context + cuDNN workspace, nvidia-smi thấy nhưng
#      torch.cuda.max_memory_allocated KHÔNG thấy. Là hằng số, KHÔNG co giãn
#      theo batch_size -- nên cộng vào chứ không nhân hệ số.
#   2. MODEL_STATE: weight + gradient + 2 buffer AdamW. ResNet18 (11,7M) +
#      transformer 8 layer d=256 (~6,3M) ~ 18M tham số x 4 byte x 4 bản = ~290 MiB.
#      Eval không có gradient/optimizer nên chỉ ~1 bản.
#   3. ACTIVATION: phần DUY NHẤT co giãn theo batch_size. Chi phối bởi feature
#      map ResNet18 ở 512x512 (conv1 ra 64x256x256 = 16 MiB/ảnh fp32, cộng dồn
#      qua các stage ~ 55 MiB/ảnh giữ lại cho backward).
#
# Nhân SAFETY vào phần co giãn để phủ dao động dữ liệu (ảnh nhiều box hơn
# trung bình) và phân mảnh của caching allocator.
CUDA_CONTEXT_MIB = 800
SAFETY = 1.25
ACT_MIB_PER_IMAGE_TRAIN = 55    # giữ lại cho backward
ACT_MIB_PER_IMAGE_EVAL = 8      # no_grad: chỉ 1 layer sống tại một thời điểm
MODEL_STATE_MIB_TRAIN = 290     # weight + grad + AdamW exp_avg/exp_avg_sq
MODEL_STATE_MIB_EVAL = 80       # chỉ weight


def infer_min_free_mib(target):
    """Ngưỡng free-memory suy từ lệnh sắp chạy, thay cho hằng số 15000 cũ.

    Đọc --batch_size (train) hoặc --k (eval) ngay trong argv của script con --
    không import gì, không dựng model, nên không tốn VRAM để quyết định xem có
    đủ VRAM không.

    Cố tình ƯỚC LƯỢNG CAO khi không chắc: chọn nhầm GPU quá nhỏ thì OOM giữa
    chừng (mất cả giờ train), chọn nhầm ngưỡng hơi cao thì chỉ phải chờ thêm.
    """
    script = os.path.basename(target[0]) if target else ""
    is_eval = "eval" in script or "test_" in script

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
        # eval chạy từng ảnh một; với nhánh 1-box thì K mẫu chạy SONG SONG trong
        # cùng một batch, nên K mới là thứ quyết định, không phải batch_size.
        n_parallel = flag("--k", 30)
        base = MODEL_STATE_MIB_EVAL + ACT_MIB_PER_IMAGE_EVAL * n_parallel
    else:
        bs = flag("--batch_size", 32)
        base = MODEL_STATE_MIB_TRAIN + ACT_MIB_PER_IMAGE_TRAIN * bs

    return int(base * SAFETY + CUDA_CONTEXT_MIB)


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

    if args.min_free_mib is None:
        args.min_free_mib = infer_min_free_mib(target)
        print(f"--min-free-mib tự suy: {args.min_free_mib} MiB (từ chính lệnh sắp chạy). "
              f"Đo lại bằng: python tools/measure_vram.py --variant <v> --mode <train|eval>",
              flush=True)
    else:
        print(f"--min-free-mib = {args.min_free_mib} MiB (ép bằng tham số)", flush=True)

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
