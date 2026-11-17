# object-detection

Code của dự án `multi_condition` (bối cảnh: [`../CLAUDE.md`](../CLAUDE.md)). Branch `delta`: CE-Loc bài ADD từ DELTA4 trở đi + baseline
định vị ZETA; mọi thứ cũ hơn (ALPHA / BETA / GAMMA, DELTA1–3.x, baseline detector, diffuse2seg, user study) ở branch `new`.

| | nội dung |
|---|---|
| [`ce_localization/`](ce_localization/README.md) | CE-Loc (DELTA4+): chọn box chỗ thêm vật bằng khuếch tán trên toạ độ box; `box_review/` = web rà box tay |
| [`baseline/`](baseline/README.md) | ZETA1–3, 1.1: finetune + chấm BootPlace / CSANet / HiddenObjects bằng đúng code chấm CE-Loc; ZETA7 / 7.1: OracleLoc (mốc oracle, không weight) |
| [`vlm/`](vlm/README.md) | ZETA4–6: RePlan + VLM thương mại / open-source xin 30 box / ảnh, chấm bằng `baseline/score.py`; ZETA4.1, 8.x: finetune SFT LoRA (`finetune.py`) |
| `tools/run_on_free_gpu.py` | chạy một script trên GPU có nhiều bộ nhớ trống nhất (thử lại khi OOM), `--reserve-gib` giữ trước bộ nhớ (`tools/gpu_hold.py`; `--keeper` giữ chỗ cho cả chuỗi lệnh) — dùng chung; cách dùng + khuôn chuỗi: [mục "Giữ bộ nhớ GPU"](#giữ-bộ-nhớ-gpu-trên-server-dùng-chung-toolsrun_on_free_gpupy-toolsgpu_holdpy) |
| `tools/latency.py` | latency 1 ảnh — một định nghĩa cho CE-Loc / baseline (cả OracleLoc) / VLM (`celoc` / `zeta` / `vlm`) |
| `tests/` | toàn bộ test: `python -m pytest tests -q` (từ `object-detection/`, test nhanh); `--runslow` thêm trọn luồng train / eval + DDP |
| `notebooks/` | notebook Kaggle (**gitignore**, chỉ ở local): `add_kaggle.ipynb` (CE-Loc), `baseline_kaggle.ipynb` (ZETA1–3), `vlm_kaggle.ipynb` (ZETA6) |
| `data/`, `weights/` | dữ liệu + weight, **không vào git** (zip thủ công lên server) |

Package import theo tên đầy đủ (`ce_localization.models...`); script tự thêm `object-detection/` vào `sys.path` nên vẫn chạy
`python train.py` trong thư mục project.

## `weights/` và `data/`

```
weights/add/<run>/          last.pth, best.pth, history.json, snapshots/epNNN.pth, pretrained/ của từng run (delta8, delta8_1, zeta1 … zeta3, zeta4_1, zeta8_x)
data/all_phase2_V2/         CE-130 (nhánh xoá vật cộng dồn, annotation)
data/samples/               ảnh inpaint + density + annotation của bài (train / test)
data/turn_index.json        chỉ mục (nhánh, lượt) <-> samples (tools/build_turn_index.py)
data/cocount/               CE-CoCount (Image/ Anno/ Anno_with_exam_bbox/)
data/cocount_review.jsonl   bản rà box vật CE-CoCount (`box_review/app.py cocount`) — đầu vào eval `review5`
data/ce130_train.jsonl      box positive gán tay của `samples/train` — đích train DELTA7+
```

Kết quả eval (ngoài repo, `../output/add/`), nhóm theo package: `celoc/<run>/<tập>/` và `baseline/<run>/<tập>/` (`epNNN.json` = snapshot,
`best.json` = best.pth, kèm `*_boxes.json`; tập = `ce130_test_all` | `cocount_review5` | …), `vlm/<model>/<tập>/`; việc chung: hình ở
`viz/`, latency ở `latency/`, kết quả rà box model ở `review/`. Log `../log/add/{celoc,baseline}/<run>/`, `../log/add/vlm/<model>/`.

## Môi trường

Một `requirements.txt`. Server: env `ce-locmodel` (`/mnt/disk1/aiotlab/envs/ce-locmodel/bin/python`), Python 3.10, torch 2.3.0+cu121,
3× A30 24 GB dùng chung; `huggingface_hub` phải `< 1.0`. Server 2 (RTX PRO 5000 48 GB): `/mnt/disk1/aiotlab/haitn/envs/{ce-locmodel,vllm}`
(vLLM: `vlm/README.md`). Local: `ce_localization/.venv-cpu` (torch 2.2 CPU) cho test; `box_review/`
cần thêm `gradio`.

CE-Loc: bài NeurIPS 2026 của chính người dùng; repo gốc + bản cập nhật của tác giả ở `../refs/` (read-only).

## Giữ bộ nhớ GPU trên server dùng chung (`tools/run_on_free_gpu.py`, `tools/gpu_hold.py`)

Mục này viết cho cả người chạy lệnh lẫn Claude khi đưa lệnh. Chi tiết cài đặt nằm trong docstring đầu [`tools/gpu_hold.py`](tools/gpu_hold.py).

### Vấn đề

Server 1 (3× A30) và server 2 (1× RTX PRO 5000 48 GB) đều dùng chung với người khác. Có hai kiểu mất chỗ đã gặp thật:

1. **Mất chỗ trong lúc job đang chạy.** Bộ nhớ job dùng không cố định:
   - lúc khởi động chưa lên tới đỉnh;
   - giữa các epoch / batch, PyTorch có thể trả bớt.

   Người khác chen vào phần vừa trống. Tới lúc job cần lại thì OOM giữa chừng; pilot ZETA4.1 (2026-11-08) đã OOM lần đầu vì vậy.
2. **Mất chỗ giữa các bước của một chuỗi.** Một chuỗi nhiều run, ví dụ finetune → merge (CPU) → sinh bằng vLLM → chấm (CPU) → run kế, có những lúc **không bước nào giữ GPU**:
   - lúc một bước GPU vừa thoát;
   - lúc đang chạy bước CPU.

   Người khác lấy mất chỗ, và bước GPU sau phải chờ hàng giờ hoặc OOM. Nếu tách ra chạy tay từng lệnh thì còn mất thêm thời gian chết giữa hai lần dán lệnh.

Cách giải: **chiếm trước và giữ chỗ có chủ đích**, kể cả lúc chưa dùng hết. Nhờ vậy chuỗi chạy liền một mạch, không ai chen vào được.

### Các mức dùng

| tình huống | cờ | cơ chế |
|---|---|---|
| Một job torch (train / finetune / eval) | `run_on_free_gpu.py --reserve-gib G -- script.py …` | Lúc khởi động cấp ngay một khối G GiB rồi trả về **cache của PyTorch**, không trả driver. `torch.cuda.empty_cache` bị vô hiệu. `nvidia-smi` thấy job giữ G GiB từ đầu tới hết; mọi cấp phát sau của job lấy từ khối này. Không đủ G ngay lúc khởi động thì thoát 1 (rồi `--retries` thử lại). |
| GPU đang bị người khác chiếm, chưa đủ G | thêm `--grab-margin-gib M` | **Gom dần:** mỗi 10 s đọc bộ nhớ trống, xin thêm (trống − M) GiB, giữ phần đã có, đủ G mới chạy job. Luôn chừa M GiB cho người khác. Cứ 10 phút in "vẫn chờ" một lần. |
| Job vLLM | thêm `--release` (đi cùng `--grab-margin-gib`) | Gom đủ G rồi **nhả hết** về driver, sau đó chạy script ở tiến trình con. vLLM đo bộ nhớ trống lúc khởi động và tự giữ `vllm.gpu_mem_gib`, nên giữ trong cache của tiến trình cha vô ích. Khe hở giữa lúc nhả và lúc vLLM giữ chỉ vài giây. |
| **Chuỗi nhiều bước / nhiều run** | `gpu_hold.py --keeper DIR` chạy nền + mỗi bước GPU `--handoff DIR` | **Người giữ chỗ** giữ G GiB suốt chuỗi. Mỗi bước GPU xin đúng phần mình cần, chạy xong trả lại. Xem mục dưới. |

`cudnn.benchmark` (train / eval CE-Loc) gọi `emptyCache` ở C++ mỗi lần dò thuật toán conv (iter đầu, cỡ batch mới) ⇒ khối giữ về
driver, vá Python không chặn được. Vì vậy job có một luồng nền: mỗi 1 s, cache giữ < G thì xin lại phần thiếu (chừa lề). Ngày
2026-11-16, 9.12 trên server 2 mất khối 9 GiB ở iter đầu, người giữ chỗ làn bên lấy ngay ⇒ OOM.

**Chọn G:** G = đỉnh đo được + 2–4 GiB.
- Lấy đỉnh từ dòng "GPU đỉnh … GB" trong log pilot của đúng config / batch.
- G phải nhỏ hơn bộ nhớ đang trống; nếu không thì dùng `--grab-margin-gib` để gom dần.
- Khối < 1 MiB nằm ở pool riêng, ngoài khối giữ, nên vẫn cần khoảng dư.
- Job vượt G thì PyTorch tự xin thêm như thường (OOM thì nó nhả cache rồi thử lại).

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` được đặt mặc định trước lần chạm CUDA đầu. Nhờ vậy các khối gom dần nằm trong cùng một segment ảo liền, và tensor lớn bất kỳ dùng lại được.

### Chuỗi nhiều run: người giữ chỗ (`--keeper`) và bàn giao (`--handoff`)

**Các tiến trình** (đều dưới một `nohup bash -c '…'`):

```
bash chuỗi ─┬─ gpu_hold.py --keeper K --grab 1 --need N KEEP      (nền, sống suốt chuỗi)
            ├─ bước GPU: run_on_free_gpu.py --gpu X --grab-margin-gib 0.3 --handoff K --reserve-gib g [--release] -- script …
            ├─ bước CPU: python script …                          (không cần bàn giao)
            └─ …
```

**Giao thức file trong thư mục `K`** (mỗi chuỗi một thư mục riêng; hai chuỗi không được dùng chung `K`):

| file | ai ghi | nghĩa |
|---|---|---|
| `keeper.pid` | người giữ | PID người giữ. Bước GPU dùng nó để biết người giữ còn sống không. |
| `ready` | người giữ | Đang giữ ≥ N GiB. Chuỗi chờ file này **trước mỗi bước GPU**. Người giữ xoá nó lúc nhả. |
| `want` | bước GPU | `"PID GiB"`: bước xin nhả bấy nhiêu GiB. Bước xoá nó khi thoát (atexit). |
| `freed` | người giữ | Đã nhả xong, bước bắt đầu gom. |
| `stop` | chuỗi (`trap … EXIT`) / người dùng | Người giữ nhả hết rồi thoát. |

**Một vòng bàn giao:**
1. Chuỗi chờ `ready`.
2. Bước GPU khởi tạo CUDA (vài giây, làm **trước** khi xin để thu hẹp khe hở), rồi ghi `want` = `"PID g + margin + đệm"`. Đệm 0,5 GiB cho bước torch, 2 GiB cho `--release` (tiến trình vLLM chính và EngineCore mỗi cái ~0,5 GiB context).
   GPU kín tới mức không mở nổi context (OOM ở `mem_get_info` — nhiều người giữ chỗ cùng chuỗi, mỗi cái chỉ chừa 1 GiB; 3 làn DELTA9 2026-11-16) thì bước ghi `want` **trước**, chờ nhả rồi mới mở context (thử lại mỗi giây, tối đa 60 s).
3. Người giữ nhả **đúng phần đó**, giữ lại phần dư (KEEP − phần xin), rồi ghi `freed`. Khe hở ~1 s.
4. Bước gom đủ g (chừa 0,3 GiB) rồi chạy.
5. Bước thoát, xoá `want`.
6. Người giữ gom lại, chỉ chừa 1 GiB và đọc mỗi 1 s. Không chừa M như lần gom đầu: phần trống lúc này là chỗ của chính chuỗi vừa trả ra.
7. Đủ N thì ghi lại `ready`.

Bước CPU chạy trong lúc người giữ đang giữ lại chỗ.

**`--need N` và G (`KEEP`):**
- Người giữ ghi `ready` khi đã giữ được N GiB, rồi vẫn gom tiếp tới KEEP.
- Đặt KEEP lớn nghĩa là "còn bao nhiêu lấy hết, chừa 1 GiB".
- Lần gom đầu chừa `--grab M` (mặc định 8 GiB nếu không ghi).

**Chọn số:**
- **N** ≥ bước GPU lớn nhất cần, tính cả đệm ở trên.
- **g của bước torch** = đỉnh + khoảng nhỏ (như một job đơn).
- **g của bước vLLM `--release`** ≥ `vllm.gpu_mem_gib` của config + vài GiB. vLLM tự giữ sau khi nhả nên phần dư người giữ giữ lại là KEEP − (g + 2,3).

Lần chạy ZETA8 (KEEP 32): người giữ giữ lại 10,7 / 5,7 / 3,7 GiB suốt bước vLLM của 8.1 / 8.3 / 8.2.

### Nhiều làn song song trên một GPU: `tools/gpu_pool.py`

Nhiều keeper riêng cùng gom trên một GPU sẽ lấy chỗ của nhau (DELTA9 2026-11-16). Dùng một cây (docstring đầu file):
- **gốc** `gpu_pool.py root DIR TOTAL --margin M`: gom dần tới TOTAL = tổng (đỉnh lớn nhất mỗi làn) + 1–2 GiB dư;
- **làn** `gpu_pool.py lane DIR/<làn> NEED`: xin gốc NEED (gốc phục vụ **lần lượt**, xong khách này mới xét khách sau), giữ suốt làn;
- **bước** `run_on_free_gpu.py --handoff DIR/<làn> …` như với `--keeper`; bước xin nhiều hơn làn đang giữ thì làn xin thêm từ gốc, xong trả lại.

Gốc không gom trong lúc khách đang nhận chỗ hay ngay sau khi làn bàn giao cho bước. Bước chạy `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.
Bài học lần chạy 2026-11-16 (kẹt chéo 8 giờ):
- khởi động gốc xong (`pool.pid`) mới mở làn; làn tự chờ gốc tới 120 s. Làn tự gom khi gốc chưa lên thì gốc không tính phần đó và giữ thừa;
- bước gom nhiều khúc thì gộp lại MỘT segment: khúc model chưa chạm vẫn bị `emptyCache` của cuDNN trả về driver;
- làn gom lại sau bước quá 120 s chưa đủ thì vẫn ready, gom tiếp lúc rảnh. Chờ đủ mới ready thì các làn chờ nhau mãi.
Khuôn đầy đủ: lệnh ablation DELTA9 (`log/add/ablation/orchestra_*.log`).

### Khuôn lệnh chuỗi

Khuôn dưới đây là bản ZETA8 đã chạy thật trên server 2 (2026-11-09 → 10, `log/add/chain/zeta8_full_1109_1750.log`): 3 run × (finetune → merge → sinh vLLM → chấm), liền 16 giờ. Khuôn đã thêm 3 sửa lỗi:
- `RDY` biết dừng khi người giữ chết;
- `--gpu` ép cùng GPU với người giữ;
- `CUDA_DEVICE_ORDER`.

Dán nguyên khối vào shell. Đã `bash -n` kiểm cú pháp cả khối ngoài lẫn chuỗi bên trong.

```bash
cd /mnt/disk1/aiotlab/haitn/object-detection && git pull && git log --oneline -1 && df -h /mnt/disk1 | tail -1
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
export H=/mnt/disk1/aiotlab/haitn VP=/mnt/disk1/aiotlab/haitn/envs/vllm/bin/python CP=/mnt/disk1/aiotlab/haitn/envs/ce-locmodel/bin/python
export HF_HOME=$H/hf_cache TORCH_HOME=$H/torch_cache TMPDIR=$H/tmp HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID GPU=0 NEED=32 KEEP=32 NAME=zeta8_full D=$(date +%m%d_%H%M)
export K=$H/tmp/keeper_$NAME
export G="../tools/run_on_free_gpu.py --gpu $GPU --retries 0 --grab-margin-gib 0.3 --handoff $K"
LOG=$H/log/add/chain/${NAME}_$D.log
mkdir -p $(dirname $LOG) $K && rm -f $K/want $K/ready $K/freed $K/stop $K/keeper.pid
nohup bash -c '
T() { echo "[$(date "+%F %T")] $*"; }
RDY() { until [ -e $K/ready ]; do kill -0 $KP 2>/dev/null || { T "NGƯỜI GIỮ CHỖ ĐÃ CHẾT -> dừng chuỗi"; exit 1; }; sleep 5; done; T "giữ chỗ đủ $NEED GiB"; }
cd $H/object-detection/vlm
CUDA_VISIBLE_DEVICES=$GPU $VP ../tools/gpu_hold.py --keeper $K --grab 1 --need $NEED $KEEP &
KP=$!; trap "touch $K/stop" EXIT
T "người giữ chỗ PID $KP: GPU $GPU, ready khi đủ $NEED GiB, gom tới $KEEP GiB"
for R in 1:qwen3-vl-4b:ep003 3:llava-ov2-8b:ep002 2:internvl3.5-8b:ep002; do
  IFS=: read I N E <<< "$R"
  W=../weights/add/zeta8_$I C=config/server2/$N-sft.yaml O=../../output/add/vlm/$N-sft
  if [ -e $W/snapshots/$E.pth ]; then
    T "===== ZETA8.$I $N: đã có $E, bỏ qua finetune"
  else
    RS=""; [ -e $W/last.pth ] && RS="--resume"
    RDY; T "===== ZETA8.$I $N: finetune $RS -> $W"
    $VP $G --reserve-gib 32 -- finetune.py --config $C --save-dir $W --num-workers 8 $RS || { T "LỖI finetune ZETA8.$I"; continue; }
  fi
  T "ZETA8.$I: merge $E (CPU)"
  rm -rf $W/merged
  $VP finetune.py --config $C --save-dir $W --merge $E --out $W/merged || { T "LỖI merge ZETA8.$I"; continue; }
  RDY; T "ZETA8.$I: sinh --all (vLLM)"
  $VP $G --reserve-gib 26 --release -- run.py --config $C --all --no-score --model $W/merged --name $E --out-root $O || { T "LỖI sinh ZETA8.$I"; rm -rf $W/merged; continue; }
  T "ZETA8.$I: chấm (CPU)"
  $CP run.py --config $C --all --score-only --model $W/merged --name $E --out-root $O || T "LỖI chấm ZETA8.$I"
  rm -rf $W/merged
  T "ZETA8.$I xong -> $O/$E"
done
touch $K/stop; wait $KP
T "HẾT chuỗi $NAME"' > $LOG 2>&1 &
echo "PID $! -> $LOG"
echo "tail -f $LOG"
```

Đọc khuôn:
- **Biến `export` trước `nohup bash -c '…'`**: chuỗi nằm trong nháy đơn nên chỉ thấy biến đã `export`. `$KP` thì đặt ngay trong chuỗi.
- **`GPU=` phải trùng nhau ở người giữ (`CUDA_VISIBLE_DEVICES=$GPU`) và mọi bước (`--gpu $GPU` trong `$G`).**
  - Server 2 chỉ có GPU 0.
  - Server 1 có 3 GPU. Nếu không ép, `run_on_free_gpu` chọn GPU **trống nhất**, mà đó gần như chắc chắn **không phải** GPU người giữ đang chiếm. Hậu quả: bước chạy sang GPU khác, còn người giữ vẫn nhả chỗ của mình một cách vô ích.
  - `CUDA_DEVICE_ORDER=PCI_BUS_ID` làm chỉ số CUDA khớp với chỉ số `nvidia-smi`.
  - Chọn GPU bằng dòng `nvidia-smi` đầu khối.
- **`--retries 0`**: bước lỗi thì chuỗi ghi `LỖI …` rồi sang run kế (`|| { …; continue; }`) thay vì thử lại. Lỗi code thì thử lại vô ích; OOM thì đã có người giữ lo chỗ.
- **Chuỗi idempotent**, chạy lại sau khi sửa lỗi là tiếp được từ chỗ dừng:
  - có snapshot epoch đích thì bỏ qua finetune;
  - có `last.pth` thì `--resume`;
  - `merged/` dựng lại mỗi lần và xoá sau khi chấm.
- **`trap "touch $K/stop" EXIT`**: chuỗi thoát vì bất kỳ lý do gì (xong, `exit`, SIGTERM) thì người giữ cũng nhả và thoát theo. Dòng `touch $K/stop; wait $KP` cuối chuỗi chờ người giữ nhả xong mới in "HẾT".
- **`RDY`** chờ `ready` trước **mỗi** bước GPU, không chờ trước bước CPU. Người giữ chết (bị kill, lỗi CUDA) thì `RDY` dừng cả chuỗi thay vì chờ mãi.
  - Bước GPU đang chạy mà người giữ chết thì không sao: bước thấy `keeper.pid` không còn sống thì tự gom như thường.
- **Bước CPU** (merge, chấm `--score-only`) gọi thẳng `python`, không qua `$G`.
- Dòng `rm -f $K/…` trước `nohup` dọn file tín hiệu còn sót từ lần trước. Người giữ cũng tự dọn lúc khởi động.

Đổi sang chuỗi khác: giữ nguyên phần đầu, `T`, `RDY`, dòng người giữ chỗ, `trap` và hai dòng cuối; chỉ thay thân vòng `for`. Ví dụ chuỗi eval CE-Loc nhiều snapshot:
- `cd $H/object-detection/ce_localization`;
- mỗi bước: `RDY; $CP $G --reserve-gib <đỉnh+2> -- eval.py …`;
- `NEED` = `--reserve-gib` + 0,8.

### Theo dõi

```bash
tail -f $LOG                                   # [giữ chỗ] ... = người giữ, [gom GPU] ... = bước GPU, [ngày giờ] ... = chuỗi
grep -e "giữ chỗ" -e "gom GPU" -e LỖI $LOG | tail -20
ls $K                                          # ready = đang giữ đủ; want + freed = một bước GPU đang chạy
nvidia-smi --query-compute-apps=pid,used_memory --format=csv     # PID người giữ chiếm KEEP GiB (hoặc phần dư khi bước chạy)
```

Dòng log chuẩn của một vòng:

```
[giữ chỗ] đủ 32.0 GiB -> ready
[gom GPU] xin người giữ chỗ (…/keeper_zeta8_full) nhả 32.8 GiB
[giữ chỗ] nhả 32.0 GiB cho bước GPU (PID …), giữ lại 0.0
[gom GPU] người giữ chỗ đã nhả
[gom GPU] đủ 32.0 GiB sau 0 phút
[giữ GPU] giữ 32.0 GiB cho cả job (gom dần; empty_cache vô hiệu)
... (job) ...
[giữ chỗ] bước GPU xong -> gom lại (chừa 1 GiB, đọc mỗi 1 s)
```

Lần gom đầu có thể lâu nếu GPU đang bị chiếm; ZETA8 chờ 28 phút mới đủ 32 GiB. Log in "vẫn chờ" mỗi 10 phút.

### Dừng chuỗi giữa chừng

Chỉ `kill <PID chuỗi>` thì **chưa đủ**:
- bước đang chạy vẫn tiếp tục;
- bash trên Linux hoãn `trap` (tạo `stop`) tới khi tiến trình con hiện tại kết thúc;
- nếu giết bước trước, chuỗi coi đó là lỗi rồi **mở bước kế**.

Phải giết **từ gốc xuống** cả cây tiến trình. Chú ý `--release` chạy vLLM ở tiến trình con, và vLLM còn tiến trình EngineCore.

```bash
CHAIN=<PID chuỗi>  K=/mnt/disk1/aiotlab/haitn/tmp/keeper_<NAME>
kill_tree() { local c=$(pgrep -P $1); kill $1 2>/dev/null; for x in $c; do kill_tree $x; done; }
kill_tree $CHAIN; sleep 5
touch $K/stop; pgrep -af -- "$K" || echo "không còn tiến trình nào của chuỗi"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

`kill_tree` gom danh sách con **trước** khi giết cha. Lý do: cha chết thì con bị chuyển sang init, và `pgrep -P` không còn tìm được chúng. Cách này đã thử trên cây giả 3 tầng: dừng sạch, chuỗi không kịp mở bước kế, `trap` vẫn tạo `stop`.

Nếu sau bước trên vẫn thấy PID của mình trong `nvidia-smi`, dùng `kill -9 <PID>`. Bước bị `kill -9` không xoá được `$K/want`; lần chạy sau dòng `rm -f` sẽ dọn.

**Không dùng `kill -- -<PGID>`.** Nhóm tiến trình của job nền chỉ tách riêng khi shell bật job control. Nếu PGID trùng với shell đăng nhập thì lệnh này giết luôn phiên ssh.

### Lưu ý khác

- Giữ chỗ là chiếm GPU của người khác có chủ đích. Chỉ giữ khi có chuỗi đang chạy thật, và luôn để `trap` / `stop` nhả khi xong.
  - Lần gom đầu chừa M GiB (`--grab`) cho người khác.
  - Chỉ đặt KEEP > N khi muốn chiếm thêm có chủ đích.
- Người giữ chỉ giữ bộ nhớ, không dùng compute. GPU utilization lúc chạy bước CPU là 0 %, đó là bình thường.
- Môi trường của người giữ phải có torch (env `vllm` hoặc `ce-locmodel`). Server 1 không ghi được `/tmp` nên `K` và `TMPDIR` đặt dưới `$H/tmp`.
- Test: `tests/test_run_on_free_gpu.py` (giả CUDA, chạy ở local): giữ / gom / `--release`, người giữ bàn giao rồi gom lại, `--need`,
  nhả một phần, `want` = "PID GiB".
