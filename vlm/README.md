# vlm — ZETA4 (RePlan) + ZETA5 (VLM thương mại) + ZETA6 (VLM open-source) cho bài ADD, chạy theo cách của tác giả

Mỗi ảnh test gọi VLM **một lần**, xin **30 box** chỗ thêm một vật cùng lớp. Chấm bằng đúng code chấm của ZETA / CE-Loc
(`baseline/score.py`). **Kết quả chính = C-NLL của box `score` (confidence model trả) cao nhất** (người dùng chốt 2026-11-04;
ZETA1–3 / DELTA vẫn lấy box C-NLL thấp nhất trong 30). Tham khảo: min C-NLL trong 30 box, cùng lần gọi.

**Làm giống code GPT của tác giả nhất có thể** (`refs/inference_layout.py` + `refs/example4.py`, gpt-4.1-mini; người dùng chốt
2026-11-03). Chung cho mọi model (`config/base.yaml`):
- system prompt và câu hỏi user nguyên văn ("Canvas size: WxH.\nObject(s) to insert: …"), text rồi ảnh;
- canvas = cỡ ảnh gốc, ảnh gửi nguyên file (`image.max_side: null`), toạ độ pixel canvas;
- JSON theo schema `insertion_layout` của tác giả (`candidates`: name, box, score, placement_type, reason), gửi **qua API**
  (OpenRouter `response_format` + `provider.require_parameters`, vLLM structured outputs; strict False),
  xếp theo `score` giảm dần. Model / bản vLLM không nhận schema ⇒ dừng (`Fatal`);
- few-shot = 2 hội thoại của tác giả (`icl: author`, file `icl/author.json`; ảnh lấy từ `data/all_phase2_V2/`), hay `icl: 0`
  (zero-shot);
- không đặt sampling (mặc định của model); suy nghĩ tắt / thấp nhất có thể (gpt-4.1-mini không suy nghĩ).

Chỉ khác tác giả: xin top 30 thay top 3 (schema `maxItems` 30); 0 box hợp lệ thì thử lại; ví dụ 1 của tác giả sai khoá (`"chair"` /
`"seashell"`), đã sửa trong `icl/author.json`. ⚠️ Ví dụ 1 là ảnh GT nhánh **test** `2_b1` của CE-130 ⇒ 4 mẫu `test_all`
(`2_b1_t1..t4`) trùng cảnh; code in cảnh báo.

Parse: JSON `candidates` (xếp theo score) -> JSON chung (list 4 số, dict khoá box) -> regex nhóm 4 số; bỏ phần `<think>`.

Quyết định, lý do chọn model, kết quả: `docs/EXPERIMENT_ZETA.md` mục 0 + 13 (ở gốc `multi_condition/`).

| nhóm | config | model | chạy ở đâu | ghi chú |
|---|---|---|---|---|
| **ZETA4** RePlan | `config/server2/replan-7b.yaml` | `TainU/RePlan-Qwen2.5-VL-7B` (planner) | server 2, vLLM | prompt / parse riêng (`prompt: replan`): greedy = box chính + 29 sample; ZETA mục 14 |
| **ZETA4.1 / 8.x** finetune | `config/server2/{replan-7b,qwen3-vl-4b,internvl3.5-8b,llava-ov2-8b}-sft.yaml` | snapshot LoRA đã merge (`--model DIR`) | server 2 | SFT + LoRA (`finetune.py`, mục dưới); ZETA8: prompt tác giả zero-shot, `ask_k: 1` + `samples` (greedy + 29) |
| **ZETA5** thương mại | `config/openrouter/gpt-6-luna.yaml` | `openai/gpt-6-luna` | OpenRouter | họ OpenAI; tắt suy nghĩ; không nhận temperature |
| | `config/openrouter/gpt-4.1-mini.yaml` | `openai/gpt-4.1-mini` | OpenRouter | đúng model tác giả dùng; $0,4 / $1,6 |
| **ZETA6** open-source | `config/kaggle/qwen3-vl-4b.yaml` (+ `server2/`, cùng hash) | `Qwen/Qwen3-VL-4B-Instruct` | Kaggle 2 × T4 / server 2, vLLM | fp16, `coord: norm1000` |
| | `config/kaggle/llava-ov2-8b.yaml` (+ `server2/`, cùng hash) | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | Kaggle 2 × T4 / server 2, vLLM | fp16, `coord: norm1000` |
| | `config/server2/internvl3.5-8b.yaml` | `OpenGVLab/InternVL3_5-8B` | server 2, vLLM | bf16, `coord: norm1000` |
| | `config/openrouter/glm-5.3-flash.yaml` | `z-ai/glm-5.3-flash` | OpenRouter (320B MoE) | suy nghĩ `low` (không tắt được) |

Đã thử và loại (số pilot ở ZETA mục 13.2): Gemini Robotics-ER 2, Perceptron, Elorian Glimpse, Qwen3.8-27B. Backend OpenAI trực tiếp
và Batch API bỏ 2026-11-07 (key OpenAI hết hạn, mọi lần chạy full / latency đều qua OpenRouter; git giữ lịch sử).

## Key

Chép `.env.example` thành `vlm/.env` rồi điền `OPENROUTER_API_KEY` (gitignore: đừng commit, đừng dán key vào chat / log).
- Code chỉ in tên biến và 4 ký tự cuối; kiểm key bắt đầu bằng `sk-or-`, không có ký tự trắng / lạ. Biến đã `export` được ưu tiên hơn
  file; trống / `None` / `null` / `<...>` = không có key ⇒ dừng.
- OpenRouter in hạn mức key + số dư `/credits` lúc khởi động (số dư <= 0 ⇒ dừng); tiền = `usage.cost` thật; `--max-cost` chặn một lần chạy.

## Chạy (từ `object-detection/vlm/`)

| cờ | làm gì |
|---|---|
| `--dry-run` | in system, lượt few-shot và prompt 2 ảnh đầu mỗi tập; không gọi model, không cần key |
| `--pilot K` | K ảnh / tập, cố định theo `seed`; in tiền thật, token, thời gian và **ước cho cả tập** |
| `--all` | toàn bộ: CE-130 `test_all` 6.122 ảnh + CoCount `review5` 1.263 ảnh. Pilot là tập con ⇒ dùng lại câu trả lời pilot |
| `--dataset ce130 \| cocount \| all` | mặc định `all` (cả hai tập) |
| `--max-cost USD` | dừng êm khi tiền lần chạy này chạm ngưỡng; câu trả lời đã lưu, chạy lại cùng lệnh để tiếp |
| `--no-score` / `--score-only` | tách sinh câu trả lời (venv vLLM) và chấm (venv chính) |

Box vật CoCount đã rà tay: `--cocount-review` mặc định `../data/cocount_review.jsonl` (`object-detection/data/`, cùng tên trong zip).

- **Resume**: `responses.jsonl` ghi từng câu trả lời ngay khi có; chạy lại cùng lệnh thì bỏ qua ảnh đã xong. Đổi prompt / model /
  sampling ⇒ `prompt_hash` khác ⇒ hỏi lại.
- **Lỗi API** (sau backoff; 402 in-flight thì chờ rồi thử lại) không được ghi: chạy lại sẽ hỏi lại ảnh đó. 20 lỗi liên tiếp ⇒ dừng.
- **0 box hợp lệ**: thử lại 2 lần ở nhiệt 0,7, rồi chấm bằng box cả ảnh (bị phạt), đếm `n_fail`. Thiếu box thì lặp vòng cho đủ 30
  (box score cao nhất và min C-NLL không đổi), đếm `n_short`.
- **Chọn box**: box xếp theo `score` giảm dần (bằng nhau giữ thứ tự model trả; box score cao nhất mà hỏng — diện tích 0 — thì lấy box kế
  tiếp; box không có score xếp sau). Chấm parse lại từ `text` trong `responses.jsonl` ⇒ `--score-only` chấm lại câu trả lời cũ theo
  luật này. Ảnh không box nào có score (JSON không theo schema / regex) ⇒ box đầu model trả, đếm `n_noscore`.
- **OpenRouter**: bỏ tham số model không nhận (theo `/api/v1/models`), in hạn mức key + số dư lúc khởi động.
- **Tiến độ** mỗi ~60 s (ảnh xong, ETA, $, token vào / ra / suy nghĩ, 0 box / thiếu box). **Ước cả tập** in cuối lần chạy, ghi vào
  `<name>_run.json`: tiền = TB $/ảnh × số ảnh (kèm p90), giờ = số ảnh × giây / lần gọi ÷ song song.

Output ở `output/add/vlm/<model>/{ce130_test_all,cocount_review5}/` (dưới gốc `multi_condition/` hay `/mnt/disk1/aiotlab/haitn/`),
tên file theo `pilot<K>` hay `test`:

| file | nội dung |
|---|---|
| `<name>.jsonl` | **1 dòng / ảnh, mọi thông tin**: `class`, `orig_wh`, `n_obj`, `n_valid`, `cnll_top` (box score cao nhất = số chính), `cnll_min`, `cnll_gt` (từng lỗ GT), `boxes` (xếp theo score; mỗi box: `box` pixel ảnh gốc, `box_512` canvas chấm, `score`, `name` / `placement_type` / `reason` nếu model trả, `cnll`, CoCount thêm `cnll_objsize[_exemplar]`; 0 box ⇒ 1 box cả ảnh `fallback`), `attempts`, token, `cost`, `sec`, `request_sec`, `finish`, `provider`, `gen_id`, `text` thô |
| `<name>.json` | số tổng: `top_score` (**chính**, K = 1) và `min_cnll_30` (tham khảo), mỗi cái `results` + `prior` theo khoá (`inpainted` / `cocount[_objsize[_exemplar]]`); `selected` (0 box, thiếu box, token, tiền), `full_estimate`, `this_run`, `n_noscore` |
| `responses.jsonl` | câu trả lời thô ghi dần lúc chạy — để resume (chung pilot / all), không cần đọc |

**Thời gian / ảnh**: lúc chạy chỉ ghi để theo dõi — `sec` = cả lần gọi (gồm chờ thử lại; vLLM theo khối ⇒ thông lượng), `request_sec`
= request thành công cuối (API; vLLM khối 1 ảnh). **Số báo cáo = latency 1 ảnh của `tools/latency.py`** (một định nghĩa chung mọi
method, chốt 2026-11-04 — docstring tool), không lấy từ hai trường này.

Chưa đủ câu trả lời (dừng giữa chừng / `--no-score`) ⇒ `<name>.json` chỉ có tiền / token, chưa có `<name>.jsonl`. Kaggle còn ghi
`diagnose.json` (số box, finish, token, toạ độ trong ảnh theo pixel / 0–1000).

Log `log/add/vlm/<model>/`: `pilot<K>_<MMDD_HHMM>.log`, `full_<MMDD_HHMM>.log` (sinh `--all`), `score_full_<MMDD_HHMM>.log`
(`--score-only`); chuỗi nhiều model / bước: `log/add/eval_chain/`.

## So box các VLM — `plot_compare.py`

Hình 2 × 3 / ảnh (chữ tiếng Anh): ô Original (ảnh + vật đang có + box GT) và 5 ô model (mặc định GLM-5.3-Flash, gpt-4.1-mini, GPT-6 Luna,
LLaVA-OV-2-8B, Qwen3-VL-4B; `--models` chọn khác, vd `replan-7b`, `internvl3.5-8b`): box dự đoán đỏ mảnh + một box đỏ đậm —
`--bold min` (mặc định) box C-NLL thấp nhất, `--bold top` box
`score` cao nhất. Đọc `output/add/vlm/<model>/<tập>/test.jsonl` (model chưa chạy full thì `pilot<K>.jsonl` K lớn nhất) + ảnh, ghi
`output/add/viz/vlm_compare/<tập>/<image_id>.png`.

```bash
python plot_compare.py --dataset all --pilot 20      # 20 ảnh pilot (cùng tập run.py --pilot 20) cả hai tập; --n 1 = thử 1 ảnh
python plot_compare.py --dataset all --pilot 10 --bold top --models replan-7b glm-5.3-flash gpt-4.1-mini llava-ov2-8b internvl3.5-8b \
    --out ../../output/add/viz/vlm_compare_replan       # chọn model (tối đa 5)
```

## Chạy vLLM (Kaggle 2 × T4, server 2)

- Venv vLLM riêng: `vllm==0.33.0` + `pyyaml scipy matplotlib` (kèm torch / transformers); không cài `requirements.txt` của repo (ghim
  `numpy<2`). vLLM 0.33 là wheel CUDA 13 ⇒ driver >= 580. Sinh câu trả lời ở venv vLLM (`--no-score`), chấm ở venv chính (`--score-only`).
- Code đọc bộ nhớ trống của GPU mà `tools/run_on_free_gpu.py` chọn; thiếu `vllm.gpu_mem_gib` thì dừng ngay (không chờ). Khoá `vllm.*` và
  khoá `engine` chỉ đổi tốc độ / bộ nhớ (`max_num_seqs`, `tensor_parallel_size`, `limit_mm_per_prompt`, …) nằm ngoài `prompt_hash` ⇒
  chỉnh giữa các phiên vẫn resume. **Không** đổi `dtype`, `max_model_len`, `hf_overrides` / ảnh, sampling (vào hash ⇒ hỏi lại từ đầu).
- **Toạ độ** `coord`: `pixel` (mặc định, như prompt tác giả) | `norm1000` (Qwen3-VL, LLaVA-OV-2, InternVL3.5 trả thang 0–1000 dù prompt
  xin pixel — pilot 2026-11-04); chỉ đổi cách đọc, không vào hash.
- **Kaggle** (`config/kaggle/`; T4 không bf16 / FP8 / FlashAttention 2 ⇒ `dtype: half`, `tensor_parallel_size: 2`, bộ nhớ tính MỖI
  card). `vllm.data_parallel: 2` ⇒ 2 tiến trình con (`--shard i/2`, mỗi con một GPU), ghi `responses_shard<i>.jsonl` rồi gộp. Notebook
  `notebooks/vlm_kaggle.ipynb` (local, gitignore): cài vLLM, nối `responses.jsonl` phiên trước, `--pilot K` / `--all` + `diagnose.json`,
  gói `zeta6_<tag>.zip` (giải nén vào gốc `multi_condition/` hay `/mnt/disk1/aiotlab/haitn/`, rồi `--score-only`).
- **Server 2** RTX PRO 5000 48 GB (`config/server2/`; Qwen3-VL-4B / LLaVA-OV-2-8B = bản Kaggle một card, cùng hash; InternVL3.5-8B
  bf16; RePlan). Venv `envs/vllm`; không có `nvcc` ⇒ **`export VLLM_USE_FLASHINFER_SAMPLER=0`**; IPv6 hỏng ⇒ tải weight trước bằng IPv4
  rồi chạy `HF_HUB_OFFLINE=1`; cache compile vào `haitn/` (`VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR`). GPU dùng chung ⇒ `vllm.gpu_mem_gib`
  vừa số GPU đang trống. Latency 1 ảnh mọi method chạy local đo ở đây (ZETA mục 4).

## Finetune VLM — ZETA4.1 RePlan, ZETA8.x open-source (`finetune.py`, server 2, venv vLLM)

SFT + LoRA (tự viết, `lora.py`) trên CE-130 train (ZETA mục 14.1, 16). Phụ thuộc kiến trúc ở `adapters.py` (`finetune.adapter`: `qwen`
= Qwen2.5-VL / Qwen3-VL, `internvl` = InternVL3.5 (tile 448 + `<IMG_CONTEXT>`), `llava_ov2` = LLaVA-OV-2 (code remote); merge chép file
gốc không phải weight để vLLM nạp được); dữ liệu / câu đích / gom batch ở
`sft.py` — prompt = đúng đường suy luận (`prompt.build_messages`, thứ tự text / ảnh như `backends.chat_messages`), ảnh cùng
`engine.mm_processor_kwargs` với vLLM; đích RePlan (`prompt: replan`, toạ độ `qwen_resized`) hoặc JSON tác giả 1 candidate (`coord`).
Loss chỉ chiếu `lm_head` ở token có nhãn; mỗi batch sắp theo độ dài ước (token ảnh, cache `lengths.json`) trước khi chia micro
(`group_by_length`); resume kiểm config (`RESUME_KEYS`). Snapshot mỗi epoch `snapshots/epNNN.pth` (chỉ LoRA), `last.pth`
(+ optimizer, `--resume` ở biên epoch); `--limit N --max-steps M` cho pilot (ghi `itNNNNN.pth`). Giữ trước bộ nhớ GPU:
`run_on_free_gpu.py --reserve-gib` (đỉnh pilot + khoảng nhỏ).

Eval greedy + N câu sample cho mọi prompt: khối `samples: {n, temperature, top_p}` (tên cũ `replan: {samples, …}` vẫn đọc); `k` = số box
CHẤM (30), `ask_k` = số box prompt / schema xin mỗi câu (ZETA8: 1); vLLM ràng buộc schema cho cả câu greedy lẫn câu sample.

```bash
python ../tools/run_on_free_gpu.py -- finetune.py --config config/server2/replan-7b-sft.yaml --save-dir ../weights/add/zeta4_1
python finetune.py --config config/server2/replan-7b-sft.yaml --save-dir ../weights/add/zeta4_1 --merge ep001 --out ../weights/add/zeta4_1/merged_ep001
python ../tools/run_on_free_gpu.py -- run.py --config config/server2/replan-7b-sft.yaml --all --no-score \
    --model ../weights/add/zeta4_1/merged_ep001 --name ep001 --out-root ../../output/add/vlm/replan-7b-sft
```
`run.py --model` / `--name` ghi đè model / tên thư mục (model vào prompt_hash ⇒ mỗi snapshot một hash).


## Dữ liệu gửi đi

- OpenRouter: config đặt `provider.data_collection: deny` ⇒ chỉ qua provider không giữ dữ liệu để huấn luyện.

Test: `python -m pytest tests/vlm -q` từ `object-detection/`, CPU, backend / HTTP giả, không gọi mạng.
