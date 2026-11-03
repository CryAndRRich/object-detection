# vlm — ZETA-VLM: VLM cho bài ADD, chạy theo cách của tác giả

Mỗi ảnh test gọi VLM **một lần**, xin **30 box** chỗ thêm một vật cùng lớp. Chấm bằng đúng code chấm của ZETA / CE-Loc
(`baseline/score.py`). Có hai kết quả: K = 30 (lấy box C-NLL thấp nhất, như ZETA1–3) và K = 1 (box có `score` cao nhất).

**Làm giống code GPT của tác giả nhất có thể** (`refs/inference_layout.py` + `refs/example4.py`, gpt-4.1-mini; người dùng chốt
2026-11-03). Mặc định ở `config/base.yaml`, chung cho mọi model:
- system prompt và câu hỏi user nguyên văn ("Canvas size: WxH.\nObject(s) to insert: …"), text rồi ảnh;
- canvas = cỡ ảnh gốc, ảnh gửi nguyên file (`image.max_side: null`);
- toạ độ pixel canvas, JSON theo schema `insertion_layout` của tác giả (`candidates`: name, box, score, placement_type, reason),
  gửi **qua API** (`schema: api`, strict False), xếp theo `score` giảm dần;
- few-shot = 2 hội thoại của tác giả (`icl: author`, file `icl/author.json`; ảnh lấy từ `data/all_phase2_V2/`);
- không đặt sampling (mặc định của model); suy nghĩ tắt / thấp nhất có thể (gpt-4.1-mini không suy nghĩ).

Chỉ khác tác giả:
- xin top 30 thay top 3 (giao thức ZETA; schema `maxItems` 30);
- 0 box hợp lệ thì thử lại (tác giả không có);
- ví dụ 1 của tác giả sai khoá (`"chair"` / `"seashell"`), đã sửa trong `icl/author.json`.

⚠️ Ví dụ 1 của tác giả là ảnh GT nhánh **test** `2_b1` của CE-130, nên 4 mẫu `test_all` (`2_b1_t1..t4`) trùng cảnh; code in cảnh báo.

Khoá dự phòng, đổi trong config của từng model nếu pilot hỏng:
- `schema: prompt`: backend không nhận schema qua API thì chép schema vào cuối system;
- `format: qwen | gemini | perceptron | json_norm`: kiểu box gốc 0–1000 của model, nếu model không ra được pixel canvas;
- `icl: 0` (zero-shot) hay `icl: N` (N mẫu CE-130 train, đáp án là lỗ GT).

Quyết định, đối chiếu code tác giả, lý do chọn model, kết quả: `docs/EXPERIMENT_ZETA.md` mục 13 (ở gốc `multi_condition/`).

| nhóm | config | model | ghi chú |
|---|---|---|---|
| **OpenRouter** (4) | `config/openrouter/qwen3.8-27b.yaml` | `qwen/qwen3.8-27b` | open weight; tắt suy nghĩ |
| | `config/openrouter/glm-5.3-flash.yaml` | `z-ai/glm-5.3-flash` | open weight; suy nghĩ `low` (không tắt được) |
| | `config/openrouter/perceptron-mk1.5.yaml` | `perceptron/perceptron-mk1.5` | thương mại, "physical AI"; tắt suy nghĩ |
| | `config/openrouter/gpt-6-luna.yaml` | `openai/gpt-6-luna` | thương mại, họ OpenAI; tắt suy nghĩ; không nhận temperature |
| **Server A30** (2, vLLM) | `config/server/internvl3.5-8b.yaml` | `OpenGVLab/InternVL3_5-8B` | open weight; FP8 chỉ weight; schema bằng structured outputs của vLLM |
| | `config/server/llava-ov2-8b.yaml` | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | open weight; docs không nói có ra box ⇒ pilot quyết định |
| **Gemini API** (1) | `config/gemini/robotics-er-2.yaml` | `gemini-robotics-er-2-preview` | thương mại, chuyên không gian; schema qua `responseJsonSchema`; thinking minimal; free tier |
| *(đối chiếu, ngoài 7 model)* | `config/openrouter/gpt-4.1-mini.yaml` | `openai/gpt-4.1-mini` | đúng model tác giả dùng; $0,4 / $1,6 |

Cả 5 model OpenRouter có `structured_outputs` (kiểm `/api/v1/models` 2026-11-03). Khi gửi schema, backend thêm
`provider.require_parameters` và bỏ các tham số model không nhận.

## Key

Chép `.env.example` thành `vlm/.env` rồi điền `OPENROUTER_API_KEY` và `GEMINI_API_KEY`.
- `vlm/.env` đã bị gitignore: đừng commit, đừng dán key vào chat / log.
- Code chỉ in tên biến và 4 ký tự cuối của key.
- Biến môi trường đã `export` được ưu tiên hơn file.

## Chạy (từ `object-detection/vlm/`)

| cờ | làm gì |
|---|---|
| `--dry-run` | in system, lượt few-shot và prompt 2 ảnh đầu mỗi tập; không gọi model, không cần key |
| `--pilot K` | K ảnh / tập, cố định theo `seed`; in tiền thật, token, thời gian và **ước cho cả tập** |
| `--all` | toàn bộ: CE-130 `test_all` 6.122 ảnh + CoCount `review5` 1.263 ảnh. Pilot là tập con ⇒ dùng lại câu trả lời pilot, không trả tiền hai lần |
| `--dataset ce130 \| cocount \| all` | mặc định `all` (cả hai tập) |
| `--max-cost USD` | dừng êm khi tiền lần chạy này chạm ngưỡng; câu trả lời đã lưu, chạy lại cùng lệnh để tiếp |
| `--no-score` / `--score-only` | tách sinh câu trả lời (venv vLLM) và chấm (venv chính) |

- **Resume**: `responses.jsonl` ghi từng câu trả lời ngay khi có. Chạy lại cùng lệnh thì bỏ qua ảnh đã xong. Đổi prompt / model /
  sampling ⇒ `prompt_hash` khác ⇒ hỏi lại.
- **Lỗi API** (sau khi đã thử lại theo backoff) không được ghi: chạy lại sẽ hỏi lại ảnh đó. 20 lỗi liên tiếp ⇒ dừng (hết quota /
  mạng).
- **Model trả lời nhưng 0 box hợp lệ**: thử lại 2 lần ở nhiệt 0,7, rồi chấm bằng box cả ảnh (bị phạt) và đếm `n_fail`. Thiếu box
  thì lặp vòng các box đã có cho đủ 30 (min C-NLL không đổi), đếm `n_short`.
- **Tiến độ**: in mỗi ~60 s, gồm số ảnh đã xong, ETA, $ lần này, $ trung bình / ảnh, token vào / ra / suy nghĩ, số ảnh 0 box / thiếu
  box.
- **Ước cả tập**: in cuối lần chạy, ghi vào `<name>_run.json`. Tiền = TB $/ảnh × số ảnh (kèm p90). Giờ = số ảnh × giây / lần gọi ÷
  song song (Gemini còn bị chặn theo `rpm`).

Output nằm ở `../../output/add/vlm/<model>/{ce130_test_all,cocount_review5}/` (trên server: `/mnt/disk1/aiotlab/haitn/output/...`),
tên file theo `pilot<K>` hay `test`:

| file | nội dung |
|---|---|
| `responses.jsonl` | mọi câu trả lời (text thô, box, token, cost, giây) |
| `<name>_raw.jsonl` | 30 box / ảnh, pixel ảnh gốc, định dạng `score.py` |
| `<name>.json` + `_boxes.json` | K = 30 |
| `<name>_k1.json` | K = 1 |
| `<name>_run.json` | tiền, token, thời gian, ước cả tập |

Gợi ý khi chạy: log ở `log/add/vlm/<model>/`, theo khuôn nohup trong `docs/EXPERIMENT_ZETA.md`.

## Server A30 (vLLM) — venv riêng

- Hai model 8B ở bf16 cần ~16 GiB, quá chỗ trống ~9–10 GiB của GPU dùng chung. Nên dùng FP8 chỉ weight (`fp8_per_tensor`, kernel
  Marlin trên Ampere), khoảng 8 GiB weight, cộng KV cố định `vllm.kv_cache_gib` (2 GiB).
- Few-shot của tác giả có 3 ảnh / prompt, nên `max_model_len` 12288 và ảnh bị giới hạn token (InternVL `max_dynamic_patch` 4,
  LLaVA `max_pixels` ~1 MP). Không vừa bộ nhớ thì thử `icl: 0` hoặc bản model nhỏ hơn.
- Code đọc bộ nhớ trống của GPU mà `tools/run_on_free_gpu.py` đã chọn. Không đủ `vllm.gpu_mem_gib` thì dừng ngay (vLLM cũng báo lỗi
  khi utilization vượt bộ nhớ trống).
- vLLM phải >= 0.25.0 (bản đầu có LLaVA-OneVision-2). Bản mới (0.33) chỉ có wheel CUDA 13, cần driver >= 580 ⇒ chạy `nvidia-smi`
  trước rồi chọn bản hợp driver.
- Venv vLLM cài `vllm==<bản>` + `pyyaml scipy matplotlib`, có kèm torch / transformers. Không cài `requirements.txt` của repo: ghim
  `numpy<2` có thể làm hỏng vllm.
- Cách chạy:
  1. Sinh câu trả lời bằng python của venv vLLM:
     `<venv>/bin/python ../tools/run_on_free_gpu.py -- run.py --config config/server/<model>.yaml --pilot 50 --no-score`
  2. Chấm ở venv chính: `python run.py --config config/server/<model>.yaml --pilot 50 --score-only`

## Dữ liệu gửi đi

- OpenRouter: config đặt `provider.data_collection: deny`, nên chỉ đi qua provider không giữ dữ liệu để huấn luyện.
- Gemini free tier: Google **dùng dữ liệu để cải thiện sản phẩm**, có thể có người đọc. Bật billing thì không. Gói free chỉ nên dùng
  cho pilot.

Test: `python -m pytest tests/vlm -q` từ `object-detection/`, chạy CPU, backend / HTTP giả, không gọi mạng.
