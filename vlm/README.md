# vlm — ZETA5 (VLM thương mại) + ZETA6 (VLM open-source) cho bài ADD, chạy theo cách của tác giả

Mỗi ảnh test gọi VLM **một lần**, xin **30 box** chỗ thêm một vật cùng lớp. Chấm bằng đúng code chấm của ZETA / CE-Loc
(`baseline/score.py`). Có hai kết quả: K = 30 (lấy box C-NLL thấp nhất, như ZETA1–3) và K = 1 (box có `score` cao nhất).

**Làm giống code GPT của tác giả nhất có thể** (`refs/inference_layout.py` + `refs/example4.py`, gpt-4.1-mini; người dùng chốt
2026-11-03). Chung cho mọi model (`config/base.yaml`):
- system prompt và câu hỏi user nguyên văn ("Canvas size: WxH.\nObject(s) to insert: …"), text rồi ảnh;
- canvas = cỡ ảnh gốc, ảnh gửi nguyên file (`image.max_side: null`), toạ độ pixel canvas;
- JSON theo schema `insertion_layout` của tác giả (`candidates`: name, box, score, placement_type, reason), gửi **qua API**
  (OpenRouter `response_format` + `provider.require_parameters`, Gemini `responseJsonSchema`, vLLM structured outputs; strict False),
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
| **ZETA5** thương mại | `config/gemini/robotics-er-2.yaml` | `gemini-robotics-er-2-preview` | Gemini API | chuyên không gian; thinking minimal; free tier |
| | `config/openrouter/gpt-6-luna.yaml` | `openai/gpt-6-luna` | OpenRouter | họ OpenAI; tắt suy nghĩ; không nhận temperature |
| | `config/openrouter/gpt-4.1-mini.yaml` | `openai/gpt-4.1-mini` | OpenRouter | đúng model tác giả dùng; $0,4 / $1,6 |
| **ZETA6** open-source | *(chưa có config)* | Qwen3-VL-4B | GPU server, vLLM | |
| | `config/openrouter/glm-5.3-flash.yaml` | `z-ai/glm-5.3-flash` | OpenRouter (320B MoE, không vừa A30) | suy nghĩ `low` (không tắt được) |
| | `config/server/internvl3.5-8b.yaml` | `OpenGVLab/InternVL3_5-8B` | GPU server, vLLM | FP8 chỉ weight |
| | `config/server/llava-ov2-8b.yaml` | `lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct` | GPU server, vLLM | docs không nói có ra box ⇒ pilot quyết định |

ZETA6: ưu tiên chạy trên GPU server; OpenRouter nếu rẻ; nặng quá thì hạ bản nhỏ hơn. Đã thử và loại: Perceptron, Elorian Glimpse,
Qwen3.8-27B — xem `docs/EXPERIMENT_ZETA.md` mục 0.

## Key

Chép `.env.example` thành `vlm/.env` rồi điền `OPENROUTER_API_KEY` và `GEMINI_API_KEY`.
- `vlm/.env` đã bị gitignore: đừng commit, đừng dán key vào chat / log.
- Code chỉ in tên biến và 4 ký tự cuối của key; kiểm key OpenRouter bắt đầu bằng `sk-or-`, key không có ký tự trắng / lạ.
- Biến môi trường đã `export` được ưu tiên hơn file.

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
  (min C-NLL không đổi), đếm `n_short`.
- **OpenRouter**: bỏ tham số model không nhận (theo `/api/v1/models`), in hạn mức key + số dư lúc khởi động.
- **Tiến độ** mỗi ~60 s (ảnh xong, ETA, $, token vào / ra / suy nghĩ, 0 box / thiếu box). **Ước cả tập** in cuối lần chạy, ghi vào
  `<name>_run.json`: tiền = TB $/ảnh × số ảnh (kèm p90), giờ = số ảnh × giây / lần gọi ÷ song song (Gemini còn chặn theo `rpm`).

Output ở `output/add/vlm/<model>/{ce130_test_all,cocount_review5}/` (dưới gốc `multi_condition/` hay `/mnt/disk1/aiotlab/haitn/`),
tên file theo `pilot<K>` hay `test`:

| file | nội dung |
|---|---|
| `responses.jsonl` | mọi câu trả lời (text thô, box, token, cost, giây) |
| `<name>_raw.jsonl` | 30 box / ảnh, pixel ảnh gốc, định dạng `score.py` |
| `<name>.json` + `_boxes.json` | K = 30 |
| `<name>_k1.json` | K = 1 |
| `<name>_run.json` | tiền, token, thời gian, ước cả tập |

Log: `log/add/vlm/<model>/`; log launcher (chạy nhiều model một lượt) ở `log/add/vlm/launch/`. Khuôn nohup: `docs/EXPERIMENT_ZETA.md`.

## So box các VLM — `plot_compare.py`

Hình 2 × 3 / ảnh pilot (chữ tiếng Anh): ô Original (ảnh + vật đang có + box GT) và một ô / model (box dự đoán + box C-NLL thấp nhất).
Chỉ đọc `output/add/vlm/<model>/<tập>/<name>_boxes.json` + ảnh, ghi `output/add/viz/vlm_compare/<tập>/<image_id>.png`.

```bash
python plot_compare.py --dataset ce130 --n 1      # thử 1 ảnh; --dataset all = mọi ảnh pilot cả hai tập; --name pilot10 mặc định
```

## Server A30 (vLLM) — venv riêng

- Hai model 8B ở bf16 cần ~16 GiB, quá chỗ trống ~9–10 GiB của GPU dùng chung ⇒ FP8 chỉ weight (`fp8_per_tensor`, Marlin trên
  Ampere) ~8 GiB + KV cố định `vllm.kv_cache_gib` (2 GiB).
- Few-shot tác giả có 3 ảnh / prompt ⇒ `max_model_len` 12288, ảnh giới hạn token (InternVL `max_dynamic_patch` 4, LLaVA
  `max_pixels` ~1 MP). Không vừa thì `icl: 0` hay bản model nhỏ hơn.
- Code đọc bộ nhớ trống của GPU mà `tools/run_on_free_gpu.py` đã chọn; không đủ `vllm.gpu_mem_gib` thì dừng ngay.
- vLLM >= 0.25.0 (bản đầu có LLaVA-OneVision-2). Bản mới (0.33) chỉ có wheel CUDA 13, cần driver >= 580 ⇒ `nvidia-smi` trước.
- Venv vLLM: `vllm==<bản>` + `pyyaml scipy matplotlib` (kèm torch / transformers). Không cài `requirements.txt` của repo (ghim
  `numpy<2` có thể làm hỏng vllm).
- Chạy: (1) venv vLLM `<venv>/bin/python ../tools/run_on_free_gpu.py -- run.py --config config/server/<model>.yaml --pilot 50
  --no-score`; (2) venv chính `python run.py --config config/server/<model>.yaml --pilot 50 --score-only`.

## Dữ liệu gửi đi

- OpenRouter: config đặt `provider.data_collection: deny` ⇒ chỉ qua provider không giữ dữ liệu để huấn luyện.
- Gemini free tier: Google **dùng dữ liệu để cải thiện sản phẩm**, có thể có người đọc; bật billing thì không. Free chỉ nên cho pilot.

Test: `python -m pytest tests/vlm -q` từ `object-detection/`, CPU, backend / HTTP giả, không gọi mạng.
