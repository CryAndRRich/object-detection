"""User study bài ADD (docs/EXPERIMENT_GAMMA.md mục 17): người chấm xem 4 box đề xuất / (ảnh test, model), đánh dấu box không ổn.

  1. `eval.py --dump-boxes` (server, GPU): 30 box thô / mẫu test của từng model
  2. `build.py`  : gộp các dump -> `items.json` (4 box / model / ảnh theo `selection.py`, thứ tự màn xáo ngẫu nhiên, tên hiển thị model)
  3. `app.py`    : web Gradio chấm (local, không model), ghi `ratings.jsonl`
  4. `score.py`  : `ratings.jsonl` -> chỉ số
"""
