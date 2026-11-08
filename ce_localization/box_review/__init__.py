"""Web rà box tay (người dùng, 2026-10-20): mỗi màn một ảnh + các nhóm box; bấm box = đánh dấu loại / lỗi, Enter lưu, Ctrl+Z hoàn tác.

  sources.py   nguồn dữ liệu -> danh sách màn (CE-CoCount theo frame; CE-130 theo nhánh: box vật + box lỗ)
  app.py       phiên rà (lưu jsonl, nháp, hoàn tác) + giao diện Gradio + dòng lệnh
  cegen.py     chấm kết quả CE-Gen (thành công / thất bại + box vật thêm / xoá; server 2, link share): `prepare` cache + `serve`
"""
