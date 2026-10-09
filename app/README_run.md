# Smart Traffic Tracking — hướng dẫn cài và chạy app

Ứng dụng Streamlit phát hiện, theo dõi (ByteTrack/BoT-SORT) và đếm phương tiện qua vạch trong video,
dùng mô hình YOLO11m `best_final.pt` (BDD100K). App chỉ phát hiện **4 lớp mục tiêu: car, bus, truck, motor**. Chạy được trên Windows và Linux, có GPU NVIDIA hoặc chỉ CPU.

## Cấu trúc thư mục

```
app/
├── app.py
├── best_final.pt            # mô hình (đặt cùng thư mục với app.py)
├── requirements-gpu.txt
├── requirements-cpu.txt
├── README_run.md
└── .streamlit/config.toml   # maxUploadSize = 1000 MB
```

Yêu cầu Python 3.10 đến 3.15 (PyTorch 2.14.1 có bản cho các phiên bản này). Nên dùng môi trường ảo:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
python -m pip install --upgrade pip
```

## 1. Máy có GPU NVIDIA (CUDA)

Kiểm tra driver bằng `nvidia-smi` (dòng "CUDA Version" cho biết mức CUDA tối đa driver hỗ trợ).

```bash
# Bước 1: PyTorch bản CUDA 12.6 (cài TRƯỚC, đúng --index-url)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
# (driver hỗ trợ CUDA 13 có thể dùng https://download.pytorch.org/whl/cu130)

# Bước 2: các thư viện còn lại
pip install -r requirements-gpu.txt

# Kiểm tra CUDA
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Lưu ý: trên Windows, `pip install torch` từ PyPI (không có `--index-url`) sẽ cài bản **CPU**; khi đó app báo
"CUDA khả dụng: không" và tự chạy CPU.

## 2. Máy không có GPU (chỉ CPU)

```bash
# Bước 1: PyTorch bản CPU (nhẹ hơn nhiều)
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu

# Bước 2
pip install -r requirements-cpu.txt
```

Trên CPU, app tự chọn cấu hình **"CPU (nhẹ)"**: imgsz 480, xử lý 1/2 số khung (stride 2), conf 0,30, tối đa 900 khung
được xử lý, xem trước rộng 640 px. Mô hình được huấn luyện ở 640 nên giảm imgsz làm giảm độ chính xác, nhất là vật thể
nhỏ (đèn giao thông, xe máy, người đi xe). Nhóm chưa đo FPS trên CPU.

## 3. Chạy app

Chạy **từ trong thư mục `app/`** để Streamlit đọc được `.streamlit/config.toml`:

```bash
cd app
python app.py --selftest            # tự kiểm tra: nạp mô hình, suy luận ảnh giả, kiểm tra số lớp, logic đếm, ffmpeg
python app.py --selftest --device CPU
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`. Nếu phải chạy từ thư mục khác:
`streamlit run app/app.py --server.maxUploadSize 1000`.

## 4. Rút gọn `best_final.pt` (161 MB → khoảng 40 MB) bằng `strip_optimizer`

File gốc còn chứa trạng thái optimizer nên nặng khoảng 161 MB, vượt giới hạn 100 MB mỗi file của GitHub.
`strip_optimizer` bỏ optimizer và thông tin huấn luyện, chỉ giữ trọng số EMA (trong checkpoint vốn đã lưu ở FP16), nên **không đổi kết quả suy luận** (đã đối chiếu: trọng số trùng khớp từng tensor, kết quả dự đoán trên ảnh thử giống hệt).

```bash
# Ghi ra file mới, giữ nguyên file gốc:
python -c "from ultralytics.utils.torch_utils import strip_optimizer; strip_optimizer('best_final.pt', 'best_final_stripped.pt')"
```

Sau đó đổi tên `best_final_stripped.pt` thành `best_final.pt` trong thư mục `app/` (hoặc nhập đường dẫn ở thanh bên).
Nếu gọi `strip_optimizer('best_final.pt')` không có tham số thứ hai thì file gốc bị **ghi đè**, hãy sao lưu trước.

(Thư mục `app/` hiện đã có sẵn một bản `best_final.pt` đã rút gọn, 40,5 MB, tạo từ file gốc bằng lệnh trên.)

## 5. Lớp được phát hiện

App cố định chỉ phát hiện 4 lớp (không chọn thêm được):

| id | lớp | nghĩa |
|---|---|---|
| 2 | car | ô tô |
| 3 | bus | xe buýt |
| 4 | truck | xe tải |
| 6 | motor | xe máy |

## 6. Đầu ra

- Video đã chú thích (H.264 `.mp4`, phát được trong trình duyệt). Nếu chuyển mã thất bại vẫn tải được bản gốc (`mp4v`/`MJPG`).
- `class_summary.csv`: theo 4 lớp — số lượt `A->B`, `B->A`, tổng, số ID đã xuất hiện.
- `crossing_events.csv`: mỗi lượt qua vạch — `frame`, `second`, `track_id`, `class_id`, `class_name`, `direction`.
- CSV mã hóa UTF-8 có BOM để Excel trên Windows hiển thị đúng.

## 7. Xử lý sự cố

| Hiện tượng | Cách xử lý |
|---|---|
| "Không tìm thấy mô hình" | Đặt `best_final.pt` cạnh `app.py`, sửa đường dẫn ở thanh bên hoặc tải file `.pt` lên |
| Không tải được video > 200 MB | Chưa chạy từ thư mục `app/` nên không đọc `config.toml`; dùng `--server.maxUploadSize 1000` |
| Chọn GPU nhưng app chạy CPU | PyTorch đang là bản CPU; cài lại bằng `--index-url .../cu126` |
| Lần đầu theo dõi in "requirements: lap not found" | Thiếu gói `lap`; `pip install lap` (đã có trong requirements) |
| Video không phát trong trình duyệt | Chuyển mã H.264 thất bại; tải bản gốc và mở bằng VLC |
