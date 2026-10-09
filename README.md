# Smart Traffic Monitoring

Đồ án phát hiện, theo dõi và đếm phương tiện giao thông trong video bằng **YOLO11m** + **ByteTrack/BoT-SORT**, huấn luyện trên **BDD100K** và triển khai demo bằng **Streamlit**.

## Tổng quan

1. **Phát hiện:** YOLO11m fine-tune trên BDD100K (định dạng YOLO, 10 lớp). App chỉ hiển thị 4 lớp mục tiêu: `car`, `bus`, `truck`, `motor`.
2. **Theo dõi:** gán ID cố định cho mỗi xe qua các khung hình (ByteTrack mặc định, hoặc BoT-SORT).
3. **Đếm:** người dùng đặt vạch trên video, app đếm lượt xe qua vạch theo hai hướng `A->B` và `B->A`.
4. **Xuất kết quả:** video đã chú thích (`.mp4`), `class_summary.csv`, `crossing_events.csv`.

## Kết quả mô hình (`best_final.pt`)

| Tập | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Val (10 lớp) | 0,5757 | 0,3298 | 0,7377 | 0,5299 |
| Test (10 lớp) | 0,5792 | 0,3279 | 0,6380 | 0,5382 |

mAP50-95 trên 4 lớp mục tiêu: 0,4417 (val) và 0,4305 (test).

Hạn chế: mô hình chưa hội tụ hoàn toàn; các lớp nhỏ và ít mẫu (`motor`, `bike`, `train`) còn yếu. Precision trên test thấp hơn val chủ yếu do lớp `train` chỉ có 15 mẫu val. Nhóm chưa đo FPS trên CPU.

## Cấu trúc repo

```
├── train/smart_traffic_tracking.ipynb   # huấn luyện (Kaggle)
├── val_test/val-best-traffic.ipynb      # đánh giá trên val/test
└── app/                                 # ứng dụng Streamlit
    ├── app.py
    ├── best_final.pt                    # mô hình (đã rút gọn, ~40 MB)
    ├── requirements-cpu.txt / requirements-gpu.txt
    └── README_run.md                    # hướng dẫn cài đặt chi tiết
```

## Chạy app

```bash
cd app
python -m venv .venv && source .venv/bin/activate
# CPU:
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-cpu.txt
# GPU: xem app/README_run.md

streamlit run app.py        # mở http://localhost:8501
```

Phải chạy từ thư mục `app/` để Streamlit đọc `.streamlit/config.toml` (cho phép tải video đến 1000 MB). Không có GPU thì app tự dùng cấu hình "CPU (nhẹ)".

## Dữ liệu

Dùng bộ **BDD100K** (Yu et al., CVPR 2020) đã chuyển sang định dạng YOLO, lấy từ notebook Kaggle [BDD100K with YOLO – Setup and Training Validation](https://www.kaggle.com/code/a7madmostafa/bdd100k-with-yolo-setup-and-training-validation) của tác giả `a7madmostafa`. Nhóm không tự thu thập hay gán nhãn; chỉ làm sạch nhãn train/val. Điều khoản sử dụng theo BDD100K và trang Kaggle gốc. Dữ liệu không nằm trong repo.

## Lưu ý

Metadata `model.names` trong `best_final.pt` sai thứ tự ở các id 5, 7, 8, 9 (chỉ số đánh giá không bị ảnh hưởng). App dùng bảng lớp cố định nên hiển thị đúng.
