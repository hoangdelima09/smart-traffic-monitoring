"""Smart Traffic Tracking — ứng dụng Streamlit phát hiện, theo dõi và đếm phương tiện.

Mô hình: YOLO11m huấn luyện trên BDD100K (10 lớp), tệp `best_final.pt` đặt cùng thư mục với file này.
Ứng dụng chỉ phát hiện 4 lớp mục tiêu: car, bus, truck, motor.

Cách chạy:
    streamlit run app.py              # giao diện web
    python app.py --selftest          # tự kiểm tra nhanh (không cần streamlit server)

Tên lớp lấy từ bảng cố định `CLASS_NAMES` theo id; kết quả được tự vẽ bằng OpenCV.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Hằng số
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "best_final.pt"

# Bảng tên lớp ĐÚNG theo data.yaml của bộ dữ liệu (không lấy từ model.names).
CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "rider",
    2: "car",
    3: "bus",
    4: "truck",
    5: "bike",
    6: "motor",
    7: "traffic light",
    8: "traffic sign",
    9: "train",
}
CLASS_NAMES_VI: dict[int, str] = {
    0: "người đi bộ",
    1: "người đi xe",
    2: "ô tô",
    3: "xe buýt",
    4: "xe tải",
    5: "xe đạp",
    6: "xe máy",
    7: "đèn giao thông",
    8: "biển báo",
    9: "tàu hỏa",
}
NUM_CLASSES = len(CLASS_NAMES)
TARGET_CLASS_IDS: list[int] = [2, 3, 4, 6]  # car, bus, truck, motor — chỉ phát hiện 4 lớp này

# Màu theo lớp (BGR cho OpenCV).
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (60, 180, 255),
    1: (255, 128, 0),
    2: (0, 200, 0),
    3: (0, 140, 255),
    4: (0, 0, 230),
    5: (180, 60, 160),
    6: (255, 200, 0),
    7: (0, 255, 255),
    8: (255, 255, 0),
    9: (128, 128, 128),
}
LINE_COLOR = (255, 0, 255)

TRACKER_OPTIONS = ["bytetrack.yaml", "botsort.yaml"]
DEVICE_OPTIONS = ["Auto", "GPU", "CPU"]
IMGSZ_OPTIONS = [320, 384, 416, 480, 512, 576, 640, 704, 768]
REF_POINT_OPTIONS = ["Tâm đáy bbox", "Tâm bbox"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv"]
IMAGE_TYPES = ["jpg", "jpeg", "png"]

TRAIL_LEN = 30  # số điểm quỹ đạo gần nhất được vẽ
TRACK_MAX_AGE = 90  # số khung (đã xử lý) không thấy track thì xóa trạng thái hình học của nó
DIR_AB = "A->B"
DIR_BA = "B->A"

PRESET_GPU = "GPU (chính xác)"
PRESET_CPU = "CPU (nhẹ)"
PRESETS: dict[str, dict] = {
    PRESET_GPU: {
        "imgsz": 640,
        "stride": 1,
        "half": True,
        "conf": 0.25,
        "max_frames": 0,
        "preview_width": 960,
        "preview_every": 5,
    },
    PRESET_CPU: {
        "imgsz": 480,
        "stride": 2,
        "half": False,
        "conf": 0.30,
        "max_frames": 900,
        "preview_width": 640,
        "preview_every": 10,
    },
}

# ---------------------------------------------------------------------------
# Thiết bị
# ---------------------------------------------------------------------------


def cuda_available() -> bool:
    """Kiểm tra CUDA an toàn (một số máy có driver lỗi sẽ ném ngoại lệ)."""
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(choice: str) -> tuple[str, str | None]:
    """Đổi lựa chọn Auto/GPU/CPU thành chuỗi thiết bị cho Ultralytics; trả kèm cảnh báo nếu phải lùi về CPU."""
    has_cuda = cuda_available()
    if choice == "GPU":
        if has_cuda:
            return "cuda:0", None
        return "cpu", "Đã chọn GPU nhưng máy không có CUDA (hoặc PyTorch bản CPU). Ứng dụng tự chuyển sang CPU."
    if choice == "CPU":
        return "cpu", None
    return ("cuda:0" if has_cuda else "cpu"), None


def is_cuda(device: str) -> bool:
    return device.startswith("cuda")


def device_info(device: str) -> dict[str, str]:
    """Thông tin thiết bị và phiên bản thư viện để hiển thị."""
    import ultralytics

    info = {
        "Thiết bị đang dùng": device,
        "PyTorch": torch.__version__,
        "Ultralytics": ultralytics.__version__,
        "Streamlit": st.__version__,
        "OpenCV": cv2.__version__,
        "CUDA khả dụng": "có" if cuda_available() else "không",
    }
    if is_cuda(device):
        try:
            props = torch.cuda.get_device_properties(0)
            info["GPU"] = torch.cuda.get_device_name(0)
            info["VRAM"] = f"{props.total_memory / 1024**3:.1f} GB"
        except Exception as exc:  # không chặn ứng dụng chỉ vì không đọc được thông tin GPU
            info["GPU"] = f"không đọc được ({exc})"
    else:
        info["Số luồng CPU (torch)"] = str(torch.get_num_threads())
    return info


def precision_kwargs(half: bool) -> dict:
    """Tham số độ chính xác cho predict/track, tương thích nhiều phiên bản Ultralytics.

    Từ khoảng Ultralytics 8.4.16x, `half` bị thay bằng `quantize` (16 = FP16, None = FP32); bản cũ hơn
    (ví dụ 8.4.138 dùng khi huấn luyện) chỉ có `half`. FP16 chỉ được bật khi chạy CUDA.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 16 if half else None}
    except Exception:
        pass
    return {"half": bool(half)}


DEFAULT_CPU_THREADS = max(1, torch.get_num_threads())
MAX_CPU_THREADS = max(1, os.cpu_count() or 1)

# ---------------------------------------------------------------------------
# Thư mục tạm (an toàn trên Windows: OpenCV không mở được đường dẫn có ký tự Unicode)
# ---------------------------------------------------------------------------


def _usable_dir(path: Path) -> bool:
    if os.name == "nt" and not str(path).isascii():
        return False
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".probe_{uuid.uuid4().hex}"
        probe.write_bytes(b"ok")
        probe.unlink()
        return True
    except OSError:
        return False


def pick_work_root() -> Path:
    """Chọn thư mục gốc cho file tạm; trên Windows tránh đường dẫn có dấu tiếng Việt."""
    candidates = [Path(tempfile.gettempdir()) / "smart_traffic_tracking"]
    if os.name == "nt":
        candidates.append(Path(os.environ.get("PUBLIC", r"C:\Users\Public")) / "smart_traffic_tracking")
        candidates.append(Path(os.environ.get("SystemDrive", "C:") + "\\") / "smart_traffic_tracking")
    candidates.append(APP_DIR / ".tmp_smart_traffic")
    for candidate in candidates:
        if _usable_dir(candidate):
            return candidate
    return Path(tempfile.mkdtemp(prefix="smart_traffic_"))


def cleanup_stale_sessions(root: Path, max_age_hours: float = 12.0) -> None:
    """Xóa thư mục phiên cũ (best-effort, bỏ qua file đang bị khóa trên Windows)."""
    now = time.time()
    for child in root.glob("session_*"):
        try:
            if child.is_dir() and now - child.stat().st_mtime > max_age_hours * 3600:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            pass


def safe_unlink(path: Path | str | None) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Nạp mô hình
# ---------------------------------------------------------------------------


def load_yolo(model_path: str, device: str):
    """Nạp YOLO và chuyển sang thiết bị chỉ định (không dùng cache — dùng cho selftest)."""
    from ultralytics import YOLO

    model = YOLO(model_path, task="detect")
    model.to(device)
    return model


def check_model_classes(model) -> str | None:
    """Trả về thông báo lỗi nếu mô hình không có đúng 10 lớp, ngược lại None."""
    n = len(model.names)
    if n != NUM_CLASSES:
        return f"Mô hình có {n} lớp, nhưng ứng dụng yêu cầu đúng {NUM_CLASSES} lớp (BDD100K)."
    return None


@dataclass
class ModelBundle:
    model: object
    lock: threading.Lock
    error: str | None
    load_seconds: float


@st.cache_resource(show_spinner="Đang nạp mô hình theo dõi...", max_entries=4)
def get_track_bundle(model_path: str, device: str, mtime: float) -> ModelBundle:
    """Mô hình dùng cho theo dõi video; cache theo (đường dẫn, thiết bị, thời điểm sửa file)."""
    start = time.perf_counter()
    model = load_yolo(model_path, device)
    error = check_model_classes(model)
    return ModelBundle(model, threading.Lock(), error, time.perf_counter() - start)


@st.cache_resource(show_spinner="Đang nạp mô hình phát hiện ảnh...", max_entries=4)
def get_detect_bundle(model_path: str, device: str, mtime: float) -> ModelBundle:
    """Một bản mô hình riêng cho tab Ảnh để không đụng trạng thái tracker của bản dùng cho video."""
    start = time.perf_counter()
    model = load_yolo(model_path, device)
    error = check_model_classes(model)
    return ModelBundle(model, threading.Lock(), error, time.perf_counter() - start)


def reset_tracker(model) -> None:
    """Xóa tracker cũ để video mới bắt đầu với ID từ đầu.

    Ứng dụng luôn gọi `track(..., persist=True)`; xóa thuộc tính `predictor.trackers` khiến callback
    `on_predict_start` của Ultralytics tạo tracker mới (đúng loại tracker đang chọn) ở khung kế tiếp.
    Cách này cho kết quả như nhau ở các phiên bản Ultralytics cũ và mới.
    """
    predictor = getattr(model, "predictor", None)
    if predictor is not None and hasattr(predictor, "trackers"):
        del predictor.trackers
    try:
        from ultralytics.trackers.basetrack import BaseTrack

        BaseTrack.reset_id()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Đếm xe qua vạch
# ---------------------------------------------------------------------------


@dataclass
class TrackGeometry:
    last_side: int = 0  # dấu phía gần nhất khác 0 (-1 = phía A, +1 = phía B)
    last_cross: float = 0.0  # giá trị tích có hướng tương ứng với last_side
    last_point: tuple[float, float] | None = None
    last_seen: int = 0
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))


class LineCrossingCounter:
    """Đếm phương tiện khi điểm tham chiếu đổi phía so với đoạn thẳng P1-P2.

    - Phía được xác định bằng dấu của tích có hướng (P2 - P1) x (P - P1) trong hệ tọa độ ảnh.
      Phía A (dấu âm) nằm bên trái khi nhìn từ P1 sang P2 trên màn hình; với vạch ngang mặc định
      (P1 bên trái, P2 bên phải) thì A ở phía trên, B ở phía dưới.
    - Chỉ tính khi đoạn di chuyển giữa hai lần quan sát liên tiếp của cùng ID cắt đúng đoạn P1-P2
      (không tính phần kéo dài của đường thẳng).
    - Mỗi ID chỉ được tính tối đa một lần cho mỗi chiều; track có số lần xuất hiện < min_hits bị bỏ qua.
    - Lớp của một ID là lớp xuất hiện nhiều nhất trong lịch sử của ID đó (biểu quyết đa số).
    """

    def __init__(self, p1: tuple[float, float], p2: tuple[float, float], min_hits: int = 3):
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.min_hits = max(1, int(min_hits))
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        self.length_sq = dx * dx + dy * dy
        self.eps = 1e-6 * max(1.0, self.length_sq)
        self.geometry: dict[int, TrackGeometry] = {}
        self.hits: Counter = Counter()
        self.class_votes: dict[int, Counter] = {}
        self.counted: set[tuple[int, str]] = set()
        self.counts: dict[int, Counter] = {i: Counter() for i in CLASS_NAMES}
        self.events: list[dict] = []

    def cross_value(self, point: tuple[float, float]) -> float:
        (x1, y1), (x2, y2) = self.p1, self.p2
        return (x2 - x1) * (point[1] - y1) - (y2 - y1) * (point[0] - x1)

    def side_of(self, cross: float) -> int:
        if abs(cross) <= self.eps:
            return 0
        return 1 if cross > 0 else -1

    def majority_class(self, track_id: int) -> int:
        votes = self.class_votes.get(track_id)
        return votes.most_common(1)[0][0] if votes else -1

    def _hits_segment(self, a: tuple[float, float], b: tuple[float, float], cross_a: float, cross_b: float) -> bool:
        """Giao điểm của đoạn a-b với đường thẳng P1P2 có nằm trong đoạn P1-P2 không."""
        if self.length_sq <= 0:
            return False
        t = cross_a / (cross_a - cross_b)
        ix = a[0] + t * (b[0] - a[0])
        iy = a[1] + t * (b[1] - a[1])
        u = ((ix - self.p1[0]) * (self.p2[0] - self.p1[0]) + (iy - self.p1[1]) * (self.p2[1] - self.p1[1])) / self.length_sq
        return 0.0 <= u <= 1.0

    def update(self, track_id: int, class_id: int, point: tuple[float, float], frame_idx: int, time_s: float, step: int) -> dict | None:
        """Cập nhật một quan sát; trả về sự kiện qua vạch (dict) hoặc None."""
        geo = self.geometry.get(track_id)
        if geo is None:
            geo = self.geometry[track_id] = TrackGeometry()
        self.hits[track_id] += 1
        self.class_votes.setdefault(track_id, Counter())[class_id] += 1
        geo.last_seen = step
        geo.trail.append((int(point[0]), int(point[1])))

        cross = self.cross_value(point)
        side = self.side_of(cross)
        event = None
        if side == 0:
            return None
        if (
            geo.last_side != 0
            and side != geo.last_side
            and self.hits[track_id] >= self.min_hits
            and geo.last_point is not None
            and self._hits_segment(geo.last_point, point, geo.last_cross, cross)
        ):
            direction = DIR_AB if geo.last_side < 0 else DIR_BA
            key = (track_id, direction)
            if key not in self.counted:
                self.counted.add(key)
                cls = self.majority_class(track_id)
                self.counts.setdefault(cls, Counter())[direction] += 1
                event = {
                    "frame": int(frame_idx),
                    "second": round(float(time_s), 3),
                    "track_id": int(track_id),
                    "class_id": int(cls),
                    "class_name": CLASS_NAMES.get(cls, str(cls)),
                    "direction": direction,
                }
                self.events.append(event)
        geo.last_side = side
        geo.last_cross = cross
        geo.last_point = (float(point[0]), float(point[1]))
        return event

    def prune(self, step: int, max_age: int = TRACK_MAX_AGE) -> None:
        stale = [tid for tid, geo in self.geometry.items() if step - geo.last_seen > max_age]
        for tid in stale:
            del self.geometry[tid]

    def total(self, direction: str | None = None) -> int:
        if direction is None:
            return len(self.events)
        return sum(1 for e in self.events if e["direction"] == direction)

    def summary_dataframe(self, class_ids: list[int] | None = None) -> pd.DataFrame:
        """Bảng tổng hợp theo lớp: số lượt qua vạch mỗi chiều và số ID đã xuất hiện (đủ min_hits)."""
        unique_ids: Counter = Counter()
        for tid, n in self.hits.items():
            if n >= self.min_hits:
                unique_ids[self.majority_class(tid)] += 1
        ids = class_ids if class_ids else list(CLASS_NAMES)
        rows = []
        for cid in ids:
            ab = self.counts.get(cid, Counter())[DIR_AB]
            ba = self.counts.get(cid, Counter())[DIR_BA]
            rows.append(
                {
                    "class_id": cid,
                    "class_name": CLASS_NAMES[cid],
                    "ten_lop": CLASS_NAMES_VI[cid],
                    DIR_AB: ab,
                    DIR_BA: ba,
                    "total_crossings": ab + ba,
                    "unique_track_ids": unique_ids[cid],
                }
            )
        return pd.DataFrame(rows)

    def events_dataframe(self) -> pd.DataFrame:
        cols = ["frame", "second", "track_id", "class_id", "class_name", "direction"]
        return pd.DataFrame(self.events, columns=cols)


# ---------------------------------------------------------------------------
# Vẽ kết quả bằng OpenCV
# ---------------------------------------------------------------------------


def percent_to_pixel(line_pct: tuple[float, float, float, float], width: int, height: int):
    x1, y1, x2, y2 = line_pct
    p1 = (x1 / 100.0 * (width - 1), y1 / 100.0 * (height - 1))
    p2 = (x2 / 100.0 * (width - 1), y2 / 100.0 * (height - 1))
    return p1, p2


def reference_point(box: np.ndarray, mode: str) -> tuple[float, float]:
    x1, y1, x2, y2 = box[:4]
    if mode == REF_POINT_OPTIONS[1]:
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return ((x1 + x2) / 2.0, float(y2))


def _put_label(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int], scale: float, thickness: int) -> None:
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    y = max(y, th + baseline + 2)
    cv2.rectangle(img, (x, y - th - baseline - 2), (x + tw + 4, y + 1), color, -1)
    brightness = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(img, text, (x + 2, y - baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, thickness, cv2.LINE_AA)


def draw_boxes(img: np.ndarray, boxes: np.ndarray, class_ids: np.ndarray, confs: np.ndarray, track_ids: np.ndarray | None) -> None:
    h, w = img.shape[:2]
    scale = max(0.4, min(1.0, w / 1600))
    thickness = 1 if w < 800 else 2
    for i, box in enumerate(boxes):
        cid = int(class_ids[i])
        color = CLASS_COLORS.get(cid, (200, 200, 200))
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness + 1)
        name = CLASS_NAMES.get(cid, str(cid))
        if track_ids is not None:
            label = f"{name} #{int(track_ids[i])} {float(confs[i]):.2f}"
        else:
            label = f"{name} {float(confs[i]):.2f}"
        _put_label(img, label, (x1, y1 - 2), color, scale, thickness)


def draw_trails(img: np.ndarray, counter: LineCrossingCounter, active_ids: set[int]) -> None:
    for tid in active_ids:
        geo = counter.geometry.get(tid)
        if geo is None or len(geo.trail) < 2:
            continue
        color = CLASS_COLORS.get(counter.majority_class(tid), (200, 200, 200))
        pts = np.array(geo.trail, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)


def draw_counting_line(img: np.ndarray, p1: tuple[float, float], p2: tuple[float, float]) -> None:
    h, w = img.shape[:2]
    thickness = max(2, w // 400)
    a = (int(round(p1[0])), int(round(p1[1])))
    b = (int(round(p2[0])), int(round(p2[1])))
    cv2.line(img, a, b, LINE_COLOR, thickness, cv2.LINE_AA)
    cv2.circle(img, a, thickness + 3, LINE_COLOR, -1)
    cv2.circle(img, b, thickness + 3, LINE_COLOR, -1)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return
    nx, ny = dy / norm, -dx / norm  # pháp tuyến hướng về phía A
    mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
    offset = max(20, w // 40)
    scale = max(0.6, min(1.2, w / 1000))
    for text, sign in (("A", 1), ("B", -1)):
        org = (int(mx + sign * nx * offset) - 10, int(my + sign * ny * offset) + 10)
        org = (min(max(0, org[0]), w - 30), min(max(25, org[1]), h - 5))
        _put_label(img, text, org, LINE_COLOR, scale, 2)


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    h, w = img.shape[:2]
    scale = max(0.5, min(0.9, w / 1400))
    line_h = int(32 * scale) + 6
    box_w = int(max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] for t in lines)) + 20
    box_h = line_h * len(lines) + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, dst=img)
    for i, text in enumerate(lines):
        y = 8 + line_h * (i + 1)
        cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def resize_to_width(img: np.ndarray, max_width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    new_h = int(round(h * max_width / w))
    return cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def extract_boxes(result) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Lấy (xyxy, class_id, conf, track_id) từ Results; chỉ dùng id lớp, không dùng result.names."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        empty = np.zeros((0,), dtype=np.float32)
        return np.zeros((0, 4), dtype=np.float32), empty.astype(int), empty, None
    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
    return xyxy, cls, conf, ids


# ---------------------------------------------------------------------------
# Video: đọc metadata, ghi và chuyển mã
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> dict | None:
    """Đọc fps, kích thước, số khung và khung đầu tiên; trả None nếu video hỏng/không có khung."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            return None
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps_guessed = False
        if not fps or fps != fps or fps <= 0 or fps > 240:
            fps, fps_guessed = 30.0, True
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        ok, first = cap.read()
        if not ok or first is None:
            return None
        h, w = first.shape[:2]
        return {
            "fps": float(fps),
            "fps_guessed": fps_guessed,
            "width": int(w),
            "height": int(h),
            "frame_count": max(0, frame_count),
            "first_frame": first,
        }
    finally:
        cap.release()


def open_video_writer(base_path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter | None, Path | None]:
    """Mở VideoWriter: thử mp4v (.mp4), nếu không được thì MJPG (.avi)."""
    for fourcc, ext in (("mp4v", ".mp4"), ("MJPG", ".avi")):
        path = base_path.with_suffix(ext)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), max(1.0, fps), size)
        if writer.isOpened():
            return writer, path
        writer.release()
        safe_unlink(path)
    return None, None


def transcode_to_h264(src: Path, dst: Path, timeout_s: int = 3600) -> tuple[bool, str]:
    """Chuyển video sang H.264 (yuv420p, faststart) bằng ffmpeg đi kèm imageio-ffmpeg."""
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        return False, f"Không tìm thấy ffmpeg (imageio-ffmpeg): {exc}"
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
        str(dst),
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired:
        return False, "ffmpeg chạy quá thời gian cho phép."
    except OSError as exc:
        return False, f"Không chạy được ffmpeg: {exc}"
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        safe_unlink(dst)
        return False, (proc.stderr or "ffmpeg trả về lỗi không rõ").strip()[-800:]
    return True, ""


# ---------------------------------------------------------------------------
# Tự kiểm tra (python app.py --selftest)
# ---------------------------------------------------------------------------


def _selftest_counter() -> list[str]:
    """Kiểm tra logic đếm vạch trên dữ liệu giả; trả về danh sách lỗi."""
    errors = []
    c = LineCrossingCounter((0, 100), (200, 100), min_hits=3)
    for step, y in enumerate([60, 80, 95, 110, 130]):  # ID 1 đi xuống: A -> B
        c.update(1, 2, (100, y), step, step / 30, step)
    for step, y in enumerate([130, 110, 90, 70], start=5):  # ID 1 quay lên: B -> A
        c.update(1, 2, (100, y), step, step / 30, step)
    for step, y in enumerate([70, 120, 70, 120], start=9):  # qua lại nhiều lần: không tính thêm
        c.update(1, 2, (100, y), step, step / 30, step)
    for step, y in enumerate([90, 110]):  # ID 2 quá ngắn (2 lần < min_hits=3): bỏ qua
        c.update(2, 3, (50, y), step, step / 30, step)
    for step, y in enumerate([60, 80, 95, 110]):  # ID 3 cắt phần kéo dài ngoài đoạn vạch: bỏ qua
        c.update(3, 4, (300, y), step, step / 30, step)
    for step, (cls, y) in enumerate([(6, 60), (5, 80), (6, 95), (6, 110)]):  # ID 4: lớp theo đa số (motor)
        c.update(4, cls, (150, y), step, step / 30, step)
    if c.total(DIR_AB) != 2 or c.total(DIR_BA) != 1:
        errors.append(f"đếm sai: A->B={c.total(DIR_AB)} (kỳ vọng 2), B->A={c.total(DIR_BA)} (kỳ vọng 1)")
    if c.counts[6][DIR_AB] != 1:
        errors.append("biểu quyết lớp đa số sai (ID 4 phải là motor)")
    df = c.summary_dataframe()
    if int(df["total_crossings"].sum()) != 3:
        errors.append("bảng tổng hợp sai")
    return errors


def run_selftest(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Tự kiểm tra Smart Traffic Tracking")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="đường dẫn file .pt")
    parser.add_argument("--device", default="Auto", choices=DEVICE_OPTIONS)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args(argv)

    import ultralytics

    failures: list[str] = []
    print("=== Smart Traffic Tracking — selftest ===")
    print(f"Python {sys.version.split()[0]} | torch {torch.__version__} | ultralytics {ultralytics.__version__} "
          f"| streamlit {st.__version__} | opencv {cv2.__version__}")
    device, warn = resolve_device(args.device)
    if warn:
        print(f"[CẢNH BÁO] {warn}")
    print(f"Thiết bị: {device}" + (f" ({torch.cuda.get_device_name(0)})" if is_cuda(device) else f" ({torch.get_num_threads()} luồng)"))

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"[LỖI] Không tìm thấy mô hình: {model_path}")
        return 1

    t0 = time.perf_counter()
    model = load_yolo(str(model_path), device)
    print(f"Nạp mô hình: {time.perf_counter() - t0:.2f} s ({model_path.name}, {model_path.stat().st_size / 1e6:.1f} MB)")

    error = check_model_classes(model)
    if error:
        failures.append(error)
    print(f"Số lớp trong mô hình: {len(model.names)}; ứng dụng chỉ phát hiện 4 lớp mục tiêu:")
    for i in TARGET_CLASS_IDS:
        print(f"  {i}: {CLASS_NAMES[i]:<6} ({CLASS_NAMES_VI[i]})")
    if len(CLASS_NAMES) != 10 or sorted(CLASS_NAMES) != list(range(10)):
        failures.append("CLASS_NAMES phải có đúng 10 id 0..9")

    half = is_cuda(device)
    dummy = np.random.default_rng(0).integers(0, 255, size=(720, 1280, 3), dtype=np.uint8)
    model.predict(dummy, imgsz=args.imgsz, device=device, classes=TARGET_CLASS_IDS, verbose=False,
                  **precision_kwargs(half))  # khởi động
    runs = 5
    t0 = time.perf_counter()
    for _ in range(runs):
        result = model.predict(dummy, imgsz=args.imgsz, device=device, conf=0.25, classes=TARGET_CLASS_IDS,
                               verbose=False,
                                   **precision_kwargs(half))[0]
    ms = (time.perf_counter() - t0) * 1000 / runs
    print(f"Suy luận ảnh giả 1280x720, imgsz={args.imgsz}, half={half}: {ms:.1f} ms/ảnh "
          f"(~{1000 / ms:.1f} FPS, gồm tiền/hậu xử lý; ảnh nhiễu nên số box = {len(result.boxes)})")

    try:
        reset_tracker(model)
        for _ in range(3):
            model.track(dummy, persist=True, tracker="bytetrack.yaml", imgsz=args.imgsz, device=device,
                        conf=0.25, classes=TARGET_CLASS_IDS, verbose=False, **precision_kwargs(half))
        print("Theo dõi (bytetrack, 3 khung giả): OK")
    except Exception as exc:
        failures.append(f"track lỗi: {exc}")

    counter_errors = _selftest_counter()
    failures.extend(counter_errors)
    print("Logic đếm vạch: " + ("OK" if not counter_errors else "; ".join(counter_errors)))

    tmp_dir = Path(tempfile.mkdtemp(prefix="smart_traffic_selftest_"))
    try:
        frame = dummy.copy()
        draw_boxes(frame, np.array([[100, 100, 300, 250]]), np.array([2]), np.array([0.9]), np.array([1]))
        draw_counting_line(frame, *percent_to_pixel((0, 60, 100, 60), 1280, 720))
        draw_hud(frame, ["A->B: 0  B->A: 0", "FPS: 0.0"])
        writer, raw_path = open_video_writer(tmp_dir / "raw", 10.0, (1280, 720))
        if writer is None:
            failures.append("cv2.VideoWriter không mở được (mp4v và MJPG)")
        else:
            try:
                for _ in range(10):
                    writer.write(frame)
            finally:
                writer.release()
            ok, msg = transcode_to_h264(raw_path, tmp_dir / "out_h264.mp4")
            print(f"Ghi video ({raw_path.suffix}) + chuyển mã H.264: " + ("OK" if ok else f"LỖI ({msg})"))
            if not ok:
                failures.append(f"chuyển mã H.264 lỗi: {msg}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if failures:
        print("KẾT QUẢ: THẤT BẠI")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("KẾT QUẢ: ĐẠT")
    return 0


# ---------------------------------------------------------------------------
# Giao diện Streamlit
# ---------------------------------------------------------------------------


# Giá trị mặc định của các widget (ngoài các khóa do preset quản lý).
WIDGET_DEFAULTS: dict[str, object] = {
    "device_choice": "Auto",
    "tracker_name": TRACKER_OPTIONS[0],
    "iou": 0.7,
    "line_x1": 0,
    "line_y1": 60,
    "line_x2": 100,
    "line_y2": 60,
    "ref_point": REF_POINT_OPTIONS[0],
    "min_hits": 3,
    "draw_trails": True,
    "cpu_threads": DEFAULT_CPU_THREADS,
    "model_path_text": str(DEFAULT_MODEL_PATH),
}
PERSISTENT_WIDGET_KEYS = ["preset", *PRESETS[PRESET_GPU].keys(), *WIDGET_DEFAULTS.keys()]


def init_session_state(default_device: str) -> None:
    ss = st.session_state
    if "initialized" not in ss:
        preset = PRESET_GPU if is_cuda(default_device) else PRESET_CPU
        ss.initialized = True
        ss.processing = False
        ss.stop_requested = False
        ss.result = None
        ss.run_state = None
        ss.video_info = None
        ss.video_file_id = None
        ss.video_path = None
        ss.uploaded_model = None  # (file_id, đường dẫn đã lưu)
        for key, value in WIDGET_DEFAULTS.items():
            ss[key] = list(value) if isinstance(value, list) else value
        ss.preset = preset
        apply_preset(preset)
        root = pick_work_root()
        cleanup_stale_sessions(root)
        ss.work_dir = str(root / f"session_{uuid.uuid4().hex[:12]}")
    Path(ss.work_dir).mkdir(parents=True, exist_ok=True)
    # Streamlit xóa giá trị của widget không được vẽ trong một lượt chạy (ví dụ thanh trượt vạch đếm khi chưa có
    # video, hoặc số luồng CPU khi đang dùng GPU). Gán lại vào session_state để giữ giá trị giữa các lượt chạy.
    for key in PERSISTENT_WIDGET_KEYS:
        if key in ss:
            ss[key] = ss[key]


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def on_preset_change() -> None:
    apply_preset(st.session_state.preset)


def on_device_change() -> None:
    device, _ = resolve_device(st.session_state.device_choice)
    preset = PRESET_GPU if is_cuda(device) else PRESET_CPU
    st.session_state.preset = preset
    apply_preset(preset)


def request_start() -> None:
    st.session_state.processing = True
    st.session_state.stop_requested = False
    clear_result()


def request_stop() -> None:
    st.session_state.stop_requested = True


def clear_result() -> None:
    old = st.session_state.get("result")
    if old:
        safe_unlink(old.get("raw_path"))
        safe_unlink(old.get("h264_path"))
    st.session_state.result = None


def sidebar_model_section() -> Path | None:
    """Chọn file mô hình: đường dẫn hoặc tải lên. Trả về đường dẫn hợp lệ hoặc None."""
    ss = st.session_state
    disabled = ss.processing
    st.subheader("Mô hình")
    st.text_input("Đường dẫn file .pt", key="model_path_text", disabled=disabled,
                  help="Mặc định: best_final.pt nằm cùng thư mục với app.py")
    uploaded = st.file_uploader("Hoặc tải lên file .pt khác", type=["pt"], disabled=disabled,
                                help="Chỉ tải file mô hình từ nguồn tin cậy (file .pt có thể chứa mã thực thi).")
    if uploaded is not None:
        cached = ss.uploaded_model
        if cached is None or cached[0] != uploaded.file_id or not Path(cached[1]).is_file():
            data = uploaded.getvalue()
            digest = hashlib.sha1(data).hexdigest()[:16]
            model_dir = Path(ss.work_dir).parent / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            target = model_dir / f"model_{digest}.pt"
            if not target.is_file():
                target.write_bytes(data)
            ss.uploaded_model = (uploaded.file_id, str(target))
        st.caption(f"Đang dùng file tải lên: {uploaded.name}")
        return Path(ss.uploaded_model[1])

    path = Path(ss.model_path_text.strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    if not path.is_file():
        st.error(
            f"Không tìm thấy mô hình tại `{path}`. Hãy đặt `best_final.pt` cùng thư mục với `app.py`, "
            "sửa đường dẫn ở trên hoặc tải file .pt lên."
        )
        return None
    if path.suffix.lower() != ".pt":
        st.warning("File không có đuôi .pt; ứng dụng vẫn thử nạp.")
    return path


def render_sidebar() -> tuple[Path | None, str]:
    ss = st.session_state
    disabled = ss.processing
    with st.sidebar:
        st.header("Cấu hình")
        model_path = sidebar_model_section()

        st.subheader("Thiết bị")
        st.radio("Chọn thiết bị", DEVICE_OPTIONS, key="device_choice", horizontal=True,
                 on_change=on_device_change, disabled=disabled,
                 help="Auto = GPU nếu có CUDA, ngược lại CPU.")
        device, warn = resolve_device(ss.device_choice)
        if warn:
            st.warning(warn)
        info = device_info(device)
        if is_cuda(device):
            st.caption(f"Đang dùng **{device}** — {info.get('GPU', '?')}, VRAM {info.get('VRAM', '?')}")
        else:
            st.caption(f"Đang dùng **CPU** — torch {info['PyTorch']}")
        if not is_cuda(device):
            st.warning("Đang chạy trên CPU: tốc độ chậm hơn GPU nhiều. Nên dùng cấu hình \"CPU (nhẹ)\" "
                       "và video ngắn. Nhóm chưa có số liệu FPS đo trên CPU.")
            st.slider("Số luồng CPU cho PyTorch", 1, MAX_CPU_THREADS, key="cpu_threads", disabled=disabled)
        with st.expander("Thông tin thiết bị và phiên bản", expanded=False):
            for k, v in info.items():
                st.markdown(f"- **{k}:** {v}")

        st.subheader("Cấu hình suy luận")
        st.selectbox("Cấu hình có sẵn (preset)", list(PRESETS), key="preset", on_change=on_preset_change,
                     disabled=disabled, help="Chọn preset rồi có thể chỉnh tay từng tham số bên dưới.")
        st.select_slider("Kích thước ảnh vào (imgsz)", IMGSZ_OPTIONS, key="imgsz", disabled=disabled)
        if ss.imgsz < 640:
            st.caption("⚠️ Mô hình được huấn luyện ở 640. Giảm imgsz làm giảm độ chính xác, nhất là vật thể nhỏ "
                       "(đèn giao thông, xe máy, người đi xe).")
        st.slider("Bước nhảy khung hình (stride)", 1, 5, key="stride", disabled=disabled,
                  help="1 = xử lý mọi khung; 2 = xử lý 1 khung bỏ 1 khung...")
        if ss.stride > 1:
            st.caption("⚠️ Bỏ khung làm vật thể dịch chuyển xa hơn giữa hai lần quan sát: tracker dễ mất/đổi ID "
                       "và vật đi nhanh có thể bị đếm sót. Video xuất ra chỉ gồm các khung đã xử lý.")
        st.toggle("FP16 (half) — chỉ dùng trên CUDA", key="half", disabled=disabled or not is_cuda(device))
        st.slider("Ngưỡng tin cậy (conf)", 0.05, 0.95, step=0.05, key="conf", disabled=disabled)
        st.slider("Ngưỡng IoU cho NMS", 0.3, 0.9, step=0.05, key="iou", disabled=disabled)
        st.number_input("Số khung tối đa cần xử lý (0 = không giới hạn)", min_value=0, step=100,
                        key="max_frames", disabled=disabled)
        st.select_slider("Chiều rộng ảnh xem trước (px)", [480, 640, 800, 960], key="preview_width", disabled=disabled)
        st.slider("Cập nhật xem trước mỗi k khung", 1, 30, key="preview_every", disabled=disabled)

        st.subheader("Theo dõi")
        st.selectbox("Bộ theo dõi", TRACKER_OPTIONS, key="tracker_name", disabled=disabled,
                     help="ByteTrack nhanh hơn; BoT-SORT có bù chuyển động camera nên chậm hơn, nhất là trên CPU.")

        st.subheader("Lớp phát hiện")
        st.markdown("\n".join(f"- {CLASS_NAMES[i]} ({CLASS_NAMES_VI[i]})" for i in TARGET_CLASS_IDS))
    return model_path, device


def handle_video_upload(uploaded) -> None:
    """Lưu video tải lên vào thư mục tạm của phiên và đọc metadata (chỉ làm lại khi đổi file)."""
    ss = st.session_state
    if uploaded is None:
        return
    if ss.video_file_id == uploaded.file_id and ss.video_path and Path(ss.video_path).is_file():
        return
    safe_unlink(ss.video_path)
    clear_result()
    ext = Path(uploaded.name).suffix.lower()
    if ext.lstrip(".") not in VIDEO_TYPES:
        ext = ".mp4"
    target = Path(ss.work_dir) / f"input_{uuid.uuid4().hex[:8]}{ext}"
    uploaded.seek(0)
    with open(target, "wb") as f:
        shutil.copyfileobj(uploaded, f, length=16 * 1024 * 1024)
    ss.video_file_id = uploaded.file_id
    ss.video_path = str(target)
    ss.video_info = probe_video(target)


@dataclass
class RunConfig:
    video_path: Path
    work_dir: Path
    device: str
    half: bool
    imgsz: int
    conf: float
    iou: float
    stride: int
    max_frames: int
    classes: list[int]
    tracker: str
    line_pct: tuple[float, float, float, float]
    ref_point: str
    min_hits: int
    draw_trails: bool
    preview_width: int
    preview_every: int
    cpu_threads: int


@dataclass
class RunState:
    counter: LineCrossingCounter | None = None
    raw_path: Path | None = None
    frames_read: int = 0
    processed: int = 0
    elapsed: float = 0.0
    model_ms_sum: float = 0.0
    status: str = "running"
    error: str | None = None


def build_run_config(device: str) -> RunConfig:
    ss = st.session_state
    return RunConfig(
        video_path=Path(ss.video_path),
        work_dir=Path(ss.work_dir),
        device=device,
        half=bool(ss.half) and is_cuda(device),
        imgsz=int(ss.imgsz),
        conf=float(ss.conf),
        iou=float(ss.iou),
        stride=int(ss.stride),
        max_frames=int(ss.max_frames),
        classes=list(TARGET_CLASS_IDS),
        tracker=ss.tracker_name,
        line_pct=(float(ss.line_x1), float(ss.line_y1), float(ss.line_x2), float(ss.line_y2)),
        ref_point=ss.ref_point,
        min_hits=int(ss.min_hits),
        draw_trails=bool(ss.draw_trails),
        preview_width=int(ss.preview_width),
        preview_every=int(ss.preview_every),
        cpu_threads=int(ss.cpu_threads),
    )


def process_video(cfg: RunConfig, bundle: ModelBundle, info: dict) -> None:
    """Vòng lặp xử lý video. Kết quả (kể cả khi bị dừng giữa chừng) được lưu vào session_state.result."""
    ss = st.session_state
    model = bundle.model
    fps = info["fps"]
    width, height = info["width"], info["height"]
    p1, p2 = percent_to_pixel(cfg.line_pct, width, height)
    run = RunState(counter=LineCrossingCounter(p1, p2, cfg.min_hits))
    ss.run_state = run

    total_frames = info["frame_count"]
    expected = total_frames // cfg.stride + (1 if total_frames % cfg.stride else 0) if total_frames else 0
    if cfg.max_frames > 0:
        expected = min(expected, cfg.max_frames) if expected else cfg.max_frames

    progress = st.progress(0.0, text="Đang chuẩn bị...")
    m1, m2, m3, m4 = st.columns(4)
    fps_slot, ms_slot, tracks_slot, device_slot = m1.empty(), m2.empty(), m3.empty(), m4.empty()
    preview_slot = st.empty()
    device_slot.metric("Thiết bị", "GPU" if is_cuda(cfg.device) else "CPU",
                       help=f"{cfg.device}, half={cfg.half}, imgsz={cfg.imgsz}")

    if not is_cuda(cfg.device):
        torch.set_num_threads(max(1, cfg.cpu_threads))

    cap = None
    writer = None
    acquired = bundle.lock.acquire(blocking=False)
    if not acquired:
        run.status, run.error = "error", "Mô hình đang được một phiên khác sử dụng. Hãy thử lại sau."
        ss.result = finalize_result(run, cfg, info, transcode=False)
        return
    start = time.perf_counter()
    try:
        cap = cv2.VideoCapture(str(cfg.video_path))
        if not cap.isOpened():
            raise RuntimeError("Không mở được video.")
        writer, run.raw_path = open_video_writer(cfg.work_dir / f"annotated_{uuid.uuid4().hex[:8]}",
                                                 fps / cfg.stride, (width, height))
        if writer is None:
            raise RuntimeError("Không tạo được file video đầu ra (cv2.VideoWriter).")
        reset_tracker(model)
        ema_ms = None
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx = run.frames_read
            run.frames_read += 1
            if frame_idx % cfg.stride != 0:
                continue
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))

            t0 = time.perf_counter()
            result = model.track(
                frame,
                persist=True,
                tracker=cfg.tracker,
                conf=cfg.conf,
                iou=cfg.iou,
                imgsz=cfg.imgsz,
                device=cfg.device,
                classes=cfg.classes,
                verbose=False,
                **precision_kwargs(cfg.half),
            )[0]
            model_ms = (time.perf_counter() - t0) * 1000
            run.model_ms_sum += model_ms
            ema_ms = model_ms if ema_ms is None else 0.9 * ema_ms + 0.1 * model_ms
            step = run.processed
            run.processed += 1

            xyxy, cls, conf, ids = extract_boxes(result)
            active_ids: set[int] = set()
            if ids is not None:
                for j in range(len(ids)):
                    tid = int(ids[j])
                    active_ids.add(tid)
                    run.counter.update(tid, int(cls[j]), reference_point(xyxy[j], cfg.ref_point),
                                       frame_idx, frame_idx / fps, step)
            run.counter.prune(step)

            # Chỉ vẽ box có ID; box chưa được tracker xác nhận vẫn vẽ nhưng không có ID.
            if cfg.draw_trails:
                draw_trails(frame, run.counter, active_ids)
            draw_boxes(frame, xyxy, cls, conf, ids)
            draw_counting_line(frame, p1, p2)
            run.elapsed = time.perf_counter() - start
            proc_fps = run.processed / run.elapsed if run.elapsed > 0 else 0.0
            draw_hud(frame, [
                f"A->B: {run.counter.total(DIR_AB)}   B->A: {run.counter.total(DIR_BA)}",
                f"FPS: {proc_fps:.1f}   Tracks: {len(active_ids)}",
            ])
            writer.write(frame)

            last = expected and run.processed >= expected
            if run.processed % cfg.preview_every == 0 or run.processed == 1 or last:
                preview_slot.image(bgr_to_rgb(resize_to_width(frame, cfg.preview_width)), channels="RGB")
                fps_slot.metric("FPS xử lý", f"{proc_fps:.1f}")
                ms_slot.metric("ms/khung (mô hình)", f"{ema_ms:.1f}")
                tracks_slot.metric("Đang theo dõi", len(active_ids))
                if expected:
                    progress.progress(min(1.0, run.processed / expected),
                                      text=f"Đã xử lý {run.processed}/{expected} khung "
                                           f"(đọc {run.frames_read}/{total_frames} khung gốc)")
                else:
                    progress.progress(0.0, text=f"Đã xử lý {run.processed} khung (không rõ tổng số khung)")
            if cfg.max_frames > 0 and run.processed >= cfg.max_frames:
                break
        run.status = "done"
    except Exception as exc:  # lỗi thật (không phải dừng/rerun của Streamlit)
        run.status, run.error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        # Khi người dùng bấm Dừng, Streamlit ngắt lượt chạy bằng ngoại lệ điều khiển; khối này vẫn chạy.
        if run.status == "running":
            run.status = "stopped"
        run.elapsed = time.perf_counter() - start
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        bundle.lock.release()
        ss.processing = False
        ss.run_state = None
        ss.result = finalize_result(run, cfg, info, transcode=False)

    progress.progress(1.0, text="Hoàn tất xử lý. Đang chuyển mã video...")
    ensure_transcoded(ss.result)


def finalize_result(run: RunState, cfg: RunConfig, info: dict, transcode: bool) -> dict:
    """Gói kết quả thành dict lưu trong session_state (không gọi API Streamlit để an toàn khi bị ngắt)."""
    counter = run.counter
    summary = counter.summary_dataframe(cfg.classes) if counter else pd.DataFrame()
    events = counter.events_dataframe() if counter else pd.DataFrame()
    raw_ok = run.raw_path is not None and run.raw_path.is_file() and run.processed > 0
    if run.raw_path is not None and not raw_ok:
        safe_unlink(run.raw_path)
    return {
        "status": run.status,
        "error": run.error,
        "raw_path": str(run.raw_path) if raw_ok else None,
        "h264_path": None,
        "transcode_done": False,
        "transcode_error": None,
        "summary": summary,
        "events": events,
        "frames_read": run.frames_read,
        "processed": run.processed,
        "elapsed": run.elapsed,
        "avg_model_ms": run.model_ms_sum / run.processed if run.processed else 0.0,
        "video_fps": info["fps"],
        "settings": {
            "Thiết bị": cfg.device,
            "FP16": cfg.half,
            "imgsz": cfg.imgsz,
            "conf": cfg.conf,
            "iou": cfg.iou,
            "stride": cfg.stride,
            "Số khung tối đa": cfg.max_frames or "không giới hạn",
            "Tracker": cfg.tracker,
            "Lớp": ", ".join(CLASS_NAMES[c] for c in cfg.classes),
            "Vạch (%)": cfg.line_pct,
            "Điểm tham chiếu": cfg.ref_point,
            "Số lần xuất hiện tối thiểu": cfg.min_hits,
        },
    }


def ensure_transcoded(result: dict | None) -> None:
    """Chuyển mã sang H.264 một lần (kể cả khi kết quả bị dừng giữa chừng)."""
    if not result or result["transcode_done"] or not result["raw_path"]:
        return
    raw = Path(result["raw_path"])
    dst = raw.with_name(raw.stem + "_h264.mp4")
    with st.spinner("Đang chuyển video sang H.264 để phát trên trình duyệt..."):
        ok, msg = transcode_to_h264(raw, dst)
    result["transcode_done"] = True
    if ok:
        result["h264_path"] = str(dst)
    else:
        result["transcode_error"] = msg


def render_results(result: dict) -> None:
    ss = st.session_state
    status = result["status"]
    if status == "done":
        st.success("Đã xử lý xong video.")
    elif status == "stopped":
        st.warning("Đã dừng theo yêu cầu. Kết quả bên dưới chỉ gồm phần video đã xử lý." if ss.stop_requested
                   else "Lượt xử lý bị ngắt (có thể do thay đổi tùy chọn). Kết quả bên dưới là phần đã xử lý.")
    else:
        st.error(f"Xử lý lỗi: {result['error']}")
    if not result["raw_path"]:
        st.info("Không có khung hình nào được xử lý nên không có video đầu ra.")

    ensure_transcoded(result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Khung đã xử lý", f"{result['processed']}", help=f"Đã đọc {result['frames_read']} khung gốc")
    c2.metric("Thời gian", f"{result['elapsed']:.1f} s")
    avg_fps = result["processed"] / result["elapsed"] if result["elapsed"] > 0 else 0.0
    c3.metric("FPS xử lý trung bình", f"{avg_fps:.1f}", help=f"Mô hình trung bình {result['avg_model_ms']:.1f} ms/khung")
    events = result["events"]
    c4.metric("Lượt qua vạch", f"{len(events)}")

    if result["h264_path"]:
        st.video(result["h264_path"])
        with open(result["h264_path"], "rb") as f:
            st.download_button("Tải video đã chú thích (H.264 .mp4)", f.read(), file_name="smart_traffic_annotated.mp4",
                               mime="video/mp4", on_click="ignore")
    elif result["raw_path"]:
        st.warning("Chuyển mã H.264 thất bại nên trình duyệt có thể không phát được video. "
                   f"Vẫn có thể tải bản gốc. Chi tiết: {result['transcode_error']}")
        raw = Path(result["raw_path"])
        with open(raw, "rb") as f:
            st.download_button(f"Tải video đã chú thích (bản gốc {raw.suffix})", f.read(),
                               file_name=f"smart_traffic_annotated_raw{raw.suffix}",
                               mime="video/mp4" if raw.suffix == ".mp4" else "video/x-msvideo", on_click="ignore")

    st.subheader("Tổng hợp theo lớp")
    summary = result["summary"]
    st.dataframe(summary, hide_index=True)
    st.caption("A->B / B->A: số lượt qua vạch theo chiều; unique_track_ids: số ID khác nhau (đủ số lần xuất hiện "
               "tối thiểu) được gán vào lớp đó, có thể lớn hơn số xe thật nếu tracker đổi ID.")
    d1, d2 = st.columns(2)
    d1.download_button("Tải CSV tổng hợp theo lớp", summary.to_csv(index=False).encode("utf-8-sig"),
                        file_name="class_summary.csv", mime="text/csv", on_click="ignore")
    d2.download_button("Tải CSV sự kiện qua vạch", events.to_csv(index=False).encode("utf-8-sig"),
                       file_name="crossing_events.csv", mime="text/csv", on_click="ignore")
    with st.expander(f"Danh sách sự kiện qua vạch ({len(events)})"):
        st.dataframe(events, hide_index=True)
    with st.expander("Thiết lập đã dùng"):
        st.json({k: (list(v) if isinstance(v, tuple) else v) for k, v in result["settings"].items()})


def render_video_tab(model_path: Path | None, device: str) -> None:
    ss = st.session_state
    st.markdown("Tải video giao thông lên, đặt vạch đếm rồi bấm **Bắt đầu xử lý**.")
    uploaded = st.file_uploader("Video (mp4, avi, mov, mkv)", type=VIDEO_TYPES, disabled=ss.processing, key="video_uploader")
    if uploaded is None and not ss.processing:
        if ss.video_path:
            safe_unlink(ss.video_path)
            clear_result()
            ss.video_path = ss.video_info = ss.video_file_id = None
        st.info("Chưa có video.")
        return
    if uploaded is not None and not ss.processing:
        with st.spinner("Đang lưu video và đọc thông tin..."):
            handle_video_upload(uploaded)

    info = ss.video_info
    if info is None:
        st.error("Không đọc được video (file hỏng, không có khung hình hoặc codec không được OpenCV hỗ trợ).")
        return
    duration = info["frame_count"] / info["fps"] if info["frame_count"] else None
    st.caption(
        f"Kích thước {info['width']}x{info['height']} | FPS {info['fps']:.2f}"
        + (" (không đọc được FPS, giả định 30)" if info["fps_guessed"] else "")
        + f" | {info['frame_count'] or 'không rõ'} khung"
        + (f" | {duration:.1f} s" if duration else "")
    )

    with st.expander("Vạch đếm và quy tắc đếm", expanded=not ss.processing and ss.result is None):
        left, right = st.columns([1, 2])
        with left:
            st.slider("Đầu mút 1 — x (%)", 0, 100, key="line_x1", disabled=ss.processing)
            st.slider("Đầu mút 1 — y (%)", 0, 100, key="line_y1", disabled=ss.processing)
            st.slider("Đầu mút 2 — x (%)", 0, 100, key="line_x2", disabled=ss.processing)
            st.slider("Đầu mút 2 — y (%)", 0, 100, key="line_y2", disabled=ss.processing)
            st.radio("Điểm tham chiếu của xe", REF_POINT_OPTIONS, key="ref_point", disabled=ss.processing)
            st.slider("Số lần xuất hiện tối thiểu của track", 1, 30, key="min_hits", disabled=ss.processing,
                      help="Track xuất hiện ít hơn số khung (đã xử lý) này thì không được đếm.")
            st.toggle("Vẽ vệt quỹ đạo (30 điểm gần nhất)", key="draw_trails", disabled=ss.processing)
        with right:
            preview = info["first_frame"].copy()
            p1, p2 = percent_to_pixel((ss.line_x1, ss.line_y1, ss.line_x2, ss.line_y2), info["width"], info["height"])
            draw_counting_line(preview, p1, p2)
            st.image(bgr_to_rgb(resize_to_width(preview, 960)), caption="Xem trước vạch trên khung hình đầu tiên",
                     width="stretch")
        st.caption("Một xe được tính khi điểm tham chiếu đổi phía so với vạch giữa hai lần quan sát liên tiếp và "
                   "đoạn di chuyển cắt đúng đoạn vạch. Mỗi ID tính tối đa một lần cho mỗi chiều. Phía A nằm bên "
                   "trái khi nhìn từ đầu mút 1 sang đầu mút 2 (vạch ngang mặc định: A ở trên, B ở dưới).")
        if (ss.line_x1, ss.line_y1) == (ss.line_x2, ss.line_y2):
            st.error("Hai đầu mút trùng nhau: vạch không hợp lệ.")

    line_ok = (ss.line_x1, ss.line_y1) != (ss.line_x2, ss.line_y2)
    ready = model_path is not None and line_ok

    b1, b2, _ = st.columns([1, 1, 3])
    b1.button("▶ Bắt đầu xử lý", type="primary", on_click=request_start, disabled=ss.processing or not ready,
              width="stretch")
    b2.button("⏹ Dừng", on_click=request_stop, disabled=not ss.processing, width="stretch")
    if ss.processing:
        st.caption("Đang xử lý — đừng thay đổi tùy chọn khác; bấm **Dừng** để kết thúc sớm (vẫn giữ phần đã xử lý).")

    if ss.processing:
        if not ready:
            ss.processing = False
            st.rerun()
        bundle = get_track_bundle(str(model_path), device, model_path.stat().st_mtime)
        if bundle.error:
            ss.processing = False
            st.error(bundle.error)
            return
        process_video(build_run_config(device), bundle, info)
        st.rerun()  # vẽ lại giao diện với các widget đã mở khóa

    if ss.result is not None:
        render_results(ss.result)


def render_image_tab(model_path: Path | None, device: str) -> None:
    ss = st.session_state
    st.markdown("Phát hiện car, bus, truck, motor trên ảnh tĩnh (không theo dõi). Dùng các thiết lập ở thanh bên.")
    uploaded = st.file_uploader("Ảnh (jpg, png)", type=IMAGE_TYPES, disabled=ss.processing, key="image_uploader")
    if uploaded is None:
        return
    if model_path is None:
        st.error("Chưa có mô hình hợp lệ.")
        return
    if ss.processing:
        st.info("Đang xử lý video, hãy đợi xong rồi dùng tab Ảnh.")
        return
    img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Không đọc được ảnh.")
        return
    bundle = get_detect_bundle(str(model_path), device, model_path.stat().st_mtime)
    if bundle.error:
        st.error(bundle.error)
        return
    if not is_cuda(device):
        torch.set_num_threads(max(1, int(ss.cpu_threads)))
    with bundle.lock:
        t0 = time.perf_counter()
        result = bundle.model.predict(
            img,
            conf=float(ss.conf),
            iou=float(ss.iou),
            imgsz=int(ss.imgsz),
            device=device,
            classes=TARGET_CLASS_IDS,
            verbose=False,
            **precision_kwargs(bool(ss.half) and is_cuda(device)),
        )[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000
    xyxy, cls, conf, _ = extract_boxes(result)
    annotated = img.copy()
    draw_boxes(annotated, xyxy, cls, conf, None)
    st.image(bgr_to_rgb(annotated), caption=f"{len(xyxy)} đối tượng — {elapsed_ms:.0f} ms trên {device}",
             width="stretch")

    rows = [
        {
            "STT": i + 1,
            "class_id": int(cls[i]),
            "class_name": CLASS_NAMES.get(int(cls[i]), str(cls[i])),
            "ten_lop": CLASS_NAMES_VI.get(int(cls[i]), ""),
            "conf": round(float(conf[i]), 3),
            "x1": int(xyxy[i][0]), "y1": int(xyxy[i][1]), "x2": int(xyxy[i][2]), "y2": int(xyxy[i][3]),
        }
        for i in range(len(xyxy))
    ]
    df = pd.DataFrame(rows, columns=["STT", "class_id", "class_name", "ten_lop", "conf", "x1", "y1", "x2", "y2"])
    left, right = st.columns([2, 1])
    with left:
        st.dataframe(df, hide_index=True)
    with right:
        counts = df.groupby("class_name").size().rename("số lượng").reset_index() if len(df) else df
        st.dataframe(counts, hide_index=True)
    ok, png = cv2.imencode(".png", annotated)
    if ok:
        st.download_button("Tải ảnh đã chú thích", png.tobytes(), file_name="annotated.png", mime="image/png",
                           on_click="ignore")


def preload_model(model_path: Path | None, device: str) -> None:
    """Nạp sẵn mô hình (cache) khi mở trang và báo lỗi nếu mô hình không hợp lệ."""
    if model_path is None:
        return
    bundle = get_track_bundle(str(model_path), device, model_path.stat().st_mtime)
    if bundle.error:
        st.error(bundle.error)


def main() -> None:
    st.set_page_config(page_title="Smart Traffic Tracking", page_icon="🚦", layout="wide")
    default_device, _ = resolve_device("Auto")
    init_session_state(default_device)

    st.title("🚦 Smart Traffic Tracking")
    st.caption("Phát hiện và theo dõi car, bus, truck, motor bằng YOLO11m + ByteTrack/BoT-SORT, đếm xe qua vạch.")

    model_path, device = render_sidebar()
    try:
        preload_model(model_path, device)
    except Exception as exc:
        st.error(f"Không nạp được mô hình: {type(exc).__name__}: {exc}")
        model_path = None

    tab_video, tab_image = st.tabs(["Video", "Ảnh"])
    with tab_video:
        render_video_tab(model_path, device)
    with tab_image:
        render_image_tab(model_path, device)


def running_in_streamlit() -> bool:
    try:
        from streamlit import runtime

        return runtime.exists()
    except Exception:
        return False


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(run_selftest(sys.argv[1:]))
    if running_in_streamlit():
        main()
    else:
        print("Hãy chạy giao diện bằng:  streamlit run app.py")
        print("Hoặc tự kiểm tra bằng:     python app.py --selftest [--device CPU] [--model path/to/best_final.pt]")
