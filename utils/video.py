"""Video I/O and annotation helpers."""

from pathlib import Path
from typing import Iterable, Protocol

import cv2
import numpy as np


class AnnotatedObject(Protocol):
    bbox: tuple[float, float, float, float]
    confidence: float
    class_name: str


COLORS = {
    "car": (60, 180, 75),
    "motorcycle": (255, 130, 20),
    "bus": (0, 215, 255),
    "truck": (180, 80, 220),
}


def open_video(source: Path) -> cv2.VideoCapture:
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {source}")
    return capture


def create_writer(capture: cv2.VideoCapture, output: Path) -> cv2.VideoWriter:
    output.parent.mkdir(parents=True, exist_ok=True)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create output video: {output}")
    return writer


def annotate(frame: np.ndarray, objects: Iterable[AnnotatedObject]) -> np.ndarray:
    canvas = frame.copy()
    for obj in objects:
        x1, y1, x2, y2 = (int(value) for value in obj.bbox)
        color = COLORS.get(obj.class_name.lower(), (255, 255, 255))
        track_id = getattr(obj, "track_id", None)
        label = obj.class_name.capitalize()
        if track_id is not None:
            label += f" | ID {track_id}"
        label += f" | {obj.confidence:.2f}"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        top = max(0, y1 - text_height - 8)
        cv2.rectangle(canvas, (x1, top), (x1 + text_width + 6, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1 + 3, max(text_height + 1, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas
