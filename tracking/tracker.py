"""ByteTrack integration through Ultralytics' maintained tracking API."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from detection.detector import VEHICLE_CLASSES, VehicleDetector


@dataclass(frozen=True)
class TrackedVehicle:
    """One tracked vehicle for a video frame."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    track_id: int


class VehicleTracker:
    """Maintain ByteTrack state across successive calls to ``track``."""

    def __init__(self, detector: VehicleDetector) -> None:
        self.detector = detector

    def track(self, frame: np.ndarray) -> list[TrackedVehicle]:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        kwargs: dict[str, Any] = {
            "source": frame,
            "persist": True,
            "tracker": "bytetrack.yaml",
            "conf": self.detector.confidence,
            "classes": self.detector.vehicle_class_ids,
            "verbose": False,
        }
        if self.detector.device:
            kwargs["device"] = self.detector.device
        results = self.detector.model.track(**kwargs)
        return self.parse_result(results[0]) if results else []

    def parse_result(self, result: Any) -> list[TrackedVehicle]:
        tracks: list[TrackedVehicle] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or boxes.id is None:
            return tracks
        for box in boxes:
            if box.id is None:
                continue
            class_id = int(box.cls.item())
            class_name = self.detector.class_names.get(class_id, str(class_id)).lower()
            if class_name not in VEHICLE_CLASSES:
                continue
            tracks.append(
                TrackedVehicle(
                    bbox=tuple(float(value) for value in box.xyxy[0].detach().cpu().tolist()),
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=class_name,
                    track_id=int(box.id.item()),
                )
            )
        return tracks
