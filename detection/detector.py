"""YOLO-based vehicle detection with a small, testable interface."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


VEHICLE_CLASSES = frozenset({"car", "motorcycle", "bus", "truck"})


@dataclass(frozen=True)
class Detection:
    """One vehicle detection in pixel coordinates."""

    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


class VehicleDetector:
    """Load a pretrained YOLO model and return only supported vehicles."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        device: str | None = None,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is not installed. Run: pip install -r requirements.txt"
                ) from exc
            try:
                model = YOLO(model_path)
            except Exception as exc:
                raise RuntimeError(f"Could not load YOLO model '{model_path}': {exc}") from exc

        self.model = model
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.class_names = self._normalize_names(model.names)
        self.vehicle_class_ids = [
            class_id
            for class_id, name in self.class_names.items()
            if name.lower() in VEHICLE_CLASSES
        ]
        if not self.vehicle_class_ids:
            raise ValueError("Model does not expose any supported vehicle classes")

    @staticmethod
    def _normalize_names(names: Mapping[int, str] | list[str]) -> dict[int, str]:
        if isinstance(names, Mapping):
            return {int(key): str(value) for key, value in names.items()}
        return dict(enumerate(names))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame."""
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        kwargs: dict[str, Any] = {
            "source": frame,
            "conf": self.confidence,
            "classes": self.vehicle_class_ids,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        results = self.model.predict(**kwargs)
        return self.parse_result(results[0]) if results else []

    def parse_result(self, result: Any) -> list[Detection]:
        """Convert an Ultralytics result to stable Python records."""
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = self.class_names.get(class_id, str(class_id)).lower()
            if class_name not in VEHICLE_CLASSES:
                continue
            xyxy = box.xyxy[0].detach().cpu().tolist()
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in xyxy),
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
        return detections
