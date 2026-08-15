from types import SimpleNamespace

import numpy as np
import pytest

from detection import VehicleDetector


class Scalar:
    def __init__(self, value): self.value = value
    def item(self): return self.value


class Coordinates:
    def __init__(self, values): self.values = values
    def __getitem__(self, index): return self
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class FakeBox:
    def __init__(self, class_id, confidence, bbox, track_id=None):
        self.cls = Scalar(class_id)
        self.conf = Scalar(confidence)
        self.xyxy = Coordinates(bbox)
        self.id = Scalar(track_id) if track_id is not None else None


class FakeModel:
    names = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    def __init__(self, boxes=None): self.boxes = boxes or []
    def predict(self, **kwargs): return [SimpleNamespace(boxes=self.boxes)]
    def track(self, **kwargs): return [SimpleNamespace(boxes=SimpleNamespace(id=None))]


def test_detector_initializes_and_filters_vehicle_classes():
    detector = VehicleDetector(model=FakeModel(), confidence=0.3)
    assert detector.vehicle_class_ids == [2, 3, 5, 7]


def test_detect_returns_structured_vehicle_only():
    model = FakeModel([FakeBox(0, 0.9, [0, 0, 10, 10]), FakeBox(2, 0.8, [1, 2, 11, 12])])
    detections = VehicleDetector(model=model).detect(np.zeros((20, 20, 3), dtype=np.uint8))
    assert len(detections) == 1
    assert detections[0].class_name == "car"
    assert detections[0].bbox == (1.0, 2.0, 11.0, 12.0)


def test_invalid_frame_and_confidence():
    with pytest.raises(ValueError):
        VehicleDetector(model=FakeModel(), confidence=1.5)
    detector = VehicleDetector(model=FakeModel())
    with pytest.raises(ValueError):
        detector.detect(np.array([]))
