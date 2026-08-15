from types import SimpleNamespace

import numpy as np

from detection import VehicleDetector
from tracking import VehicleTracker
from utils.video import annotate
from utils.video import open_video
from test_detector import FakeBox, FakeModel


class TrackingModel(FakeModel):
    def track(self, **kwargs):
        boxes = [FakeBox(5, 0.91, [2, 3, 18, 19], track_id=12)]
        return [SimpleNamespace(boxes=Boxes(boxes))]


class Boxes:
    def __init__(self, boxes):
        self._boxes = boxes
        self.id = object()
    def __iter__(self): return iter(self._boxes)


def test_tracker_interface_and_annotation(tmp_path):
    tracker = VehicleTracker(VehicleDetector(model=TrackingModel()))
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    tracks = tracker.track(frame)
    assert tracks[0].track_id == 12
    assert tracks[0].class_name == "bus"
    assert annotate(frame, tracks).shape == frame.shape
    output = tmp_path / "outputs" / "tracking"
    output.mkdir(parents=True)
    assert output.is_dir()


def test_open_video_rejects_missing_source(tmp_path):
    try:
        open_video(tmp_path / "missing.mp4")
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("missing source should fail")
