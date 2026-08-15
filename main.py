"""Unified command-line interface for detection and tracking."""

import argparse
from collections import Counter
from pathlib import Path
import sys
import time

import cv2

from detection import VehicleDetector
from tracking import VehicleTracker
from utils.video import annotate, create_writer, open_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect or track road vehicles in a video with YOLO11 and ByteTrack."
    )
    parser.add_argument("--source", required=True, type=Path, help="input video path")
    parser.add_argument("--mode", required=True, choices=("detect", "track"))
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO weights or model name")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold [0, 1]")
    parser.add_argument("--output", type=Path, help="output .mp4 path")
    parser.add_argument("--device", help="Ultralytics device, e.g. cpu, 0, cuda:0")
    return parser


def default_output(mode: str, source: Path) -> Path:
    return Path("outputs") / ("detection" if mode == "detect" else "tracking") / f"{source.stem}_{mode}.mp4"


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")
    output = args.output or default_output(args.mode, args.source)
    detector = VehicleDetector(args.model, args.conf, args.device)
    tracker = VehicleTracker(detector) if args.mode == "track" else None
    capture = open_video(args.source)
    writer = create_writer(capture, output)

    frame_count = 0
    class_detections: Counter[str] = Counter()
    unique_tracks: dict[int, str] = {}
    best_frame = None
    best_object_count = -1
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            objects = tracker.track(frame) if tracker else detector.detect(frame)
            annotated = annotate(frame, objects)
            writer.write(annotated)
            frame_count += 1
            class_detections.update(obj.class_name for obj in objects)
            for obj in objects:
                track_id = getattr(obj, "track_id", None)
                if track_id is not None:
                    unique_tracks.setdefault(track_id, obj.class_name)
            if len(objects) > best_object_count:
                best_frame = annotated.copy()
                best_object_count = len(objects)
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise ValueError(f"Video contains no readable frames: {args.source}")
    elapsed = time.perf_counter() - started
    evidence_name = "detection_result.jpg" if args.mode == "detect" else "tracking_result.jpg"
    evidence = output.parent / evidence_name
    if best_frame is not None and not cv2.imwrite(str(evidence), best_frame):
        raise RuntimeError(f"Could not save evidence image: {evidence}")
    return {
        "frames": frame_count,
        "fps": frame_count / elapsed if elapsed else 0.0,
        "classes": class_detections,
        "unique_tracks": unique_tracks,
        "output": output,
        "evidence": evidence,
    }


def print_summary(summary: dict[str, object], mode: str) -> None:
    print(f"Processed frames: {summary['frames']}")
    print(f"Average processing FPS: {summary['fps']:.2f}")
    classes = summary["classes"]
    print("Detected classes (frame-level observations):")
    for name in ("car", "motorcycle", "bus", "truck"):
        print(f"  {name}: {classes.get(name, 0)}")
    if mode == "track":
        tracks = summary["unique_tracks"]
        distribution = Counter(tracks.values())
        print(f"Unique track IDs: {len(tracks)}")
        print("Class distribution by track ID:")
        for name in ("car", "motorcycle", "bus", "truck"):
            print(f"  {name}: {distribution.get(name, 0)}")
    print(f"Output video: {summary['output']}")
    print(f"Evidence image: {summary['evidence']}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
