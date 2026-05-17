from __future__ import annotations

import json
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    xyxy: np.ndarray
    score: float
    cls: int


def load_homography(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    image_points = np.asarray(cfg["image_points"], dtype=np.float32)
    field_points = np.asarray(cfg["field_points"], dtype=np.float32)
    h, _ = cv2.findHomography(image_points, field_points)
    if h is None:
        raise ValueError("Could not estimate homography from the given points.")
    return h


def image_to_field(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(pts, homography)
    return mapped.reshape(-1, 2)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


class SimpleTracker:
    def __init__(self, iou_threshold: float = 0.25, max_missing: int = 12) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_id = 0
        self.tracks: dict[int, tuple[np.ndarray, int, int]] = {}

    def update(self, detections: list[Detection]) -> list[tuple[int, Detection]]:
        assigned: set[int] = set()
        output: list[tuple[int, Detection]] = []
        for det in detections:
            best_id = None
            best_iou = 0.0
            for track_id, (box, missing, cls) in self.tracks.items():
                if track_id in assigned or cls != det.cls or missing > self.max_missing:
                    continue
                score = iou(box, det.xyxy)
                if score > best_iou:
                    best_iou = score
                    best_id = track_id
            if best_id is None or best_iou < self.iou_threshold:
                best_id = self.next_id
                self.next_id += 1
            self.tracks[best_id] = (det.xyxy, 0, det.cls)
            assigned.add(best_id)
            output.append((best_id, det))

        for track_id, (box, missing, cls) in list(self.tracks.items()):
            if track_id not in assigned:
                missing += 1
                if missing > self.max_missing:
                    del self.tracks[track_id]
                else:
                    self.tracks[track_id] = (box, missing, cls)
        return output


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Install ultralytics or provide precomputed tracks CSV.") from exc
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        result = self.model(frame, conf=self.conf, verbose=False)[0]
        detections: list[Detection] = []
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls not in {0, 32}:
                continue
            xyxy = box.xyxy.cpu().numpy()[0].astype(np.float32)
            score = float(box.conf.item())
            agent_type = 0 if cls == 32 else 1
            detections.append(Detection(xyxy=xyxy, score=score, cls=agent_type))
        return detections
