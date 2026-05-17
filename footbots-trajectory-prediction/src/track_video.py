from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

from .video_tracking import image_to_field, load_homography


def load_team_map(path: str | None) -> dict[str, int]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and track players/ball, then export field-coordinate tracks.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True, help="YOLO weights trained with class 0=player, 1=ball.")
    parser.add_argument("--homography", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--team-map", default="", help="Optional JSON: track_id -> agent_type, where 1=offense, 2=defense.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    homography = load_homography(args.homography)
    model = YOLO(args.weights)
    team_map = load_team_map(args.team_map or None)
    rows = []
    frame_idx = 0

    results = model.track(
        source=args.video,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        verbose=False,
    )
    for result in results:
        if args.max_frames and frame_idx >= args.max_frames:
            break
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            frame_idx += 1
            continue
        for xyxy_t, cls_t, conf_t, track_id_t in zip(boxes.xyxy, boxes.cls, boxes.conf, boxes.id):
            xyxy = xyxy_t.cpu().numpy()
            cls = int(cls_t.item())
            track_id = int(track_id_t.item())
            score = float(conf_t.item())
            foot = xyxy.reshape(2, 2).mean(axis=0)
            foot[1] = xyxy[3]
            field_xy = image_to_field(foot.reshape(1, 2), homography)[0]
            if cls == 1:
                agent_id = 0
                agent_type = 0
            else:
                agent_id = track_id
                agent_type = team_map.get(str(track_id), 1)
            rows.append(
                {
                    "frame": frame_idx,
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "x": float(field_xy[0]),
                    "y": float(field_xy[1]),
                    "score": score,
                    "source_track_id": track_id,
                    "det_class": cls,
                }
            )
        frame_idx += 1
    cap.release()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {len(rows)} observations to {args.out}")


if __name__ == "__main__":
    main()
