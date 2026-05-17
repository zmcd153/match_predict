from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd
import torch

from .data import build_latest_window
from .model import TrajectoryTransformer
from .video_tracking import SimpleTracker, YoloDetector, image_to_field, load_homography


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--homography", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tmp-tracks", default="runs/video_tracks.csv")
    args = parser.parse_args()

    homography = load_homography(args.homography)
    detector = YoloDetector(args.weights, args.conf)
    tracker = SimpleTracker()
    cap = cv2.VideoCapture(args.video)
    rows = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        detections = detector(frame)
        tracked = tracker.update(detections)
        for track_id, det in tracked:
            foot = det.xyxy.reshape(2, 2).mean(axis=0)
            foot[1] = det.xyxy[3]
            field_xy = image_to_field(foot.reshape(1, 2), homography)[0]
            rows.append(
                {
                    "frame": frame_idx,
                    "agent_id": track_id,
                    "agent_type": det.cls,
                    "x": float(field_xy[0]),
                    "y": float(field_xy[1]),
                    "score": det.score,
                }
            )
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break
    cap.release()

    Path(args.tmp_tracks).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.tmp_tracks, index=False)
    if not rows:
        raise RuntimeError("No detections were produced from the video.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    obs_steps = int(checkpoint["obs_steps"])
    pred_steps = int(checkpoint["pred_steps"])
    obs, agent_ids = build_latest_window(args.tmp_tracks, obs_steps=obs_steps)
    model = TrajectoryTransformer(obs_steps=obs_steps, pred_steps=pred_steps)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        pred = model(obs).squeeze(0).numpy()

    pred_rows = []
    for t in range(pred_steps):
        for idx, agent_id in enumerate(agent_ids):
            pred_rows.append(
                {
                    "future_step": t + 1,
                    "agent_id": agent_id,
                    "x": float(pred[t, idx, 0]),
                    "y": float(pred[t, idx, 1]),
                }
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(args.out, index=False)
    print(f"saved tracks to {args.tmp_tracks}")
    print(f"saved predictions to {args.out}")


if __name__ == "__main__":
    main()
