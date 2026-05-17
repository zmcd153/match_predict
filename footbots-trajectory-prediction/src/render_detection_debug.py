from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import pandas as pd

PALETTE = [
    (40, 220, 255), (80, 180, 255), (120, 255, 120), (255, 180, 80),
    (255, 120, 180), (180, 120, 255), (255, 255, 80), (80, 255, 180),
]


def color_for(agent_id: int) -> tuple[int, int, int]:
    return PALETTE[abs(int(agent_id)) % len(PALETTE)]


def draw_text(frame, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render raw detection foot points directly from image_x/image_y.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--raw-tracks", default="runs/video_raw_tracks.csv")
    parser.add_argument("--out", default="runs/video_detection_debug.mp4")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fourcc", default="mp4v")
    args = parser.parse_args()

    df = pd.read_csv(args.raw_tracks)
    required = {"frame", "agent_id", "image_x", "image_y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw tracks are missing columns: {sorted(missing)}. Re-run main.py first.")
    df = df.sort_values(["frame", "agent_id"])
    by_frame = {int(frame): group for frame, group in df.groupby("frame")}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out}")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        group = by_frame.get(frame_idx)
        if group is not None:
            for row in group.itertuples(index=False):
                agent_id = int(row.agent_id)
                x = int(round(float(row.image_x)))
                y = int(round(float(row.image_y)))
                color = color_for(agent_id)
                cv2.drawMarker(frame, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2, line_type=cv2.LINE_AA)
                cv2.circle(frame, (x, y), 5, color, -1, cv2.LINE_AA)
                draw_text(frame, f"id {agent_id}", (x + 8, y - 8), color)
        draw_text(frame, f"raw detections | frame {frame_idx}", (18, 32), (255, 255, 255))
        writer.write(frame)
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"saved video: {out}")
    print(f"frames rendered: {frame_idx}")


if __name__ == "__main__":
    main()