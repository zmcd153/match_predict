from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


FIELD_W = 105.0
FIELD_H = 68.0
PALETTE = [
    (30, 220, 255),
    (80, 170, 255),
    (80, 240, 120),
    (255, 180, 70),
    (255, 110, 180),
    (180, 120, 255),
    (240, 240, 80),
    (80, 240, 200),
    (255, 100, 100),
    (180, 255, 80),
]


def color_for(agent_id: int) -> tuple[int, int, int]:
    return PALETTE[abs(int(agent_id)) % len(PALETTE)]


def field_to_canvas(x: float, y: float, margin: int, scale: float) -> tuple[int, int]:
    return int(round(margin + x * scale)), int(round(margin + y * scale))


def draw_text(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int] = (245, 245, 245), scale: float = 0.55) -> None:
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_field(width: int, height: int, margin: int) -> tuple[np.ndarray, float]:
    scale = min((width - 2 * margin) / FIELD_W, (height - 2 * margin) / FIELD_H)
    canvas = np.full((height, width, 3), (35, 105, 85), dtype=np.uint8)

    # Subtle grass stripes.
    stripe_w = max(1, int(round(5 * scale)))
    x0 = margin
    x1 = int(round(margin + FIELD_W * scale))
    y0 = margin
    y1 = int(round(margin + FIELD_H * scale))
    for i, x in enumerate(range(x0, x1, stripe_w)):
        if i % 2 == 0:
            cv2.rectangle(canvas, (x, y0), (min(x + stripe_w, x1), y1), (40, 115, 92), -1)

    line = (230, 235, 225)
    p0 = field_to_canvas(0, 0, margin, scale)
    p1 = field_to_canvas(FIELD_W, FIELD_H, margin, scale)
    cv2.rectangle(canvas, p0, p1, line, 2, cv2.LINE_AA)

    mid_top = field_to_canvas(FIELD_W / 2, 0, margin, scale)
    mid_bottom = field_to_canvas(FIELD_W / 2, FIELD_H, margin, scale)
    cv2.line(canvas, mid_top, mid_bottom, line, 2, cv2.LINE_AA)

    center = field_to_canvas(FIELD_W / 2, FIELD_H / 2, margin, scale)
    cv2.circle(canvas, center, int(round(9.15 * scale)), line, 2, cv2.LINE_AA)
    cv2.circle(canvas, center, 3, line, -1, cv2.LINE_AA)

    # Penalty boxes, approximate standard dimensions.
    box_depth = 16.5
    box_width = 40.32
    y_box0 = (FIELD_H - box_width) / 2
    y_box1 = y_box0 + box_width
    cv2.rectangle(canvas, field_to_canvas(0, y_box0, margin, scale), field_to_canvas(box_depth, y_box1, margin, scale), line, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, field_to_canvas(FIELD_W - box_depth, y_box0, margin, scale), field_to_canvas(FIELD_W, y_box1, margin, scale), line, 2, cv2.LINE_AA)

    goal_depth = 5.5
    goal_width = 18.32
    y_goal0 = (FIELD_H - goal_width) / 2
    y_goal1 = y_goal0 + goal_width
    cv2.rectangle(canvas, field_to_canvas(0, y_goal0, margin, scale), field_to_canvas(goal_depth, y_goal1, margin, scale), line, 2, cv2.LINE_AA)
    cv2.rectangle(canvas, field_to_canvas(FIELD_W - goal_depth, y_goal0, margin, scale), field_to_canvas(FIELD_W, y_goal1, margin, scale), line, 2, cv2.LINE_AA)
    return canvas, scale


def load_tracks(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"frame", "agent_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")
    return df.sort_values(["frame", "agent_id"]).copy()


def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"future_step", "agent_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")
    return df.sort_values(["agent_id", "future_step"]).copy()


def draw_tracks(frame: np.ndarray, rows: pd.DataFrame, trails: dict[int, deque[tuple[int, int]]], margin: int, scale: float, trail_len: int) -> None:
    for item in rows.itertuples(index=False):
        agent_id = int(item.agent_id)
        pt = field_to_canvas(float(item.x), float(item.y), margin, scale)
        trails[agent_id].append(pt)
        color = color_for(agent_id)
        pts = list(trails[agent_id])[-trail_len:]
        if len(pts) >= 2:
            cv2.polylines(frame, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, color, -1, cv2.LINE_AA)
        draw_text(frame, str(agent_id), (pt[0] + 7, pt[1] - 7), color, scale=0.45)


def draw_predictions(frame: np.ndarray, pred: pd.DataFrame, margin: int, scale: float, reveal: float = 1.0, prediction_scale: float = 1.0) -> None:
    for agent_id, group in pred.groupby("agent_id"):
        color = color_for(int(agent_id))
        rows = list(group.itertuples(index=False))
        if not rows:
            continue
        anchor_x = float(rows[0].x)
        anchor_y = float(rows[0].y)
        pts = []
        for row in rows:
            x = anchor_x + (float(row.x) - anchor_x) * prediction_scale
            y = anchor_y + (float(row.y) - anchor_y) * prediction_scale
            pts.append(field_to_canvas(x, y, margin, scale))
        visible = max(2, int(round(len(pts) * reveal)))
        pts = pts[:visible]
        cv2.polylines(frame, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], False, color, 3, cv2.LINE_AA)
        for idx, pt in enumerate(pts):
            if idx % 5 == 0 or idx == len(pts) - 1:
                cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, pts[0], 7, color, -1, cv2.LINE_AA)
        draw_text(frame, f"pred {int(agent_id)}", (pts[0][0] + 8, pts[0][1] - 8), color, scale=0.45)


def render(args: argparse.Namespace) -> None:
    tracks = load_tracks(args.tracks)
    predictions = load_predictions(args.predictions) if args.predictions else pd.DataFrame()
    frames = sorted(int(v) for v in tracks["frame"].unique())
    if not frames:
        raise RuntimeError("No track frames to render.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.fourcc), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out}")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail))
    last_frame = None
    for idx, frame_no in enumerate(frames):
        canvas, scale = draw_field(args.width, args.height, args.margin)
        rows = tracks[tracks["frame"] == frame_no]
        draw_tracks(canvas, rows, trails, args.margin, scale, args.trail)
        draw_text(canvas, f"clean tracks | frame {frame_no}", (args.margin, 28), (255, 255, 255), scale=0.6)
        writer.write(canvas)
        last_frame = canvas
        if args.max_frames and idx + 1 >= args.max_frames:
            break

    if last_frame is not None and not predictions.empty:
        freeze_frames = max(1, int(round(args.prediction_seconds * args.fps)))
        for i in range(freeze_frames):
            canvas, scale = draw_field(args.width, args.height, args.margin)
            # Keep final observed positions visible under the prediction.
            final_frame = frames[min(len(frames) - 1, args.max_frames - 1)] if args.max_frames else frames[-1]
            final_rows = tracks[tracks["frame"] == final_frame]
            temp_trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail))
            draw_tracks(canvas, final_rows, temp_trails, args.margin, scale, args.trail)
            reveal = min(1.0, (i + 1) / max(1, freeze_frames // 2))
            draw_predictions(canvas, predictions, args.margin, scale, reveal=reveal, prediction_scale=args.prediction_scale)
            draw_text(canvas, "future prediction", (args.margin, 28), (255, 255, 255), scale=0.6)
            writer.write(canvas)

    writer.release()
    print(f"saved video: {out}")
    print(f"rendered track frames: {len(frames)}")
    print(f"agents in predictions: {0 if predictions.empty else predictions['agent_id'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render tracks and predictions as a clean top-down field video.")
    parser.add_argument("--tracks", default="runs/video_clean_tracks.csv")
    parser.add_argument("--predictions", default="runs/video_predictions.csv")
    parser.add_argument("--out", default="runs/video_predictions_field.mp4")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=920)
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--trail", type=int, default=45)
    parser.add_argument("--prediction-seconds", type=float, default=5.0)
    parser.add_argument("--prediction-scale", type=float, default=1.0, help="Visually exaggerate future displacement around each first prediction point.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fourcc", default="mp4v")
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()