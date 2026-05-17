from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PALETTE = [
    (40, 220, 255),
    (80, 180, 255),
    (120, 255, 120),
    (255, 180, 80),
    (255, 120, 180),
    (180, 120, 255),
    (255, 255, 80),
    (80, 255, 180),
    (255, 100, 100),
    (180, 255, 80),
]


def load_homography(path: str) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    image_points = np.asarray(cfg["image_points"], dtype=np.float32)
    field_points = np.asarray(cfg["field_points"], dtype=np.float32)
    h, _ = cv2.findHomography(image_points, field_points)
    if h is None:
        raise ValueError("Could not estimate homography.")
    return h


def field_to_image(points: np.ndarray, image_to_field_h: np.ndarray) -> np.ndarray:
    inv_h = np.linalg.inv(image_to_field_h)
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(pts, inv_h)
    return mapped.reshape(-1, 2)


def color_for(agent_id: int) -> tuple[int, int, int]:
    return PALETTE[abs(int(agent_id)) % len(PALETTE)]


def draw_label(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_polyline(frame: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(pts) < 2:
        return
    arr = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_prediction_arrow(frame: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(pts) < 2:
        return
    start = pts[-2]
    end = pts[-1]
    if abs(end[0] - start[0]) + abs(end[1] - start[1]) < 3:
        start = pts[0]
    cv2.arrowedLine(frame, start, end, color, thickness + 1, cv2.LINE_AA, tipLength=0.35)


def load_tracks(path: str, h: np.ndarray) -> dict[int, list[dict[str, object]]]:
    df = pd.read_csv(path)
    required = {"frame", "agent_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")
    df = df.sort_values(["frame", "agent_id"])
    if {"image_x", "image_y"}.issubset(df.columns):
        df["image_x"] = pd.to_numeric(df["image_x"], errors="coerce")
        df["image_y"] = pd.to_numeric(df["image_y"], errors="coerce")
    else:
        image_xy = field_to_image(df[["x", "y"]].to_numpy(dtype=np.float32), h)
        df["image_x"] = image_xy[:, 0]
        df["image_y"] = image_xy[:, 1]
    by_frame: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in df.itertuples(index=False):
        by_frame[int(row.frame)].append(
            {
                "agent_id": int(row.agent_id),
                "image_x": float(row.image_x),
                "image_y": float(row.image_y),
            }
        )
    return by_frame


def load_predictions(path: str, tracks_path: str, h: np.ndarray, px_per_meter: float) -> dict[int, list[tuple[int, int]]]:
    pred = pd.read_csv(path)
    required = {"future_step", "agent_id", "x", "y"}
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction CSV is missing columns: {sorted(missing)}")

    tracks = pd.read_csv(tracks_path)
    track_required = {"frame", "agent_id", "x", "y", "image_x", "image_y"}
    missing_tracks = track_required - set(tracks.columns)
    if missing_tracks:
        # Fallback for old CSVs: use homography back-projection.
        pred = pred.sort_values(["agent_id", "future_step"])
        image_xy = field_to_image(pred[["x", "y"]].to_numpy(dtype=np.float32), h)
        pred["image_x"] = image_xy[:, 0]
        pred["image_y"] = image_xy[:, 1]
        by_agent: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row in pred.itertuples(index=False):
            by_agent[int(row.agent_id)].append((int(round(row.image_x)), int(round(row.image_y))))
        return by_agent

    tracks = tracks.sort_values(["agent_id", "frame"])
    anchors: dict[int, tuple[float, float, float, float]] = {}
    for agent_id, group in tracks.groupby("agent_id"):
        last = group.iloc[-1]
        anchors[int(agent_id)] = (
            float(last["x"]),
            float(last["y"]),
            float(last["image_x"]),
            float(last["image_y"]),
        )

    pred = pred.sort_values(["agent_id", "future_step"])
    by_agent: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for agent_id, group in pred.groupby("agent_id"):
        agent_id = int(agent_id)
        if agent_id not in anchors:
            continue
        last_x, last_y, anchor_px, anchor_py = anchors[agent_id]
        for row in group.itertuples(index=False):
            # Draw future deltas from the actual last detected pixel location.
            # This avoids using an approximate homography to place the whole prediction path.
            dx = (float(row.x) - last_x) * px_per_meter
            dy = (float(row.y) - last_y) * px_per_meter
            by_agent[agent_id].append((int(round(anchor_px + dx)), int(round(anchor_py + dy))))
    return by_agent

def render_video(args: argparse.Namespace) -> None:
    h = load_homography(args.homography)
    tracks_by_frame = load_tracks(args.tracks, h) if args.tracks else {}
    predictions_by_agent = load_predictions(args.predictions, args.tracks, h, args.px_per_meter) if args.predictions else {}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out_path}")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail))
    frame_idx = 0
    last_frame: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_frame = frame.copy()
        for item in tracks_by_frame.get(frame_idx, []):
            agent_id = int(item["agent_id"])
            pt = (int(round(float(item["image_x"]))), int(round(float(item["image_y"]))))
            if -1000 <= pt[0] <= width + 1000 and -1000 <= pt[1] <= height + 1000:
                trails[agent_id].append(pt)
                color = color_for(agent_id)
                draw_polyline(frame, list(trails[agent_id]), color, thickness=2)
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                draw_label(frame, str(agent_id), (pt[0] + 6, pt[1] - 6), color)
        draw_label(frame, f"frame {frame_idx}", (18, 32), (255, 255, 255))
        writer.write(frame)
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()

    if last_frame is not None and predictions_by_agent:
        freeze_frames = max(1, int(round(args.prediction_seconds * fps)))
        for i in range(freeze_frames):
            frame = last_frame.copy()
            alpha = min(1.0, (i + 1) / max(1, freeze_frames // 2))
            overlay = frame.copy()
            for agent_id, pts in predictions_by_agent.items():
                color = color_for(agent_id)
                visible_count = max(2, int(round(len(pts) * alpha)))
                visible_pts = pts[:visible_count]
                draw_polyline(overlay, visible_pts, color, thickness=3)
                draw_prediction_arrow(overlay, visible_pts, color, thickness=3)
                for step_idx, pt in enumerate(visible_pts[::5], start=0):
                    cv2.circle(overlay, pt, 3, color, -1, cv2.LINE_AA)
                if visible_pts:
                    cv2.circle(overlay, visible_pts[0], 6, color, -1, cv2.LINE_AA)
                    draw_label(overlay, f"pred {agent_id}", (visible_pts[0][0] + 8, visible_pts[0][1] - 8), color)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            draw_label(frame, "future prediction", (18, 32), (255, 255, 255))
            writer.write(frame)

    writer.release()
    print(f"saved video: {out_path}")
    print(f"source frames read: {frame_idx}/{total_frames if total_frames else '?'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render tracking and prediction CSVs back onto a soccer video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--homography", default="homography.json")
    parser.add_argument("--tracks", default="runs/video_clean_tracks.csv")
    parser.add_argument("--predictions", default="runs/video_predictions.csv")
    parser.add_argument("--out", default="runs/video_predictions_overlay.mp4")
    parser.add_argument("--trail", type=int, default=40)
    parser.add_argument("--prediction-seconds", type=float, default=4.0)
    parser.add_argument("--px-per-meter", type=float, default=34.7, help="Pixel scale used to draw prediction deltas from the last observed player pixel.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fourcc", default="mp4v")
    args = parser.parse_args()
    render_video(args)


if __name__ == "__main__":
    main()