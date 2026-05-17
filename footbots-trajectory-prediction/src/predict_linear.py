from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def predict_linear(
    tracks_csv: str,
    out_csv: str,
    pred_steps: int,
    velocity_window: int,
    max_agents: int,
    min_visible: int,
    field_width: float,
    field_height: float,
) -> None:
    df = pd.read_csv(tracks_csv)
    required = {"frame", "agent_id", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")
    df = df.sort_values(["agent_id", "frame"])

    latest_frame = int(df["frame"].max())
    recent = df[df["frame"] >= latest_frame - max(velocity_window * 3, velocity_window)]
    counts = recent.groupby("agent_id")["frame"].nunique().sort_values(ascending=False)
    agent_ids = counts[counts >= min_visible].head(max_agents).index.tolist()
    if not agent_ids:
        raise RuntimeError("No agents have enough recent observations for linear prediction.")

    rows = []
    for agent_id in agent_ids:
        g = df[df["agent_id"] == agent_id].sort_values("frame").tail(velocity_window + 1)
        if len(g) < 2:
            continue
        first = g.iloc[0]
        last = g.iloc[-1]
        dt = max(1.0, float(last["frame"] - first["frame"]))
        vx = (float(last["x"]) - float(first["x"])) / dt
        vy = (float(last["y"]) - float(first["y"])) / dt

        # Smooth tiny detector jitter to zero, but keep real motion visible.
        speed = float(np.hypot(vx, vy))
        if speed < 0.01:
            vx = 0.0
            vy = 0.0

        for step in range(1, pred_steps + 1):
            x = min(field_width, max(0.0, float(last["x"]) + vx * step))
            y = min(field_height, max(0.0, float(last["y"]) + vy * step))
            rows.append({"future_step": step, "agent_id": int(agent_id), "x": x, "y": y})

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"saved linear predictions: {out_csv}")
    print(f"agents: {len(set(r['agent_id'] for r in rows)) if rows else 0}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a simple constant-velocity prediction CSV from clean tracks.")
    parser.add_argument("--tracks", default="runs/video_clean_tracks.csv")
    parser.add_argument("--out", default="runs/video_predictions_linear.csv")
    parser.add_argument("--pred-steps", type=int, default=40)
    parser.add_argument("--velocity-window", type=int, default=12)
    parser.add_argument("--max-agents", type=int, default=23)
    parser.add_argument("--min-visible", type=int, default=6)
    parser.add_argument("--field-width", type=float, default=105.0)
    parser.add_argument("--field-height", type=float, default=68.0)
    args = parser.parse_args()
    predict_linear(
        args.tracks,
        args.out,
        args.pred_steps,
        args.velocity_window,
        args.max_agents,
        args.min_visible,
        args.field_width,
        args.field_height,
    )


if __name__ == "__main__":
    main()