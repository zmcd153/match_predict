from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn

PALETTE = [
    (40, 220, 255), (80, 180, 255), (120, 255, 120), (255, 180, 80),
    (255, 120, 180), (180, 120, 255), (255, 255, 80), (80, 255, 180),
    (255, 100, 100), (180, 255, 80),
]


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.shape[1]].view(1, x.shape[1], 1, -1)


class SetAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        return self.norm2(x + self.dropout(self.ffn(x)))


class SocioTemporalBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.temporal = SetAttentionBlock(dim, heads, dropout)
        self.social = SetAttentionBlock(dim, heads, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, agents, dim = x.shape
        x_t = x.permute(0, 2, 1, 3).reshape(batch * agents, time, dim)
        x_t = self.temporal(x_t)
        x = x_t.reshape(batch, agents, time, dim).permute(0, 2, 1, 3)
        x_s = x.reshape(batch * time, agents, dim)
        x_s = self.social(x_s)
        return x_s.reshape(batch, time, agents, dim)


class TrajectoryTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int = 3,
        obs_steps: int = 20,
        pred_steps: int = 40,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        dropout: float = 0.1,
        field_size: tuple[float, float] = (105.0, 68.0),
    ) -> None:
        super().__init__()
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.field_size = field_size
        self.input_proj = nn.Sequential(nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.pos = PositionalEncoding(dim, max_len=max(obs_steps, pred_steps) + 8)
        self.encoder = nn.ModuleList([SocioTemporalBlock(dim, heads, dropout) for _ in range(depth)])
        self.future_queries = nn.Parameter(torch.randn(pred_steps, dim) * 0.02)
        self.decoder = nn.ModuleList([SocioTemporalBlock(dim, heads, dropout) for _ in range(max(1, depth // 2))])
        self.context_gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch, time, agents, _ = obs.shape
        if time != self.obs_steps:
            raise ValueError(f"Expected {self.obs_steps} observed steps, got {time}.")
        x = self.pos(self.input_proj(obs))
        for block in self.encoder:
            x = block(x)
        last_context = x[:, -1:].expand(batch, self.pred_steps, agents, -1)
        global_context = x.mean(dim=1, keepdim=True).expand(batch, self.pred_steps, agents, -1)
        queries = self.future_queries.view(1, self.pred_steps, 1, -1).expand(batch, -1, agents, -1)
        y = self.pos(queries + self.context_gate(torch.cat([last_context, global_context], dim=-1)))
        for block in self.decoder:
            y = block(y)
        pred = obs[:, -1:, :, :2] + torch.cumsum(self.output(y), dim=1)
        width, height = self.field_size
        return torch.stack([pred[..., 0].clamp(0, width), pred[..., 1].clamp(0, height)], dim=-1)


def color_for(agent_id: int) -> tuple[int, int, int]:
    return PALETTE[abs(int(agent_id)) % len(PALETTE)]


def draw_label(frame: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def draw_polyline(frame: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(pts) < 2:
        return
    cv2.polylines(frame, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def draw_prediction_arrow(frame: np.ndarray, pts: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(pts) < 2:
        return
    start = pts[-2]
    end = pts[-1]
    if abs(end[0] - start[0]) + abs(end[1] - start[1]) < 3:
        start = pts[0]
    cv2.arrowedLine(frame, start, end, color, thickness + 1, cv2.LINE_AA, tipLength=0.35)


def load_model(checkpoint_path: str) -> tuple[TrajectoryTransformer, int, int, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_steps = int(checkpoint["obs_steps"])
    pred_steps = int(checkpoint["pred_steps"])
    max_agents = int(checkpoint.get("max_agents", 23))
    model = TrajectoryTransformer(obs_steps=obs_steps, pred_steps=pred_steps)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, obs_steps, pred_steps, max_agents


def make_window(df: pd.DataFrame, current_frame: int, obs_steps: int, max_agents: int, min_visible: int) -> tuple[torch.Tensor | None, list[int], dict[int, tuple[float, float, float, float]]]:
    frames = list(range(current_frame - obs_steps + 1, current_frame + 1))
    recent = df[df["frame"].isin(frames)].copy()
    if recent.empty:
        return None, [], {}
    counts = recent.groupby("agent_id")["frame"].nunique().sort_values(ascending=False)
    agent_ids = counts[counts >= min_visible].head(max_agents).index.tolist()
    if not agent_ids:
        return None, [], {}

    agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
    data = np.full((obs_steps, max_agents, 3), np.nan, dtype=np.float32)
    anchors: dict[int, tuple[float, float, float, float]] = {}

    for row in recent.itertuples(index=False):
        agent_id = int(row.agent_id)
        if agent_id not in agent_to_idx:
            continue
        t = frame_to_idx[int(row.frame)]
        a = agent_to_idx[agent_id]
        data[t, a] = (float(row.x), float(row.y), float(row.agent_type))

    for agent_id in agent_ids:
        a = agent_to_idx[agent_id]
        g = recent[recent["agent_id"] == agent_id].sort_values("frame")
        last = g.iloc[-1]
        anchors[agent_id] = (float(last["x"]), float(last["y"]), float(last["image_x"]), float(last["image_y"]))
        xy = pd.DataFrame(data[:, a, :2]).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
        data[:, a, :2] = xy
        known = data[:, a, 2][~np.isnan(data[:, a, 2])]
        data[:, a, 2] = known[0] if len(known) else 1.0

    data = np.nan_to_num(data, nan=0.0)
    return torch.from_numpy(data).unsqueeze(0), [int(x) for x in agent_ids], anchors


def predictions_to_pixels(pred: np.ndarray, agent_ids: list[int], anchors: dict[int, tuple[float, float, float, float]], px_per_meter: float, scale: float) -> dict[int, list[tuple[int, int]]]:
    out: dict[int, list[tuple[int, int]]] = {}
    for idx, agent_id in enumerate(agent_ids):
        if agent_id not in anchors:
            continue
        last_x, last_y, anchor_px, anchor_py = anchors[agent_id]
        pts = []
        for step_xy in pred[:, idx, :]:
            dx = (float(step_xy[0]) - last_x) * px_per_meter * scale
            dy = (float(step_xy[1]) - last_y) * px_per_meter * scale
            pts.append((int(round(anchor_px + dx)), int(round(anchor_py + dy))))
        out[agent_id] = pts
    return out


def render(args: argparse.Namespace) -> None:
    tracks = pd.read_csv(args.tracks)
    required = {"frame", "agent_id", "agent_type", "x", "y", "image_x", "image_y"}
    missing = required - set(tracks.columns)
    if missing:
        raise ValueError(f"Tracks are missing columns: {sorted(missing)}. Re-run main.py to regenerate tracks.")
    tracks = tracks.sort_values(["frame", "agent_id"])
    by_frame = {int(frame): group for frame, group in tracks.groupby("frame")}

    model, obs_steps, pred_steps, max_agents = load_model(args.checkpoint)
    if args.max_agents > 0:
        max_agents = min(max_agents, args.max_agents)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*args.fourcc), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {out}")

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=args.trail))
    cached_predictions: dict[int, list[tuple[int, int]]] = {}
    frames_with_prediction = 0
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rows = by_frame.get(frame_idx)
        if rows is not None:
            for row in rows.itertuples(index=False):
                agent_id = int(row.agent_id)
                pt = (int(round(float(row.image_x))), int(round(float(row.image_y))))
                trails[agent_id].append(pt)
                color = color_for(agent_id)
                draw_polyline(frame, list(trails[agent_id]), color, 2)
                cv2.circle(frame, pt, 5, color, -1, cv2.LINE_AA)
                draw_label(frame, str(agent_id), (pt[0] + 7, pt[1] - 7), color)

        if frame_idx >= obs_steps - 1 and (frame_idx % args.predict_every == 0 or not cached_predictions):
            obs, agent_ids, anchors = make_window(tracks, frame_idx, obs_steps, max_agents, args.min_visible)
            if obs is not None:
                with torch.no_grad():
                    pred = model(obs).squeeze(0).numpy()
                cached_predictions = predictions_to_pixels(pred, agent_ids, anchors, args.px_per_meter, args.prediction_scale)
                frames_with_prediction += 1

        for agent_id, pts in cached_predictions.items():
            color = color_for(agent_id)
            visible_pts = pts[: args.pred_steps] if args.pred_steps > 0 else pts
            draw_polyline(frame, visible_pts, color, args.pred_thickness)
            draw_prediction_arrow(frame, visible_pts, color, args.pred_thickness)
            if visible_pts:
                cv2.circle(frame, visible_pts[0], 4, color, -1, cv2.LINE_AA)

        draw_label(frame, f"live prediction | frame {frame_idx}", (18, 32), (255, 255, 255))
        writer.write(frame)
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    cap.release()
    writer.release()
    print(f"saved video: {out}")
    print(f"source frames read: {frame_idx}/{total_frames if total_frames else '?'}")
    print(f"prediction updates: {frames_with_prediction}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render rolling/live trajectory predictions on every video frame.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracks", default="runs/video_clean_tracks.csv")
    parser.add_argument("--checkpoint", default="runs/metrica_game1.pt")
    parser.add_argument("--out", default="runs/video_predictions_live.mp4")
    parser.add_argument("--predict-every", type=int, default=5, help="Run the model every N frames and reuse the latest prediction between updates.")
    parser.add_argument("--min-visible", type=int, default=8, help="Minimum observations inside the model window for an agent to be predicted.")
    parser.add_argument("--max-agents", type=int, default=10)
    parser.add_argument("--pred-steps", type=int, default=20, help="How many future steps to draw; 0 draws all checkpoint steps.")
    parser.add_argument("--px-per-meter", type=float, default=34.7)
    parser.add_argument("--prediction-scale", type=float, default=1.0)
    parser.add_argument("--pred-thickness", type=int, default=2)
    parser.add_argument("--trail", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--fourcc", default="mp4v")
    args = parser.parse_args()
    render(args)


if __name__ == "__main__":
    main()