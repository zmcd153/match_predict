from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


FIELD_POINTS = [[0.0, 0.0], [105.0, 0.0], [105.0, 68.0], [0.0, 68.0]]


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
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
        obs_steps: int = 20,
        pred_steps: int = 40,
        input_dim: int = 3,
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


def init_homography(video: str, out_path: str) -> None:
    import cv2
    cap = cv2.VideoCapture(video)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from video: {video}")
    height, width = frame.shape[:2]
    margin_x = round(width * 0.08, 1)
    margin_y = round(height * 0.08, 1)
    cfg = {
        "note": "Rough template only. Replace image_points with real pitch corner/line points from your video.",
        "image_points": [
            [margin_x, margin_y],
            [width - margin_x, margin_y],
            [width - margin_x, height - margin_y],
            [margin_x, height - margin_y],
        ],
        "field_points": FIELD_POINTS,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Created rough homography template: {out_path}")
    print("Edit image_points for accurate field coordinates before trusting predictions.")


def load_homography(path: str) -> np.ndarray:
    import cv2
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    image_points = np.asarray(cfg["image_points"], dtype=np.float32)
    field_points = np.asarray(cfg["field_points"], dtype=np.float32)
    h, _ = cv2.findHomography(image_points, field_points)
    if h is None:
        raise ValueError("Could not estimate homography from the configured points.")
    return h


def image_to_field(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    import cv2
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    return cv2.perspectiveTransform(pts, homography).reshape(-1, 2)


def class_to_agent(cls: int, track_id: int, detector_mode: str) -> tuple[int, int] | None:
    if detector_mode == "custom":
        if cls == 1:
            return 0, 0
        if cls == 0:
            return track_id, 1
        return None
    if cls == 32:
        return 0, 0
    if cls == 0:
        return track_id, 1
    return None



def is_reasonable_player_box(xyxy: np.ndarray, cls: int, detector_mode: str, min_area: float, max_area: float, min_height: float) -> bool:
    mapped = class_to_agent(cls, 0, detector_mode)
    if mapped is None:
        return False
    _agent_id, agent_type = mapped
    if agent_type == 0:
        return True
    width = float(max(0.0, xyxy[2] - xyxy[0]))
    height = float(max(0.0, xyxy[3] - xyxy[1]))
    area = width * height
    if height < min_height:
        return False
    if area < min_area:
        return False
    if max_area > 0 and area > max_area:
        return False
    return True

def extract_tracks_from_video(
    video: str,
    weights: str,
    homography_path: str,
    out_csv: str,
    conf: float,
    max_frames: int,
    detector_mode: str,
    min_box_area: float,
    max_box_area: float,
    min_box_height: float,
) -> None:
    from ultralytics import YOLO

    model = YOLO(weights)
    homography = load_homography(homography_path)
    rows = []
    frame_idx = 0
    results = model.track(source=video, stream=True, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)
    for result in results:
        if max_frames and frame_idx >= max_frames:
            break
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            frame_idx += 1
            continue
        for xyxy_t, cls_t, conf_t, track_id_t in zip(boxes.xyxy, boxes.cls, boxes.conf, boxes.id):
            cls = int(cls_t.item())
            track_id = int(track_id_t.item())
            mapped = class_to_agent(cls, track_id, detector_mode)
            if mapped is None:
                continue
            agent_id, agent_type = mapped
            xyxy = xyxy_t.cpu().numpy()
            if not is_reasonable_player_box(xyxy, cls, detector_mode, min_box_area, max_box_area, min_box_height):
                continue
            foot = xyxy.reshape(2, 2).mean(axis=0)
            foot[1] = xyxy[3]
            field_xy = image_to_field(foot.reshape(1, 2), homography)[0]
            rows.append(
                {
                    "frame": frame_idx,
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "x": float(field_xy[0]),
                    "y": float(field_xy[1]),
                    "image_x": float(foot[0]),
                    "image_y": float(foot[1]),
                    "score": float(conf_t.item()),
                    "source_track_id": track_id,
                    "det_class": cls,
                }
            )
        frame_idx += 1
    if not rows:
        print("YOLO track() produced no track ids; falling back to predict() + simple IoU tracking.")
        from src.video_tracking import Detection, SimpleTracker

        tracker = SimpleTracker()
        frame_idx = 0
        results = model.predict(source=video, stream=True, conf=conf, imgsz=1280, verbose=False)
        for result in results:
            if max_frames and frame_idx >= max_frames:
                break
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for xyxy_t, cls_t, conf_t in zip(boxes.xyxy, boxes.cls, boxes.conf):
                    cls = int(cls_t.item())
                    xyxy = xyxy_t.cpu().numpy().astype(np.float32)
                    if not is_reasonable_player_box(xyxy, cls, detector_mode, min_box_area, max_box_area, min_box_height):
                        continue
                    detections.append(Detection(xyxy=xyxy, score=float(conf_t.item()), cls=cls))
            for track_id, det in tracker.update(detections):
                mapped = class_to_agent(det.cls, track_id, detector_mode)
                if mapped is None:
                    continue
                agent_id, agent_type = mapped
                xyxy = det.xyxy
                foot = xyxy.reshape(2, 2).mean(axis=0)
                foot[1] = xyxy[3]
                field_xy = image_to_field(foot.reshape(1, 2), homography)[0]
                rows.append(
                    {
                        "frame": frame_idx,
                        "agent_id": agent_id,
                        "agent_type": agent_type,
                        "x": float(field_xy[0]),
                        "y": float(field_xy[1]),
                        "image_x": float(foot[0]),
                        "image_y": float(foot[1]),
                        "score": float(det.score),
                        "source_track_id": track_id,
                        "det_class": det.cls,
                    }
                )
            frame_idx += 1

    if not rows:
        raise RuntimeError("No usable detections were produced. Try a lower --conf or better YOLO weights.")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved raw tracks: {out_csv} ({len(rows)} observations)")


def clean_tracks(raw_csv: str, out_csv: str, min_length: int = 30, max_gap: int = 10) -> None:
    df = pd.read_csv(raw_csv)
    df = df[df["x"].between(0, 105) & df["y"].between(0, 68)]
    if df.empty:
        raise RuntimeError("All detections fell outside the field. Check homography image_points.")
    counts = df.groupby("agent_id")["frame"].nunique()
    df = df[df["agent_id"].isin(counts[counts >= min_length].index)].copy()
    if df.empty:
        raise RuntimeError("No tracks survived cleaning. Lower --min-length or improve detection/tracking.")
    frames = range(int(df["frame"].min()), int(df["frame"].max()) + 1)
    cleaned = []
    for agent_id, group in df.groupby("agent_id"):
        group = group.sort_values("frame").drop_duplicates("frame", keep="last")
        agent_type = int(group["agent_type"].mode().iloc[0])
        indexed = group.set_index("frame").reindex(frames)
        indexed["agent_id"] = agent_id
        indexed["agent_type"] = agent_type
        interp_cols = ["x", "y"]
        if {"image_x", "image_y"}.issubset(indexed.columns):
            interp_cols += ["image_x", "image_y"]
        indexed[interp_cols] = indexed[interp_cols].interpolate(limit=max_gap, limit_area="inside")
        indexed = indexed.dropna(subset=["x", "y"]).reset_index().rename(columns={"index": "frame"})
        output_cols = ["frame", "agent_id", "agent_type", "x", "y"]
        if {"image_x", "image_y"}.issubset(indexed.columns):
            output_cols += ["image_x", "image_y"]
        cleaned.append(indexed[output_cols])
    out_df = pd.concat(cleaned, ignore_index=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved clean tracks: {out_csv} ({len(out_df)} observations)")


def build_latest_window(csv_path: str, obs_steps: int, max_agents: int) -> tuple[torch.Tensor, list[int]]:
    df = pd.read_csv(csv_path)
    frames = sorted(df["frame"].unique().tolist())
    if len(frames) < obs_steps:
        raise ValueError(f"Need at least {obs_steps} frames, got {len(frames)}.")
    frames = frames[-obs_steps:]
    recent = df[df["frame"].isin(frames)]
    visible_counts = recent.groupby("agent_id")["frame"].nunique().sort_values(ascending=False)
    agent_ids = visible_counts.head(max_agents).index.tolist()
    if not agent_ids:
        raise RuntimeError("No agents are visible in the latest observation window.")
    agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
    data = np.full((obs_steps, max_agents, 3), np.nan, dtype=np.float32)
    for row in recent.itertuples(index=False):
        if row.agent_id in agent_to_idx:
            data[frame_to_idx[row.frame], agent_to_idx[row.agent_id]] = (float(row.x), float(row.y), float(row.agent_type))

    for idx in range(len(agent_ids)):
        xy = pd.DataFrame(data[:, idx, :2]).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
        data[:, idx, :2] = xy
        if np.isnan(data[:, idx, 2]).any():
            known = data[:, idx, 2][~np.isnan(data[:, idx, 2])]
            data[:, idx, 2] = known[0] if len(known) else 1.0
    data = np.nan_to_num(data, nan=0.0)
    return torch.from_numpy(data).unsqueeze(0), agent_ids


def predict_tracks(checkpoint_path: str, tracks_csv: str, out_csv: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    obs_steps = int(checkpoint["obs_steps"])
    pred_steps = int(checkpoint["pred_steps"])
    max_agents = int(checkpoint.get("max_agents", 23))
    obs, agent_ids = build_latest_window(tracks_csv, obs_steps, max_agents)
    model = TrajectoryTransformer(obs_steps=obs_steps, pred_steps=pred_steps)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        pred = model(obs).squeeze(0).numpy()

    rows = []
    for t in range(pred_steps):
        for idx, agent_id in enumerate(agent_ids):
            rows.append({"future_step": t + 1, "agent_id": agent_id, "x": float(pred[t, idx, 0]), "y": float(pred[t, idx, 1])})
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved predictions: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run soccer video/tracking trajectory prediction.")
    parser.add_argument("--checkpoint", default="runs/metrica_game1.pt", help="Trajectory model checkpoint.")
    parser.add_argument("--tracks", default="", help="Existing tracking CSV. If omitted with --video, tracks are generated from video.")
    parser.add_argument("--video", default="", help="Input soccer video.")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO weights. Default uses COCO person/sports-ball classes.")
    parser.add_argument("--detector-mode", choices=["coco", "custom"], default="coco", help="coco: person=0, sports ball=32. custom: player=0, ball=1.")
    parser.add_argument("--homography", default="homography.json", help="Homography JSON path.")
    parser.add_argument("--init-homography", action="store_true", help="Create a rough homography template from the video frame size and exit.")
    parser.add_argument("--raw-tracks", default="runs/video_raw_tracks.csv")
    parser.add_argument("--clean-tracks", default="runs/video_clean_tracks.csv")
    parser.add_argument("--out", default="runs/predictions.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--min-box-area", type=float, default=80.0, help="Filter tiny player detections by pixel area.")
    parser.add_argument("--max-box-area", type=float, default=30000.0, help="Filter huge player detections by pixel area; 0 disables upper limit.")
    parser.add_argument("--min-box-height", type=float, default=10.0, help="Filter tiny player detections by pixel height.")
    args = parser.parse_args()

    if args.init_homography:
        if not args.video:
            raise ValueError("--init-homography requires --video.")
        init_homography(args.video, args.homography)
        return

    tracks_csv = args.tracks
    if args.video and not tracks_csv:
        if not Path(args.homography).exists():
            raise FileNotFoundError(
                f"Missing {args.homography}. Run with --init-homography first, then edit image_points for your pitch."
            )
        extract_tracks_from_video(
            args.video,
            args.weights,
            args.homography,
            args.raw_tracks,
            args.conf,
            args.max_frames,
            args.detector_mode,
            args.min_box_area,
            args.max_box_area,
            args.min_box_height,
        )
        clean_tracks(args.raw_tracks, args.clean_tracks, min_length=args.min_length, max_gap=args.max_gap)
        tracks_csv = args.clean_tracks

    if not tracks_csv:
        raise ValueError("Provide --tracks, or provide --video to generate tracks first.")
    predict_tracks(args.checkpoint, tracks_csv, args.out)


if __name__ == "__main__":
    main()
