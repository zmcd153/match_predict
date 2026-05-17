from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x


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
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        y = self.ffn(x)
        return self.norm2(x + self.dropout(y))


class SocioTemporalBlock(nn.Module):
    """Temporal attention followed by social/player attention."""

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
    """FootBots/TranSPORTmer-style trajectory predictor."""

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
        x = self.input_proj(obs)
        x = self.pos(x)
        for block in self.encoder:
            x = block(x)

        last_context = x[:, -1:].expand(batch, self.pred_steps, agents, -1)
        global_context = x.mean(dim=1, keepdim=True).expand(batch, self.pred_steps, agents, -1)
        queries = self.future_queries.view(1, self.pred_steps, 1, -1).expand(batch, -1, agents, -1)
        y = queries + self.context_gate(torch.cat([last_context, global_context], dim=-1))
        y = self.pos(y)
        for block in self.decoder:
            y = block(y)

        delta = self.output(y)
        pred = obs[:, -1:, :, :2] + torch.cumsum(delta, dim=1)
        width, height = self.field_size
        return torch.stack([pred[..., 0].clamp(0, width), pred[..., 1].clamp(0, height)], dim=-1)


class TrackWindowDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        obs_steps: int,
        pred_steps: int,
        stride: int = 2,
        max_agents: int = 23,
        min_visible_ratio: float = 0.75,
    ) -> None:
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.total_steps = obs_steps + pred_steps
        df = pd.read_csv(csv_path)
        required = {"frame", "agent_id", "agent_type", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV 缺少字段: {sorted(missing)}")

        self.agent_ids = sorted(df["agent_id"].unique().tolist())[:max_agents]
        agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        frames = sorted(df["frame"].unique().tolist())
        frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
        data = np.full((len(frames), max_agents, 3), np.nan, dtype=np.float32)
        visible = np.zeros((len(frames), max_agents), dtype=np.float32)

        for row in df.itertuples(index=False):
            if row.agent_id not in agent_to_idx:
                continue
            t = frame_to_idx[row.frame]
            a = agent_to_idx[row.agent_id]
            data[t, a] = (float(row.x), float(row.y), float(row.agent_type))
            visible[t, a] = 1.0

        self.data = data
        self.visible = visible
        self.windows = []
        for start in range(0, max(0, len(frames) - self.total_steps + 1), stride):
            mask = visible[start : start + self.total_steps]
            if mask.mean() >= min_visible_ratio:
                self.windows.append(start)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = self.windows[idx]
        window = self.data[start : start + self.total_steps].copy()
        mask = self.visible[start : start + self.total_steps].copy()
        window = np.nan_to_num(window, nan=0.0)
        obs = window[: self.obs_steps]
        target = window[self.obs_steps :, :, :2]
        target_mask = mask[self.obs_steps :]
        return torch.from_numpy(obs), torch.from_numpy(target), torch.from_numpy(target_mask)


def masked_ade(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = torch.linalg.norm(pred - target, dim=-1)
    return (dist * mask).sum() / mask.sum().clamp_min(1.0)


def convert_metrica_tracking(home_csv: str, away_csv: str, out_csv: str) -> str:
    """Convert Metrica sample tracking CSVs into frame,agent_id,agent_type,x,y."""

    home = pd.read_csv(home_csv, skiprows=2)
    away = pd.read_csv(away_csv, skiprows=2)
    rows = []

    def add_team(df: pd.DataFrame, agent_type: int, id_offset: int) -> None:
        frame_col = "Frame"
        if frame_col not in df.columns:
            raise ValueError(f"{frame_col} column was not found in {home_csv}")
        columns = list(df.columns)
        for col_idx in range(3, len(columns) - 2, 2):
            x_col = columns[col_idx]
            y_col = columns[col_idx + 1]
            if "ball" in str(x_col).lower():
                continue
            raw_name = str(x_col).strip()
            digits = "".join(ch for ch in raw_name if ch.isdigit())
            player_num = int(digits) if digits else col_idx
            agent_id = id_offset + player_num
            valid = df[[frame_col, x_col, y_col]].dropna()
            for item in valid.itertuples(index=False):
                rows.append(
                    {
                        "frame": int(item[0]),
                        "agent_id": agent_id,
                        "agent_type": agent_type,
                        "x": float(item[1]) * 105.0,
                        "y": float(item[2]) * 68.0,
                    }
                )

    add_team(home, agent_type=1, id_offset=100)
    add_team(away, agent_type=2, id_offset=200)

    ball_x = home.columns[-2]
    ball_y = home.columns[-1]
    valid_ball = home[["Frame", ball_x, ball_y]].dropna()
    for item in valid_ball.itertuples(index=False):
        rows.append(
            {
                "frame": int(item[0]),
                "agent_id": 0,
                "agent_type": 0,
                "x": float(item[1]) * 105.0,
                "y": float(item[2]) * 68.0,
            }
        )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["frame", "agent_id"]).to_csv(out_csv, index=False)
    print(f"converted Metrica tracking to {out_csv}")
    return out_csv


def convert_skillcorner_match(match_dir: str, out_csv: str, pitch_width: float = 105.0, pitch_height: float = 68.0) -> str:
    """Convert one SkillCorner open-data match folder into training CSV."""

    match_path = Path(match_dir)
    jsonl_files = list(match_path.glob("*_tracking_extrapolated.jsonl"))
    if not jsonl_files:
        raise ValueError(f"No *_tracking_extrapolated.jsonl found in {match_dir}")

    rows = []
    tracking_path = jsonl_files[0]
    with tracking_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            frame = int(item["frame"])

            ball = item.get("ball_data") or {}
            if ball.get("x") is not None and ball.get("y") is not None:
                rows.append(
                    {
                        "frame": frame,
                        "agent_id": 0,
                        "agent_type": 0,
                        "x": float(ball["x"]) + pitch_width / 2.0,
                        "y": float(ball["y"]) + pitch_height / 2.0,
                    }
                )

            for player in item.get("player_data", []):
                if player.get("x") is None or player.get("y") is None:
                    continue
                player_id = int(player.get("player_id"))
                rows.append(
                    {
                        "frame": frame,
                        "agent_id": player_id,
                        "agent_type": 1,
                        "x": float(player["x"]) + pitch_width / 2.0,
                        "y": float(player["y"]) + pitch_height / 2.0,
                    }
                )

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values(["frame", "agent_id"]).to_csv(out_csv, index=False)
    print(f"converted SkillCorner tracking to {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="训练足球运动员轨迹预测模型")
    parser.add_argument("--tracks", default="", help="清洗后的轨迹 CSV，字段为 frame,agent_id,agent_type,x,y")
    parser.add_argument("--source-format", choices=["standard", "metrica", "skillcorner"], default="standard")
    parser.add_argument("--home-tracking", default="", help="Metrica Home tracking CSV")
    parser.add_argument("--away-tracking", default="", help="Metrica Away tracking CSV")
    parser.add_argument("--skillcorner-match-dir", default="", help="SkillCorner 单场比赛文件夹")
    parser.add_argument("--converted-tracks", default="data/converted_tracks.csv")
    parser.add_argument("--out", default="runs/real.pt", help="模型保存路径")
    parser.add_argument("--obs-steps", type=int, default=20, help="观察多少帧")
    parser.add_argument("--pred-steps", type=int, default=40, help="预测未来多少帧")
    parser.add_argument("--max-agents", type=int, default=23, help="最多 agent 数，足球常用 22 人 + 球")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-visible-ratio", type=float, default=0.75)
    args = parser.parse_args()

    tracks_path = args.tracks
    if args.source_format == "metrica":
        if not args.home_tracking or not args.away_tracking:
            raise ValueError("source-format=metrica 时需要 --home-tracking 和 --away-tracking。")
        tracks_path = convert_metrica_tracking(args.home_tracking, args.away_tracking, args.converted_tracks)
    elif args.source_format == "skillcorner":
        if not args.skillcorner_match_dir:
            raise ValueError("source-format=skillcorner 时需要 --skillcorner-match-dir。")
        tracks_path = convert_skillcorner_match(args.skillcorner_match_dir, args.converted_tracks)

    if not tracks_path:
        raise ValueError("请提供 --tracks，或者用 --source-format metrica/skillcorner 先转换公开数据。")

    dataset = TrackWindowDataset(
        tracks_path,
        obs_steps=args.obs_steps,
        pred_steps=args.pred_steps,
        stride=args.stride,
        max_agents=args.max_agents,
        min_visible_ratio=args.min_visible_ratio,
    )
    if len(dataset) < 2:
        raise RuntimeError("有效训练窗口太少，请增加视频数据，或降低 --min-visible-ratio。")

    val_size = max(1, int(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(7))
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TrajectoryTransformer(obs_steps=args.obs_steps, pred_steps=args.pred_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for obs, target, mask in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            obs, target, mask = obs.to(device), target.to(device), mask.to(device)
            loss = masked_ade(model(obs), target, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, target, mask in val_loader:
                obs, target, mask = obs.to(device), target.to(device), mask.to(device)
                val_losses.append(masked_ade(model(obs), target, mask).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        print(f"train_ade={train_loss:.3f} val_ade={val_loss:.3f}")
        if val_loss < best_val:
            best_val = val_loss
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "obs_steps": args.obs_steps,
                    "pred_steps": args.pred_steps,
                    "max_agents": args.max_agents,
                    "val_ade": best_val,
                },
                args.out,
            )
            print(f"已保存最佳模型: {args.out}")


if __name__ == "__main__":
    main()
