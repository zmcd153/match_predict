from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FieldSpec:
    width: float = 105.0
    height: float = 68.0


def normalize_xy(xy: np.ndarray, field: FieldSpec = FieldSpec()) -> np.ndarray:
    scale = np.array([field.width, field.height], dtype=np.float32)
    return xy.astype(np.float32) / scale


def denormalize_xy(xy: np.ndarray, field: FieldSpec = FieldSpec()) -> np.ndarray:
    scale = np.array([field.width, field.height], dtype=np.float32)
    return xy.astype(np.float32) * scale


class TrackWindowDataset(Dataset):
    def __init__(self, csv_path: str, obs_steps: int, pred_steps: int, stride: int = 1, max_agents: int = 23) -> None:
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.total_steps = obs_steps + pred_steps
        self.max_agents = max_agents
        df = pd.read_csv(csv_path)
        required = {"frame", "agent_id", "agent_type", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")

        self.agent_ids = sorted(df["agent_id"].unique().tolist())[:max_agents]
        self.agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        frames = sorted(df["frame"].unique().tolist())
        self.frames = frames
        frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}

        data = np.full((len(frames), max_agents, 3), np.nan, dtype=np.float32)
        for row in df.itertuples(index=False):
            if row.agent_id not in self.agent_to_idx:
                continue
            t = frame_to_idx[row.frame]
            a = self.agent_to_idx[row.agent_id]
            data[t, a] = (float(row.x), float(row.y), float(row.agent_type))

        self.data = data
        self.windows = list(range(0, max(0, len(frames) - self.total_steps + 1), stride))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.windows[idx]
        window = self.data[start : start + self.total_steps].copy()
        window = np.nan_to_num(window, nan=0.0)
        obs = window[: self.obs_steps]
        target = window[self.obs_steps :, :, :2]
        return torch.from_numpy(obs), torch.from_numpy(target)


class MaskedTrackWindowDataset(Dataset):
    """Trajectory windows for real tracking data with missing-observation masks."""

    def __init__(
        self,
        csv_path: str,
        obs_steps: int,
        pred_steps: int,
        stride: int = 1,
        max_agents: int = 23,
        min_visible_ratio: float = 0.75,
    ) -> None:
        self.obs_steps = obs_steps
        self.pred_steps = pred_steps
        self.total_steps = obs_steps + pred_steps
        self.max_agents = max_agents
        df = pd.read_csv(csv_path)
        required = {"frame", "agent_id", "agent_type", "x", "y"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")

        self.agent_ids = sorted(df["agent_id"].unique().tolist())[:max_agents]
        self.agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)}
        frames = sorted(df["frame"].unique().tolist())
        frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
        data = np.full((len(frames), max_agents, 3), np.nan, dtype=np.float32)
        visible = np.zeros((len(frames), max_agents), dtype=np.float32)

        for row in df.itertuples(index=False):
            if row.agent_id not in self.agent_to_idx:
                continue
            t = frame_to_idx[row.frame]
            a = self.agent_to_idx[row.agent_id]
            data[t, a] = (float(row.x), float(row.y), float(row.agent_type))
            visible[t, a] = 1.0

        self.data = data
        self.visible = visible
        self.windows: list[int] = []
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


def build_latest_window(csv_path: str, obs_steps: int, max_agents: int = 23) -> tuple[torch.Tensor, list[int]]:
    df = pd.read_csv(csv_path)
    frames = sorted(df["frame"].unique().tolist())
    if len(frames) < obs_steps:
        raise ValueError(f"Need at least {obs_steps} frames, got {len(frames)}.")
    frames = frames[-obs_steps:]
    agent_ids = sorted(df["agent_id"].unique().tolist())[:max_agents]
    agent_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    frame_to_idx = {frame: idx for idx, frame in enumerate(frames)}
    data = np.zeros((obs_steps, max_agents, 3), dtype=np.float32)
    sub = df[df["frame"].isin(frames)]
    for row in sub.itertuples(index=False):
        if row.agent_id not in agent_to_idx:
            continue
        data[frame_to_idx[row.frame], agent_to_idx[row.agent_id]] = (
            float(row.x),
            float(row.y),
            float(row.agent_type),
        )
    return torch.from_numpy(data).unsqueeze(0), agent_ids
