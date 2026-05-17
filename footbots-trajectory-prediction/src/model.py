from __future__ import annotations

import math

import torch
from torch import nn


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
    """Transformer encoder block used as a permutation-equivariant SAB."""

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

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        y, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        y = self.ffn(x)
        return self.norm2(x + self.dropout(y))


class SocioTemporalBlock(nn.Module):
    """Decoupled temporal SABT followed by social SABS."""

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
    """FootBots/TranSPORTmer-style deterministic trajectory predictor.

    The model encodes observed trajectories with alternating temporal and social
    set attention. A learned query for each future step decodes future positions.
    """

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
        last_pos = obs[:, -1:, :, :2]
        pred = last_pos + torch.cumsum(delta, dim=1)
        width, height = self.field_size
        pred_x = pred[..., 0].clamp(0.0, width)
        pred_y = pred[..., 1].clamp(0.0, height)
        return torch.stack([pred_x, pred_y], dim=-1)


def ade(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - target, dim=-1).mean()


def fde(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred[:, -1] - target[:, -1], dim=-1).mean()
