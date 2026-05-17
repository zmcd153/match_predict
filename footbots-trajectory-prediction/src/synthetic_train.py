from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .model import TrajectoryTransformer, ade, fde

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x


def make_synthetic(n: int, obs_steps: int, pred_steps: int, agents: int = 23) -> tuple[torch.Tensor, torch.Tensor]:
    total = obs_steps + pred_steps
    rng = np.random.default_rng(7)
    xy = rng.uniform([5.0, 5.0], [100.0, 63.0], size=(n, 1, agents, 2)).astype(np.float32)
    velocity = rng.normal(0.0, 0.35, size=(n, 1, agents, 2)).astype(np.float32)
    agent_type = np.zeros((n, total, agents, 1), dtype=np.float32)
    agent_type[:, :, 0, 0] = 0
    agent_type[:, :, 1 : 1 + (agents - 1) // 2, 0] = 1
    agent_type[:, :, 1 + (agents - 1) // 2 :, 0] = 2
    positions = []
    ball = xy[:, :, :1]
    for t in range(total):
        attraction = (ball - xy) * 0.012
        team_bias = np.where(agent_type[:, :1, :, 0] == 1, 0.03, -0.02).astype(np.float32)
        bias_vec = np.zeros_like(velocity)
        bias_vec[..., 0] = team_bias
        velocity = 0.94 * velocity + attraction + bias_vec
        if t == obs_steps:
            velocity[:, :, 0] += rng.normal(0.0, 1.2, size=(n, 1, 2)).astype(np.float32)
        xy = xy + velocity + rng.normal(0.0, 0.04, size=xy.shape).astype(np.float32)
        xy[..., 0] = np.clip(xy[..., 0], 0.0, 105.0)
        xy[..., 1] = np.clip(xy[..., 1], 0.0, 68.0)
        ball = xy[:, :, :1]
        positions.append(xy.copy())
    pos = np.concatenate(positions, axis=1)
    full = np.concatenate([pos, agent_type], axis=-1)
    return torch.from_numpy(full[:, :obs_steps]), torch.from_numpy(pos[:, obs_steps:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--obs-steps", type=int, default=20)
    parser.add_argument("--pred-steps", type=int, default=40)
    parser.add_argument("--out", default="runs/synthetic.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, target = make_synthetic(args.samples, args.obs_steps, args.pred_steps)
    loader = DataLoader(TensorDataset(obs, target), batch_size=args.batch_size, shuffle=True)
    model = TrajectoryTransformer(obs_steps=args.obs_steps, pred_steps=args.pred_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_obs, batch_target in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            batch_obs = batch_obs.to(device)
            batch_target = batch_target.to(device)
            pred = model(batch_obs)
            loss = ade(pred, batch_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            pred = model(obs[:64].to(device))
            val_ade = ade(pred, target[:64].to(device)).item()
            val_fde = fde(pred, target[:64].to(device)).item()
        print(f"loss={np.mean(losses):.3f} val_ade={val_ade:.3f} val_fde={val_fde:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "obs_steps": args.obs_steps,
            "pred_steps": args.pred_steps,
        },
        args.out,
    )
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
