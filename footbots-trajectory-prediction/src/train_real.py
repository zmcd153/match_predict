from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from .data import MaskedTrackWindowDataset
from .model import TrajectoryTransformer

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **_: x


def masked_ade(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dist = torch.linalg.norm(pred - target, dim=-1)
    denom = mask.sum().clamp_min(1.0)
    return (dist * mask).sum() / denom


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the trajectory transformer from real tracked trajectories.")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out", default="runs/real.pt")
    parser.add_argument("--obs-steps", type=int, default=20)
    parser.add_argument("--pred-steps", type=int, default=40)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-agents", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--min-visible-ratio", type=float, default=0.75)
    args = parser.parse_args()

    dataset = MaskedTrackWindowDataset(
        args.tracks,
        obs_steps=args.obs_steps,
        pred_steps=args.pred_steps,
        stride=args.stride,
        max_agents=args.max_agents,
        min_visible_ratio=args.min_visible_ratio,
    )
    if len(dataset) < 2:
        raise RuntimeError("Not enough valid windows. Lower --min-visible-ratio or collect more tracking data.")

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
        losses = []
        for obs, target, mask in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            obs = obs.to(device)
            target = target.to(device)
            mask = mask.to(device)
            pred = model(obs)
            loss = masked_ade(pred, target, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for obs, target, mask in val_loader:
                obs = obs.to(device)
                target = target.to(device)
                mask = mask.to(device)
                val_losses.append(masked_ade(model(obs), target, mask).item())
        train_loss = float(np.mean(losses))
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
            print(f"saved best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
