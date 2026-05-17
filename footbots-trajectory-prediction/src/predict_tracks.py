from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from .data import build_latest_window
from .model import TrajectoryTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-agents", type=int, default=23)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    obs_steps = int(checkpoint["obs_steps"])
    pred_steps = int(checkpoint["pred_steps"])
    obs, agent_ids = build_latest_window(args.tracks, obs_steps=obs_steps, max_agents=args.max_agents)
    model = TrajectoryTransformer(obs_steps=obs_steps, pred_steps=pred_steps)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    with torch.no_grad():
        pred = model(obs).squeeze(0).numpy()

    rows = []
    for t in range(pred_steps):
        for idx, agent_id in enumerate(agent_ids):
            rows.append(
                {
                    "future_step": t + 1,
                    "agent_id": agent_id,
                    "x": float(pred[t, idx, 0]),
                    "y": float(pred[t, idx, 1]),
                }
            )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
