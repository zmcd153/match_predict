from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw tracking CSV and interpolate short gaps.")
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--max-gap", type=int, default=10)
    parser.add_argument("--field-width", type=float, default=105.0)
    parser.add_argument("--field-height", type=float, default=68.0)
    args = parser.parse_args()

    df = pd.read_csv(args.tracks)
    required = {"frame", "agent_id", "agent_type", "x", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track CSV is missing columns: {sorted(missing)}")
    df = df[(df["x"].between(0, args.field_width)) & (df["y"].between(0, args.field_height))]
    counts = df.groupby("agent_id")["frame"].nunique()
    keep_ids = counts[counts >= args.min_length].index
    df = df[df["agent_id"].isin(keep_ids)].copy()

    frames = range(int(df["frame"].min()), int(df["frame"].max()) + 1)
    cleaned = []
    for agent_id, group in df.groupby("agent_id"):
        group = group.sort_values("frame").drop_duplicates("frame", keep="last")
        agent_type = int(group["agent_type"].mode().iloc[0])
        indexed = group.set_index("frame").reindex(frames)
        indexed["agent_id"] = agent_id
        indexed["agent_type"] = agent_type
        indexed[["x", "y"]] = indexed[["x", "y"]].interpolate(limit=args.max_gap, limit_area="inside")
        indexed = indexed.dropna(subset=["x", "y"]).reset_index().rename(columns={"index": "frame"})
        cleaned.append(indexed[["frame", "agent_id", "agent_type", "x", "y"]])

    out_df = pd.concat(cleaned, ignore_index=True) if cleaned else pd.DataFrame(columns=["frame", "agent_id", "agent_type", "x", "y"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"saved {len(out_df)} cleaned observations to {args.out}")


if __name__ == "__main__":
    main()
