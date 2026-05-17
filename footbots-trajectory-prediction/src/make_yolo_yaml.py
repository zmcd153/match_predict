from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a YOLO dataset yaml for player/ball detection.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--names", default="player,ball")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    out = Path(args.out) if args.out else dataset_dir / "data.yaml"
    names = [name.strip() for name in args.names.split(",") if name.strip()]
    text = [
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    text.extend([f"  {idx}: {name}" for idx, name in enumerate(names)])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
