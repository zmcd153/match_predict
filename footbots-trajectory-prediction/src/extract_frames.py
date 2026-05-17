from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames for manual annotation.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--every", type=int, default=30, help="Save one frame every N frames.")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every == 0:
            path = out_dir / f"frame_{frame_idx:08d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
            if args.max_frames and saved >= args.max_frames:
                break
        frame_idx += 1
    cap.release()
    print(f"saved {saved} frames to {out_dir}")


if __name__ == "__main__":
    main()
