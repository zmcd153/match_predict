from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export one video frame for homography point calibration.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, default=0, help="Frame index to export.")
    parser.add_argument("--out", default="runs/calibration_frame.jpg")
    args = parser.parse_args()

    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {args.frame} from {args.video}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    height, width = frame.shape[:2]
    print(f"saved: {out}")
    print(f"image size: width={width}, height={height}")
    if total:
        print(f"video frames: {total}")
    print("Open this image and record at least 4 pitch points as pixel [x, y] coordinates.")


if __name__ == "__main__":
    main()