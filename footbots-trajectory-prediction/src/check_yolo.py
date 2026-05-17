from __future__ import annotations

import argparse
from pathlib import Path


KEEP_BY_MODE = {
    "custom": {0: "player", 1: "ball"},
    "coco": {0: "person/player", 32: "sports ball/ball"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Quickly inspect YOLO detections on a video or image source.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--mode", choices=["custom", "coco"], default="custom")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="check_yolo")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    keep = KEEP_BY_MODE[args.mode]
    counts = {class_id: 0 for class_id in keep}
    total = 0
    frames = 0

    results = model.predict(
        source=args.source,
        stream=True,
        imgsz=args.imgsz,
        conf=args.conf,
        save=args.save,
        project=args.project,
        name=args.name,
        verbose=False,
    )
    for result in results:
        frames += 1
        for box in result.boxes:
            cls = int(box.cls.item())
            if cls in keep:
                counts[cls] += 1
                total += 1
        if args.max_frames and frames >= args.max_frames:
            break

    print(f"weights: {args.weights}")
    print(f"source: {args.source}")
    print(f"model names: {model.names}")
    print(f"frames checked: {frames}")
    print(f"kept detections: {total}")
    for class_id, label in keep.items():
        print(f"  class {class_id} ({label}): {counts[class_id]}")
    if args.save:
        print(f"saved visual check under: {Path(args.project) / args.name}")


if __name__ == "__main__":
    main()