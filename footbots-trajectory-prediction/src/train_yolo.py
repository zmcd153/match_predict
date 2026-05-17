from __future__ import annotations

import argparse
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_names(raw: str) -> list[str]:
    names = [item.strip() for item in raw.split(",") if item.strip()]
    if not names:
        raise ValueError("At least one class name is required.")
    return names


def list_images(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def write_data_yaml(dataset_dir: Path, yaml_path: Path, names: list[str]) -> None:
    lines = [
        f"path: {dataset_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_label_file(path: Path, class_count: int) -> list[str]:
    errors: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{path}:{line_no} expected 5 values, got {len(parts)}")
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(item) for item in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_no} contains a non-numeric YOLO value")
            continue
        if class_id < 0 or class_id >= class_count:
            errors.append(f"{path}:{line_no} class id {class_id} is outside 0..{class_count - 1}")
        if any(value < 0.0 or value > 1.0 for value in values):
            errors.append(f"{path}:{line_no} bbox values must be normalized between 0 and 1")
        if values[2] <= 0.0 or values[3] <= 0.0:
            errors.append(f"{path}:{line_no} bbox width/height must be positive")
    return errors


def validate_dataset(dataset_dir: Path, names: list[str]) -> None:
    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        joined = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Dataset is missing required YOLO folders:\n{joined}")

    train_images = list_images(dataset_dir / "images" / "train")
    val_images = list_images(dataset_dir / "images" / "val")
    if not train_images:
        raise ValueError(f"No training images found in {dataset_dir / 'images' / 'train'}")
    if not val_images:
        raise ValueError(f"No validation images found in {dataset_dir / 'images' / 'val'}")

    errors: list[str] = []
    empty_label_files = 0
    for split, images in [("train", train_images), ("val", val_images)]:
        label_dir = dataset_dir / "labels" / split
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                errors.append(f"Missing label for {image}: expected {label}")
                continue
            if label.stat().st_size == 0:
                empty_label_files += 1
            errors.extend(validate_label_file(label, len(names)))

    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        extra = "" if len(errors) <= 20 else f"\n  ... and {len(errors) - 20} more"
        raise ValueError(f"Dataset label validation failed:\n{preview}{extra}")

    print(f"dataset: {dataset_dir}")
    print(f"classes: {', '.join(names)}")
    print(f"train images: {len(train_images)}")
    print(f"val images: {len(val_images)}")
    if empty_label_files:
        print(f"empty label files: {empty_label_files} (allowed, means no objects in those images)")


def train_yolo(args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Install ultralytics first: pip install -r requirements.txt") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device or None,
        workers=args.workers,
        patience=args.patience,
        cache=args.cache,
        exist_ok=args.exist_ok,
        resume=args.resume,
    )
    save_dir = Path(getattr(results, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"training finished, but best.pt was not found at expected path: {best}")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for soccer player/ball detection.")
    parser.add_argument("--dataset-dir", default="dataset", help="YOLO dataset root with images/train, images/val, labels/train, labels/val.")
    parser.add_argument("--names", default="player,ball", help="Comma-separated class names. Default: player,ball.")
    parser.add_argument("--data-yaml", default="", help="Output data.yaml path. Default: <dataset-dir>/data.yaml.")
    parser.add_argument("--model", default="yolov8s.pt", help="Base YOLO model, e.g. yolov8n.pt/yolov8s.pt/yolo11s.pt.")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="Examples: 0, cpu. Empty lets Ultralytics choose.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="soccer_topview")
    parser.add_argument("--cache", action="store_true", help="Cache images during training if RAM/disk allows.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--resume", action="store_true", help="Resume a previous Ultralytics training run.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate dataset and write data.yaml; do not train.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    names = parse_names(args.names)
    args.data_yaml = Path(args.data_yaml).resolve() if args.data_yaml else dataset_dir / "data.yaml"

    validate_dataset(dataset_dir, names)
    write_data_yaml(dataset_dir, args.data_yaml, names)
    print(f"saved yaml: {args.data_yaml}")

    if args.dry_run:
        print("dry run complete; training was not started.")
        return

    best = train_yolo(args)
    print(f"best weights: {best}")
    print("Use this with main.py: --weights <best.pt> --detector-mode custom")


if __name__ == "__main__":
    main()