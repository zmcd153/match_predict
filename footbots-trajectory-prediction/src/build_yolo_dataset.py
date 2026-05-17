from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class FrameItem:
    image_path: Path
    source_name: str
    frame_index: int


def parse_videos(raw: str) -> list[Path]:
    videos = [Path(item.strip()) for item in raw.split(",") if item.strip()]
    if not videos:
        raise ValueError("At least one video path is required.")
    missing = [path for path in videos if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing videos:\n" + "\n".join(f"  - {path}" for path in missing))
    return videos


def extract_frames(video: Path, out_dir: Path, every: int, max_frames: int) -> list[FrameItem]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_idx = 0
    saved = 0
    items: list[FrameItem] = []
    source_name = video.stem.replace(" ", "_")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every == 0:
            name = f"{source_name}_frame_{frame_idx:08d}.jpg"
            image_path = out_dir / name
            cv2.imwrite(str(image_path), frame)
            items.append(FrameItem(image_path=image_path, source_name=source_name, frame_index=frame_idx))
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1
    cap.release()
    print(f"extracted {len(items)} frames from {video}")
    return items


def load_or_extract_frames(videos: list[Path], frames_dir: Path, every: int, max_frames_per_video: int, reuse_frames: bool) -> list[Path]:
    if reuse_frames and frames_dir.exists():
        images = sorted(p for p in frames_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        if images:
            print(f"reusing {len(images)} existing frames from {frames_dir}")
            return images

    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    all_items: list[FrameItem] = []
    for video in videos:
        all_items.extend(extract_frames(video, frames_dir, every=every, max_frames=max_frames_per_video))
    return [item.image_path for item in all_items]


def yolo_line(cls_id: int, xyxy: list[float], width: int, height: int) -> str | None:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 1 or box_h <= 1:
        return None
    x_center = (x1 + x2) / 2 / width
    y_center = (y1 + y2) / 2 / height
    norm_w = box_w / width
    norm_h = box_h / height
    return f"{cls_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"


def auto_label(images: list[Path], labels_dir: Path, model_name: str, conf: float, imgsz: int) -> None:
    import cv2
    from ultralytics import YOLO

    labels_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(model_name)
    total_objects = 0
    total_images_with_objects = 0
    for idx, image in enumerate(images, start=1):
        frame = cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image}")
        height, width = frame.shape[:2]
        result = model.predict(source=str(image), conf=conf, imgsz=imgsz, verbose=False)[0]
        lines: list[str] = []
        for box in result.boxes:
            coco_cls = int(box.cls.item())
            if coco_cls == 0:
                target_cls = 0  # player
            elif coco_cls == 32:
                target_cls = 1  # ball
            else:
                continue
            line = yolo_line(target_cls, box.xyxy.cpu().numpy()[0].tolist(), width, height)
            if line:
                lines.append(line)
        label_path = labels_dir / f"{image.stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_objects += len(lines)
        if lines:
            total_images_with_objects += 1
        if idx % 50 == 0 or idx == len(images):
            print(f"auto-labeled {idx}/{len(images)} images")
    print(f"auto-label summary: {total_objects} boxes on {total_images_with_objects}/{len(images)} images")


def split_dataset(images: list[Path], source_labels_dir: Path, dataset_dir: Path, val_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:val_count])

    for rel in ["images/train", "images/val", "labels/train", "labels/val"]:
        out = dataset_dir / rel
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count_actual = 0
    for image in shuffled:
        split = "val" if image in val_set else "train"
        if split == "train":
            train_count += 1
        else:
            val_count_actual += 1
        shutil.copy2(image, dataset_dir / "images" / split / image.name)
        src_label = source_labels_dir / f"{image.stem}.txt"
        dst_label = dataset_dir / "labels" / split / f"{image.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")

    print(f"dataset written: {dataset_dir}")
    print(f"train images: {train_count}")
    print(f"val images: {val_count_actual}")


def write_data_yaml(dataset_dir: Path) -> None:
    text = "\n".join(
        [
            f"path: {dataset_dir.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: player",
            "  1: ball",
        ]
    )
    (dataset_dir / "data.yaml").write_text(text + "\n", encoding="utf-8")
    print(f"saved yaml: {dataset_dir / 'data.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a YOLO player/ball dataset from videos with automatic pre-labeling.")
    parser.add_argument("--videos", required=True, help="Comma-separated video paths.")
    parser.add_argument("--frames-dir", default="data/auto_frames")
    parser.add_argument("--labels-dir", default="data/auto_labels")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--every", type=int, default=30, help="Save one frame every N frames.")
    parser.add_argument("--max-frames-per-video", type=int, default=300)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model", default="yolov8s.pt", help="COCO YOLO model used for pre-labeling.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--reuse-frames", action="store_true")
    parser.add_argument("--skip-auto-label", action="store_true", help="Use existing labels-dir instead of running YOLO pre-labeling.")
    args = parser.parse_args()

    videos = parse_videos(args.videos)
    frames_dir = Path(args.frames_dir)
    labels_dir = Path(args.labels_dir)
    dataset_dir = Path(args.dataset_dir)

    images = load_or_extract_frames(videos, frames_dir, args.every, args.max_frames_per_video, args.reuse_frames)
    if not images:
        raise RuntimeError("No frames were extracted or found.")

    if labels_dir.exists() and not args.skip_auto_label:
        shutil.rmtree(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_auto_label:
        print(f"using existing labels from {labels_dir}")
    else:
        auto_label(images, labels_dir, model_name=args.model, conf=args.conf, imgsz=args.imgsz)

    split_dataset(images, labels_dir, dataset_dir, val_ratio=args.val_ratio, seed=args.seed)
    write_data_yaml(dataset_dir)
    print("Next: review/correct labels, then run: python -m src.train_yolo --dataset-dir dataset")


if __name__ == "__main__":
    main()