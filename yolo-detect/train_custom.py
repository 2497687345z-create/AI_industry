"""
Custom YOLOv8 training pipeline — train on your own objects.

Quick start:
  1. Collect images into datasets/<name>/images/
  2. Annotate with LabelImg or Roboflow (YOLO format: .txt per image)
  3. Configure datasets/<name>/dataset.yaml
  4. python train_custom.py --name my-objects

Usage:
    python train_custom.py --name my-objects                    # Train
    python train_custom.py --name my-objects --epochs 200       # More epochs
    python train_custom.py --name my-objects --model yolov8s.pt # Larger model
    python train_custom.py --name my-objects --resume           # Resume from last checkpoint
    python train_custom.py --name my-objects --predict img.jpg  # Test on an image after training
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"


def create_dataset_structure(name, classes):
    """Create a new dataset directory with the correct structure."""
    ds_dir = DATASETS_DIR / name
    if ds_dir.exists():
        overwrite = input(f"Dataset '{name}' already exists. Overwrite? [y/N] ")
        if overwrite.lower() != "y":
            return ds_dir

    # Create folders
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (ds_dir / sub).mkdir(parents=True, exist_ok=True)

    # Create dataset.yaml
    yaml_path = ds_dir / "dataset.yaml"
    class_lines = "\n".join(f"  {i}: {c}" for i, c in enumerate(classes))
    yaml_content = f"""# Dataset: {name}
path: {ds_dir.as_posix()}
train: images/train
val: images/val
nc: {len(classes)}
names:
{class_lines}
"""
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"""
Dataset '{name}' created at: {ds_dir}

Next steps:
  1. Put training images into:  {ds_dir / 'images' / 'train'}
  2. Put validation images into: {ds_dir / 'images' / 'val'}
  3. Each image needs a .txt label file next to it in labels/train/ or labels/val/
     Label format (YOLO):  class_id x_center y_center width height
     All values normalized to 0-1.

  Recommended annotation tools:
    - LabelImg:  https://github.com/HumanSignal/labelImg
    - Roboflow:  https://roboflow.com (online, auto-splits train/val)

  Train with:
    python train_custom.py --name {name}
""")
    return ds_dir


def train(ds_name, model_name="yolov8n.pt", epochs=100, imgsz=640, resume=False, device="auto"):
    yaml_path = DATASETS_DIR / ds_name / "dataset.yaml"

    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} not found.")
        print(f"Create it first: python train_custom.py --create {ds_name} --classes class1 class2")
        return

    if resume:
        # Find latest checkpoint
        weights_dir = BASE_DIR / "runs" / "detect"
        checkpoints = sorted(weights_dir.glob(f"*/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not checkpoints:
            print("ERROR: No checkpoint found to resume from.")
            return
        model_path = str(checkpoints[0])
        print(f"Resuming from: {model_path}")
        model = YOLO(model_path)
    else:
        model = YOLO(model_name)

    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=imgsz,
        device=device,
    )
    return results


def predict(ds_name, image_path, model_path=None):
    """Run inference with the trained model."""
    if model_path is None:
        # Auto-find best model
        weights_dir = BASE_DIR / "runs" / "detect"
        candidates = sorted(weights_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("ERROR: No trained model found. Train first with --train.")
            return
        model_path = str(candidates[0])
        print(f"Using model: {model_path}")

    model = YOLO(model_path)
    results = model(image_path)

    # Save and show
    for r in results:
        r.save()  # saved to runs/detect/predict/
        r.show()


def main():
    parser = argparse.ArgumentParser(description="Custom YOLOv8 Training Pipeline")
    parser.add_argument("--name", help="Dataset name (subfolder under datasets/)")
    parser.add_argument("--create", help="Create new dataset structure with this name")
    parser.add_argument("--classes", nargs="+", help="Class names for --create, e.g. --classes screw bolt nut")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base model: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default 640)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--predict", help="Run inference on an image using the trained model")
    parser.add_argument("--predict-video", help="Run inference on a video")
    parser.add_argument("--device", default="auto", help="Device: auto, 0, cpu")
    args = parser.parse_args()

    # --create mode
    if args.create:
        if not args.classes:
            print("ERROR: --create needs --classes. Example: --create my-objects --classes screw bolt")
            return
        create_dataset_structure(args.create, args.classes)
        return

    if not args.name:
        parser.print_help()
        return

    # --predict mode
    if args.predict:
        predict(args.name, args.predict)
        return

    # --predict-video mode (webcam = webcam)
    if args.predict_video:
        model_path = None
        weights_dir = BASE_DIR / "runs" / "detect"
        candidates = sorted(weights_dir.glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            model_path = str(candidates[0])
        else:
            print("ERROR: No trained model found.")
            return

        model = YOLO(model_path)
        if args.predict_video == "webcam":
            model.predict(source=0, show=True)
        else:
            model.predict(source=args.predict_video, show=True)
        return

    # Default: train
    train(args.name, args.model, args.epochs, args.imgsz, args.resume, args.device)


if __name__ == "__main__":
    main()
