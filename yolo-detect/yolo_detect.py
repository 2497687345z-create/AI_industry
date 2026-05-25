"""
YOLOv8 object detection — supports all 80 COCO classes.

Usage:
    python yolo_detect.py                              # webcam, default 7 classes
    python yolo_detect.py --all-classes                # webcam, all 80 COCO classes
    python yolo_detect.py --classes dog cat person     # webcam, only dog/cat/person
    python yolo_detect.py --image photo.jpg            # single image
    python yolo_detect.py --video clip.mp4             # video file
    python yolo_detect.py --list-classes               # print all supported classes
"""

import argparse
import cv2
from ultralytics import YOLO

# All 80 COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# Default classes for quick start
DEFAULT_CLASSES = {"mouse", "book", "keyboard", "cell phone", "laptop", "cup", "person"}

# Colors for bounding boxes (BGR)
COLORS = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
    (128, 0, 255), (0, 128, 255),
]


def draw_boxes(frame, results, target_classes=None, show_all=False):
    """Draw bounding boxes. Only draw target_classes in color, others gray or skipped."""
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if 0 <= cls_id < len(COCO_CLASSES):
                name = COCO_CLASSES[cls_id]
            else:
                name = result.names.get(cls_id, f"class_{cls_id}")

            label = name.lower()

            if target_classes is not None and label not in target_classes:
                if not show_all:
                    continue
                color = (128, 128, 128)
            else:
                color = COLORS[cls_id % len(COLORS)]

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def detect_image(model, image_path, target_classes, show_all, conf):
    """Run detection on a single image and display it."""
    results = model(image_path, conf=conf)
    img = draw_boxes(results[0].orig_img, results, target_classes, show_all)
    cv2.imshow("YOLOv8 Detection", img)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def _open_camera(camera_id=0):
    """Open camera with the correct backend for the platform."""
    backends = [
        (cv2.CAP_DSHOW, "DShow"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "Default"),
    ]
    for backend, name in backends:
        try:
            cap = cv2.VideoCapture(camera_id, backend)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Camera {camera_id} opened with {name} backend.")
                    return cap
                cap.release()
        except Exception:
            continue
    return None


def detect_webcam(model, target_classes, show_all, conf):
    """Real-time detection from webcam."""
    cap = _open_camera(0) or _open_camera(1)
    if cap is None:
        print("ERROR: Could not open any camera.")
        return

    if target_classes:
        print(f"Detecting: {', '.join(target_classes)}")
    else:
        print("Detecting all 80 COCO classes.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)
        frame = draw_boxes(frame, results, target_classes, show_all)

        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_video(model, video_path, target_classes, show_all, conf):
    """Run detection on a video file."""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, conf=conf, verbose=False)
        frame = draw_boxes(frame, results, target_classes, show_all)
        cv2.imshow("YOLOv8 Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection - 80 COCO classes")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--video", help="Path to a video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--model", default="yolov8n.pt", help="Model: yolov8n.pt, yolov8s.pt, etc.")
    parser.add_argument("--all-classes", action="store_true",
                        help="Show all 80 COCO classes (default: only 7 common ones)")
    parser.add_argument("--classes", nargs="+", default=None,
                        help="Only detect these classes, e.g. --classes dog cat person")
    parser.add_argument("--list-classes", action="store_true",
                        help="Print all 80 supported COCO classes and exit")
    args = parser.parse_args()

    if args.list_classes:
        print("80 COCO classes supported by YOLOv8:")
        for i, name in enumerate(COCO_CLASSES):
            print(f"  {i:2d}. {name}")
        return

    model = YOLO(args.model)

    # Determine target classes
    if args.classes:
        target_classes = set(c.lower() for c in args.classes)
        # Validate
        all_coco_lower = set(COCO_CLASSES)
        for c in target_classes:
            if c not in all_coco_lower:
                print(f"Warning: '{c}' is not a COCO class. Use --list-classes to see all.")
    elif args.all_classes:
        target_classes = None  # show everything
    else:
        target_classes = DEFAULT_CLASSES

    show_all = target_classes is None

    if args.image:
        detect_image(model, args.image, target_classes, show_all, args.conf)
    elif args.video:
        detect_video(model, args.video, target_classes, show_all, args.conf)
    else:
        detect_webcam(model, target_classes, show_all, args.conf)


if __name__ == "__main__":
    main()
