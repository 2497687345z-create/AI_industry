"""
Finger / hand detection using MediaPipe Tasks (new API) + YOLOv8.

Usage:
    python finger_detect.py                        # webcam
    python finger_detect.py --image photo.jpg      # single image
"""

import argparse
import cv2
import time
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from ultralytics import YOLO

# Auto-download the hand landmarker model if not present
MODEL_PATH = Path(__file__).parent / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return str(MODEL_PATH)


# Finger landmark indices in MediaPipe
FINGER_TIPS = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky tips
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Drawing connections for hand skeleton
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def count_raised_fingers(landmarks):
    """Count how many fingers are raised based on landmark positions."""
    tips_up = []
    # Thumb: compare x with its IP joint
    tips_up.append(landmarks[4].x < landmarks[3].x if landmarks[4].x < landmarks[2].x else landmarks[4].x > landmarks[3].x)
    # Other 4 fingers: tip y < PIP joint y (tip is above PIP)
    for tip_id in [8, 12, 16, 20]:
        tips_up.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)
    return sum(tips_up)


def draw_hand_landmarks(frame, landmarks, w, h):
    """Draw hand landmarks and connections manually (compatible with new API)."""
    points = []
    for lm in landmarks:
        px, py = int(lm.x * w), int(lm.y * h)
        points.append((px, py))

    # Draw connections
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

    # Draw landmarks
    for i, (px, py) in enumerate(points):
        radius = 3 if i in FINGER_TIPS else 2
        cv2.circle(frame, (px, py), radius, (0, 0, 255), -1)

    # Count fingers
    count = count_raised_fingers(landmarks)
    cx, cy = points[0]
    cv2.putText(frame, f"Fingers: {count}", (cx, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


def draw_hands(frame, hand_result, yolo_model=None, conf=0.25):
    """Draw hand landmarks and optionally run YOLO."""
    h, w = frame.shape[:2]

    if hand_result and hand_result.hand_landmarks:
        for hand_lm in hand_result.hand_landmarks:
            draw_hand_landmarks(frame, hand_lm, w, h)

    # Optionally run YOLO for objects
    if yolo_model is not None:
        results_yolo = yolo_model(frame, conf=conf, verbose=False)
        for result in results_yolo:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                name = result.names[int(box.cls[0])]
                if name.lower() not in {"mouse", "book", "keyboard", "cell phone"}:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{name} {conf_val:.2f}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame


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


def detect_webcam(landmarker, yolo_model, conf):
    cap = _open_camera(0) or _open_camera(1)
    if cap is None:
        print("ERROR: Could not open any camera.")
        print("Check that:")
        print("  1. A camera is connected")
        print("  2. No other app is using the camera")
        print("  3. Camera privacy settings allow access")
        return

    print("Press 'q' to quit.")
    fps = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Timestamp for video mode (milliseconds)
        ts_ms = int(time.time() * 1000)
        hand_result = landmarker.detect_for_video(mp_image, ts_ms)

        frame = draw_hands(frame, hand_result, yolo_model, conf)

        # FPS display
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Finger + Object Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def detect_image(landmarker, yolo_model, image_path, conf):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Cannot read image: {image_path}")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    hand_result = landmarker.detect(mp_image)

    frame = draw_hands(frame, hand_result, yolo_model, conf)

    cv2.imshow("Finger + Object Detection", frame)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Finger + Object Detection")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO, hands only")
    parser.add_argument("--model", help="Path to custom hand landmarker model")
    args = parser.parse_args()

    model_path = args.model or ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE if args.image else vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    yolo_model = None if args.no_yolo else YOLO("yolov8n.pt")

    if args.image:
        detect_image(landmarker, yolo_model, args.image, args.conf)
    else:
        detect_webcam(landmarker, yolo_model, args.conf)


if __name__ == "__main__":
    main()
