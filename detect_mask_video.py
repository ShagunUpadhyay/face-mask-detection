"""Real-time face mask detection from a webcam.

Usage:
    python detect_mask_video.py
    python detect_mask_video.py --camera 1 --model mask_detector.keras

Press 'q' to quit.
"""
import argparse
import json
import os

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
GREEN, RED = (0, 255, 0), (0, 0, 255)


def parse_args():
    ap = argparse.ArgumentParser(description="Real-time face mask detector")
    ap.add_argument("--model", default="mask_detector.keras")
    ap.add_argument("--labels", default="labels.json")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    return ap.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading mask detector model...")
    model = load_model(args.model)

    # Class order must match training. Fall back to the alphabetical default.
    if os.path.exists(args.labels):
        with open(args.labels) as f:
            class_names = json.load(f)
    else:
        class_names = ["with_mask", "without_mask"]
    mask_idx = class_names.index("with_mask")

    cascade_path = os.path.join(
        cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    print("[INFO] Starting video stream (press 'q' to exit)...")
    vs = cv2.VideoCapture(args.camera)
    if not vs.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    try:
        while True:
            ret, frame = vs.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # mirror view
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            boxes, batch = [], []
            for x, y, fw, fh in faces:
                face_roi = frame[y : y + fh, x : x + fw]
                if face_roi.size == 0:  # guard against empty crops
                    continue
                face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, IMG_SIZE).astype("float32")
                batch.append(preprocess_input(face))
                boxes.append((x, y, fw, fh))

            if batch:
                # one prediction call per frame for all faces
                preds = model.predict(np.array(batch), verbose=0)
                for (x, y, fw, fh), pred in zip(boxes, preds):
                    mask_prob = float(pred[mask_idx])
                    no_mask_prob = 1.0 - mask_prob
                    if mask_prob > no_mask_prob:
                        label, color = f"Mask: {mask_prob * 100:.1f}%", GREEN
                    else:
                        label, color = f"No Mask: {no_mask_prob * 100:.1f}%", RED

                    cv2.putText(
                        frame,
                        label,
                        (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                    cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)

            cv2.imshow("Real-Time Face Mask Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
