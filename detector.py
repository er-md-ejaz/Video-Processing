import os
import cv2
from ultralytics import YOLO
import requests
from datetime import datetime

# -------------------- CONFIG --------------------
MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")   # use n for speed, m/l/x if you want accuracy
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "0")        # "0" for webcam, or a path like "video.mp4"
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:5000/detections")
SOURCE_ID = os.environ.get("SOURCE_ID", "camera_0")
OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "output_yolo.mp4")

# Parse VIDEO_SOURCE: if it's digit-like, cast to int for webcam
try:
    VIDEO_SOURCE = int(VIDEO_SOURCE)
except ValueError:
    pass

# -------------------- YOLO ----------------------
model = YOLO(MODEL_PATH)

# Video capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

# Writer
out = cv2.VideoWriter(
    OUTPUT_FILE,
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (frame_width, frame_height)
)

def send_detections_batch(detections, source=SOURCE_ID):
    if not detections:
        return
    payload = {"source": source, "detections": detections}
    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=3.0)
        if r.status_code not in (200, 201):
            print("Backend error:", r.status_code, r.text)
    except Exception as e:
        # Do not crash the detector if backend is down
        print("Failed to send detections:", e)

def collect_and_send(results):
    batch = []
    # model.names is a dict {class_id: "label"}
    names = model.names if hasattr(model, "names") else {}
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # class id and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names.get(cls_id, str(cls_id))

            # bounding box (xyxy)
            try:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
            except Exception:
                x1 = y1 = x2 = y2 = None

            det = {
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "timestamp": datetime.utcnow().isoformat()
            }
            batch.append(det)

    send_detections_batch(batch)

# -------------------- MAIN LOOP -----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("YOLO Detection", annotated)
    out.write(annotated)

    # push detections to backend
    collect_and_send(results)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
