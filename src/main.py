import cv2
import numpy as np
from ultralytics import YOLO

# ==============================
# CONFIGURATION
# ==============================

MODEL_PATH = "yolov8n.pt"   # Lightweight model
CONF_THRESHOLD = 0.5
CAMERA_INDEX = 0

# ==============================
# LOAD MODEL
# ==============================

model = YOLO(MODEL_PATH)

# ==============================
# START CAMERA
# ==============================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

intrusion_count = 0

print("Edge AI Safety Sentinel Started...")

# ==============================
# MAIN LOOP
# ==============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # Define Blue Line (Vertical Center Line)
    blue_line_x = width // 2

    # Draw Blue Line
    cv2.line(frame, (blue_line_x, 0), (blue_line_x, height), (255, 0, 0), 3)

    # Run YOLO Detection
    results = model(frame, verbose=False)

    intrusion_detected = False
    fall_detected = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if conf < CONF_THRESHOLD:
                continue

            # Detect only PERSON (class id 0 in COCO)
            if cls_id == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                box_width = x2 - x1
                box_height = y2 - y1

                center_x = (x1 + x2) // 2

                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # ==============================
                # INTRUSION DETECTION
                # ==============================
                if center_x > blue_line_x:
                    intrusion_detected = True

                # ==============================
                # FALL DETECTION (Aspect Ratio Based)
                # ==============================
                if box_width > box_height:
                    fall_detected = True

    # ==============================
    # ALERT OVERLAYS
    # ==============================

    if intrusion_detected:
        intrusion_count += 1
        cv2.rectangle(frame, (0, 0), (width, 60), (0, 0, 255), -1)
        cv2.putText(frame, "WARNING: INTRUSION DETECTED",
                    (50, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

    if fall_detected:
        cv2.rectangle(frame, (0, 70), (width, 130), (0, 0, 255), -1)
        cv2.putText(frame, "CRITICAL ALERT: HUMAN FALL DETECTED",
                    (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2)

    # ==============================
    # ZONE LABELS
    # ==============================

    cv2.putText(frame, "SAFE ZONE",
                (50, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.putText(frame, "RESTRICTED ZONE",
                (blue_line_x + 20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2)

    cv2.putText(frame, f"Intrusions: {intrusion_count}",
                (50, height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)

    # ==============================
    # DISPLAY
    # ==============================

    cv2.imshow("Edge AI Safety & Anomaly Sentinel", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==============================
# CLEANUP
# ==============================

cap.release()
cv2.destroyAllWindows()
