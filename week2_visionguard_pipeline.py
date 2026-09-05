import cv2
import json
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# Using nano for speed, might need to upgrade to yolov8s if accuracy drops
MODEL = 'yolov8n.pt'

# COCO classes we actually care about: person, vehicles, and bags
ALLOWED_CLASSES = [0, 2, 3, 5, 7, 24, 26, 28]

# Map them to readable names (grouping all bag types together for simplicity)
CLASS_MAP = {
    0: "Person", 2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck",
    24: "Bag", 26: "Bag", 28: "Bag"
}

# TODO: Make this dynamic or load from config file later. 
# Hardcoded for my current camera angle.
roi_pts = np.array([
    [150, 120],
    [500, 120],
    [500, 420],
    [150, 420]
], np.int32)

def check_inside_roi(pt, poly):
    # returns true if the point is inside the polygon bounds
    return cv2.pointPolygonTest(poly, pt, False) >= 0

def start_pipeline(source=0):
    print("Booting up YOLOv8...")
    model = YOLO(MODEL)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Bummer, couldn't open camera/video at '{source}'")
        return

    frame_count = 0
    print("Feed is live. Hit 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream dropped or ended.")
            break

        frame_count += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Run tracking (confidence at 0.45 seems to avoid most ghost detections)
        results = model.track(frame, persist=True, classes=ALLOWED_CLASSES, conf=0.45, verbose=False)
        boxes = results[0].boxes

        is_intruder = False
        log_data = []

        # Default zone color (green)
        zone_color = (0, 255, 0)

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                # unpack coords
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = CLASS_MAP.get(cls, "Unknown")
                conf = float(box.conf[0])
                
                # YOLO sometimes drops the ID, fallback to -1
                t_id = int(box.id[0]) if box.id is not None else -1

                # We use the bottom center of the bounding box to check if they are actually *standing* in the zone
                foot_point = (int((x1 + x2) / 2), int(y2))

                in_zone = check_inside_roi(foot_point, roi_pts)
                if in_zone:
                    is_intruder = True

                # Pack object data for the JSON log
                log_data.append({
                    "track_id": t_id,
                    "class": label,
                    "confidence": round(conf, 2),
                    "in_zone": in_zone,
                    "bbox": [x1, y1, x2, y2]
                })

                # Draw bounding boxes and tracking dots
                color = (0, 0, 255) if in_zone else (255, 165, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, foot_point, 4, (0, 255, 255), -1) # mark the feet

                text = f"ID:{t_id} {label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Trigger alert visuals if someone stepped inside
        if is_intruder:
            zone_color = (0, 0, 255)  # turn polygon red
            
            # Big red banner at the top
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 255), -1)
            cv2.putText(frame, "🚨 INTRUSION DETECTED 🚨", (30, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Dump to console (TODO: hook this up to a REST API later)
            payload = {
                "frame": frame_count,
                "time": now,
                "alert": True,
                "objects": log_data
            }
            print(f"ALERT: {json.dumps(payload)}")

        # Draw the actual zone polygon
        cv2.polylines(frame, [roi_pts], isClosed=True, color=zone_color, thickness=3)
        cv2.putText(frame, "SECURE ZONE", (roi_pts[0][0], roi_pts[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

        cv2.imshow("Security Feed - Live", frame)

        # 'q' to bail out
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change to a filepath like 'test_video.mp4' to test locally without a webcam
    start_pipeline(0)
