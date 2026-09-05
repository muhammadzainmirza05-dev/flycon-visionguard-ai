import cv2
from ultralytics import YOLO

# Waking up the nano model for real-time processing speed
print("Loading YOLO...")
model = YOLO('yolov8n.pt')

# Path to the CCTV footage from the previous test
video_path = r"C:\Users\muham\Documents\Codes\Flycon Internship\sample_cctv.mp4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Uh oh, couldn't load the video. Double-check the file path.")
    exit()

print("Running detection... hit 'q' on the video window to stop.")

while True:
    ret, frame = cap.read()
    
    # drop out if we hit the end of the file or it glitches
    if not ret:
        print("Video ended or stream dropped.")
        break

    # Run inference but only look for people (0) and vehicles (2: car, 3: bike, 5: bus, 7: truck)
    # This filters out random stuff like traffic lights or potted plants
    results = model(frame, classes=[0, 2, 3, 5, 7])

    # Let ultralytics handle drawing the bounding boxes and labels
    annotated_frame = results[0].plot()

    cv2.imshow("VisionGuard - Video Detection Test", annotated_frame)

    # waitKey(1) keeps the video playing as fast as YOLO can process it
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
