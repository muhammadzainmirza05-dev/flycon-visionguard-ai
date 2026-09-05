import cv2
from ultralytics import YOLO

# using the nano model since we need decent FPS for a live webcam feed
print("Loading YOLO...")
model = YOLO('yolov8n.pt')

# grab the default laptop webcam
print("Firing up the webcam...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Bummer, couldn't access the webcam. Is another app using it?")
    exit()

# Get camera resolution so we can set up the video writer properly
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 20  # 20 is usually a safe bet for standard webcams

# Setup saving to an .avi file
# XVID codec is pretty reliable for OpenCV stuff
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('webcam_test_output.avi', fourcc, fps, (w, h))

print("Live! Press 'q' on the window to stop recording.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Dropped a frame or camera disconnected. Bailing out...")
        break

    # Run inference
    # Filtering for person (0) and vehicles (2, 3, 5, 7) to keep out background noise
    results = model(frame, classes=[0, 2, 3, 5, 7])

    # let ultralytics do the heavy lifting for drawing the boxes and text
    annotated_frame = results[0].plot()

    # write the frame to disk
    out.write(annotated_frame)

    # show it on screen
    cv2.imshow("Live VisionGuard Feed", annotated_frame)

    # waitKey(1) updates the UI, 'q' breaks the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopping...")
        break

# Cleanup (out.release() is super important here, otherwise the video file gets corrupted)
cap.release()
out.release()
cv2.destroyAllWindows()
