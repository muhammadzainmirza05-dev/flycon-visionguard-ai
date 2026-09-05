import cv2
from ultralytics import YOLO

def start_person_counter(video_source=0, output_filename="people_counter_output.avi"):
    print("Waking up YOLO...")
    model = YOLO("yolov8n.pt")  # sticking with nano for real-time speed

    # Grab the webcam (0) or load a video file
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Uh oh, couldn't open video source: {video_source}")
        return

    # Get camera specs so we can set up the video writer properly
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Webcams sometimes return 0 for fps, fallback to a safe 30
    if not fps or fps == 0:
        fps = 30.0

    # Set up saving to an .avi file
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (w, h))

    print("Live counting started. Hit 'q' on the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or camera disconnected.")
            break

        # Run inference, but restrict it to class 0 (people only)
        # verbose=False stops YOLO from spamming the console every frame
        results = model(frame, classes=[0], verbose=False)
        result = results[0]

        # The number of bounding boxes = number of people detected
        num_people = len(result.boxes)

        # Let ultralytics draw the bounding boxes for us
        annotated_frame = result.plot()

        # Draw a little UI banner at the top for the counter
        ui_text = f"Live People Count: {num_people}"
        
        # Black background box for the text so it's readable
        cv2.rectangle(annotated_frame, (10, 10), (320, 50), (0, 0, 0), -1) 
        
        # Green text
        cv2.putText(annotated_frame, ui_text, (20, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Save frame to disk
        out.write(annotated_frame)

        # Show it on screen
        cv2.imshow("Crowd Counter Test", annotated_frame)

        # waitKey(1) updates the UI, 'q' breaks the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping...")
            break

    # Clean up (out.release() is super important or the video file gets corrupted)
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"Done! Saved the recording to {output_filename}")


if __name__ == "__main__":
    # Test on the webcam by default, change to a file path like 'cctv.mp4' to test a video
    start_person_counter(0)
