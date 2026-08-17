import cv2

# Step 1: Open the video file
video_path = r"C:\Users\muham\Documents\Codes\Flycon Internship\sample_cctv.mp4.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(
        f"Error: Could not open video file '{video_path}'. Ensure the file exists in the folder."
    )
    exit()

# Step 2: Extract video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 40)
print(" VisionGuard AI - Video Properties")
print("=" * 40)
print(f"Resolution   : {width} x {height}")
print(f"FPS          : {fps:.2f}")
print(f"Total Frames : {total_frames}")
print("=" * 40)
print("Playing video... Press 'q' on the video window to quit.")

# Step 3: Read and display video frame-by-frame
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()

    # If ret is False, the video has reached the end or failed to load
    if not ret:
        print("\nEnd of video stream or cannot read the frame.")
        break

    frame_number += 1

    # Display frame metadata on screen
    overlay_text = f"Frame: {frame_number}/{total_frames} | Res: {width}x{height} | FPS: {int(fps)}"
    cv2.putText(
        frame,
        overlay_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Show the frame in a GUI window
    cv2.imshow("VisionGuard AI - CCTV Feed", frame)

    # Press 'q' to exit early (30ms delay approximates real-time video speed)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        print("\nPlayback interrupted by user.")
        break

# Step 4: Release resources cleanly
cap.release()
cv2.destroyAllWindows()
print("Video processing complete. Windows closed safely.")