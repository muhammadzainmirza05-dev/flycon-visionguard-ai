import cv2

# Load up the sample footage (note: check if the .mp4.mp4 extension was a typo later)
video_path = r"C:\Users\muham\Documents\Codes\Flycon Internship\sample_cctv.mp4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Failed to open {video_path}. Is the path correct?")
    exit()

# Grab some basic video stats for the overlay
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Loaded video: {w}x{h} at {fps:.2f} FPS. Total frames: {total_frames}")
print("Starting feed... hit 'q' to quit.")

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()

    # drop out if we hit the end of the file
    if not ret:
        print("End of stream.")
        break

    frame_idx += 1

    # Put some debug stats on the top left of the video
    info_text = f"Frame: {frame_idx}/{total_frames} | Res: {w}x{h} | FPS: {int(fps)}"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("CCTV Test Feed", frame)

    # 30ms wait keeps it playing at roughly normal speed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        print("Playback stopped.")
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
