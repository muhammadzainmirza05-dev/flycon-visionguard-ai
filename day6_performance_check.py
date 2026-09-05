import time
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt') # using the nano model so we don't fry the CPU

# Only tracking people and vehicles (0: person, 2: car, 3: bike, 5: bus, 7: truck)
ALLOWED_CLASSES = [0, 2, 3, 5, 7]

# set to 0 for webcam, or throw in a file path like 'sample_cctv.mp4'
vid_source = 0
cap = cv2.VideoCapture(vid_source)

if not cap.isOpened():
    print(f"Bummer, couldn't open video source {vid_source}")
    exit()

# arrays to hold our timing stats
io_times = []
infer_times = []
draw_times = []
total_times = []

frames_processed = 0
MAX_FRAMES = 150 # limit the test so we don't sit here all day (~5 secs at 30fps)

print("\nStarting performance benchmark...")
print("Hit 'q' on the video window if you want to bail out early.\n")

while cap.isOpened() and frames_processed < MAX_FRAMES:
    t_loop_start = time.time()

    # --- 1. Camera I/O ---
    t_start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Dropped frame or end of video.")
        break
    io_times.append((time.time() - t_start) * 1000) # save as ms

    # --- 2. YOLO Inference ---
    t_start = time.time()
    # verbose=False keeps the console from getting spammed with frame logs
    results = model(frame, classes=ALLOWED_CLASSES, verbose=False)
    infer_times.append((time.time() - t_start) * 1000)

    # --- 3. Drawing / UI ---
    t_start = time.time()
    annotated_frame = results[0].plot()

    # calc instantaneous FPS
    t_loop_end = time.time()
    frame_time = t_loop_end - t_loop_start
    total_times.append(frame_time)

    fps = 1.0 / frame_time if frame_time > 0 else 0

    # slap the FPS on the top left
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

    draw_times.append((time.time() - t_start) * 1000)

    cv2.imshow("Benchmarking YOLO Speed", annotated_frame)
    frames_processed += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

# cleanup
cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
# Let's see what was slowing us down
# -------------------------------------------------------------
if total_times:
    avg_io = np.mean(io_times)
    avg_infer = np.mean(infer_times)
    avg_draw = np.mean(draw_times)
    avg_total = np.mean(total_times)
    avg_fps = 1.0 / avg_total if avg_total > 0 else 0

    report = (
        f"\n--- BENCHMARK RESULTS ---\n"
        f"Frames tested: {frames_processed}\n"
        f"Average FPS: {avg_fps:.2f}\n\n"
        f"Time breakdown (per frame):\n"
        f" - Camera/IO read: {avg_io:.2f} ms\n"
        f" - YOLO Inference: {avg_infer:.2f} ms\n"
        f" - Drawing labels: {avg_draw:.2f} ms\n"
        f" - Total cycle:    {avg_total * 1000:.2f} ms\n\n"
        f"Bottleneck Analysis:\n"
    )

    # Figure out the main bottleneck
    if avg_infer > avg_io and avg_infer > avg_draw:
        report += " -> INFERENCE is the slowest part.\n"
        report += " -> TODO: Try TensorRT, OpenVINO, or just lower the image resolution."
    elif avg_io > avg_infer:
        report += " -> CAMERA I/O is the slowest part.\n"
        report += " -> TODO: Need to put cv2.read() on a separate background thread."
    else:
        report += " -> DRAWING/UI is the slowest part.\n"
        report += " -> TODO: Skip cv2.imshow if we end up running this on a headless server."

    print(report)

    # save it to disk for the internship logs
    with open("benchmark_results.txt", "w") as f:
        f.write(report)
    
    print("\nSaved report to benchmark_results.txt")
