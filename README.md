VisionGuard AI — Flycon AI Internship (Weeks 1 & 2)
Alhumdulillah, I've successfully completed the first two weeks of my Flycon AI Internship! This repository holds all the code for my main project: VisionGuard AI.

I built this AI-powered CCTV surveillance system from scratch, taking heavy inspiration from enterprise security software like Avigilon Control Center. (If you want to see my daily progress and how I built the foundational skills for this, check out the "Internship week 1" logs).

🛠️ Tech Stack & Tools I Used
Programming Language: Python 3.11+

Computer Vision: OpenCV (opencv-python)

Deep Learning Framework: PyTorch & Ultralytics YOLOv8 (specifically the lightweight yolov8n.pt model)

Environment: Python Virtual Environment (yolo_env)

🚀 What I Built (Pipeline Architecture)
Week 1: Getting the Foundation & Real-Time Processing Down
Video Frame Ingestion: I wrote a custom script to ingest sample CCTV-style video and process it frame-by-frame using OpenCV.

Smart Class Filtering: To make the system efficient, I tweaked the YOLOv8 pipeline to ignore irrelevant objects. It is now strictly locked onto target surveillance classes: people and vehicles.

Bonus Challenge (Live Person Counter): I challenged myself to build a dynamic, on-screen counter that updates in real-time based on how many people are currently in the camera's view. You can find this logic in my day7_live_person_counter.py script.

Week 2: Advanced Tracking & Security Zones
Continuous Object Tracking: I integrated YOLOv8's native ByteTrack engine (model.track()). The system now assigns persistent IDs to people, cars, and bags, tracking them smoothly across frames rather than just detecting them once.

ROI (Region of Interest) Zones: I programmed a custom, polygon-based restricted area overlay directly onto the video feed.

Intrusion Alerts: I built mathematical logic to calculate the bottom-center coordinates of tracked objects (basically, where their feet or tires touch the ground). If that coordinate crosses into my restricted ROI zone, the system instantly triggers a red visual alert and a UI warning banner.

Real-Time Metadata Logging: To make this ready for a backend REST API in the future, I set up the system to stream structured JSON logs in real-time, capturing track IDs, class names, bounding box coordinates, and their current ROI status.

📊 Performance Benchmarking
Since I was running this on local CPU hardware, I wanted to see how far I could push the optimization. I logged my testing results in day6_performance_log.txt:

Model Inference Time: ~73.61 ms per frame (which I am honestly pretty happy with for a CPU!).

Overall Speed: Averaged around 12.39 FPS during live tracking.

📂 What's Inside? (File Structure)
Here is a quick map of the repo so you can navigate my code:

day2_read_video.py: My initial script for handling video file ingestion and frame rendering.

day4_video_detection.py: The basic YOLOv8 detection script focusing just on people and vehicles.

day6_performance_log.txt: My benchmarking logs measuring FPS, frame read times, and inference latency.

day7_live_person_counter.py: The script where I implemented the live on-screen counting logic.

week2_visionguard_pipeline.py: The main project file. This combines everything—ROI tracking, intrusion alerts, structured JSON logging, and the ByteTrack engine.

.gitignore: Standard setup to keep the repo clean from my heavy yolo_env virtual environment and Python cache files.

⚙️ How to Run My Pipeline
If you want to test the full pipeline on your own machine, here is how to get it set up:

Open your command prompt/terminal and activate the virtual environment:

Bash
yolo_env\Scripts\activate
(If you haven't already) Install the required dependencies:

Bash
pip install ultralytics opencv-python torch
Run the main week 2 pipeline script:

Bash
python week2_visionguard_pipeline.py
