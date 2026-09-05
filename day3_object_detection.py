import cv2
from ultralytics import YOLO

# using the nano model here for quicker testing
print("Booting up YOLO...")
model = YOLO('yolov8n.pt')

# my test image
img_path = r"C:\Users\muham\Documents\Codes\Flycon Internship\video and images\object detection.jpg"

print("Running detection...")
results = model(img_path)

# let ultralytics handle drawing the bounding boxes and labels
annotated_img = results[0].plot()

# display the result
cv2.imshow("YOLOv8 Test - Flycon", annotated_img)

print("Done. Press any key in the window to close.")
cv2.waitKey(0)
cv2.destroyAllWindows()
