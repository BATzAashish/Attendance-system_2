  **Attendance System Using YOLOv8**


## Introduction
This project implements an **Attendance System using YOLOv8**, where a person's face is detected from an image and matched against a dataset to mark attendance. The system utilizes YOLOv8 for object detection and face recognition.

---

## Dataset Preparation
### 1. Collect Images
Captured 1 image per person.
Applied augmentation techniques to increase the dataset for validation and testing.
Store images in dataset/images/.

### 2. Label Images (Bounding Boxes)
- Use the following script to manually label the images in YOLO format:
```bash
pip install labelImg
labelImg
```
- Draw bounding boxes around faces and save annotations in `dataset/labels/`.

### 3. Verify Labels
Use the following Python script to visualize labeled data:
```python
from ultralytics import YOLO
import cv2

yolo_model = YOLO("yolov8n.yaml")
yolo_model.train(data="dataset.yaml", epochs=1, imgsz=640)
```

---

## Model Training
Train the YOLOv8 model on your dataset:
```bash
yolo task=detect mode=train model=yolov8n.pt data=dataset.yaml epochs=50 imgsz=640
```
This will generate a `runs/train/` folder containing the trained model.

---

## Attendance Detection
### 1. Run the Attendance Script
Create a Python script (`attendance.py`) that loads the trained model and processes images:
```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/train/exp/weights/best.pt")
image = cv2.imread("test_image.jpg")
results = model(image)
results.show()
```

Attendance Detection

1. Detect Faces in Real-Time

After training, use OpenCV with YOLOv8 to detect a person in real-time using a webcam:
import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/train/exp/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame)
    results.show()  # Display detected image
    
    # Process detections (Add logic to store attendance if detected person is recognized)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
