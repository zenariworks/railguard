from ultralytics import YOLO

# Load dataset and train model
model = YOLO("yolov8n.pt")  # Load YOLOv8 model
model.train(data="data/processed/data.yaml", epochs=50) 