from ultralytics import YOLO

model = YOLO("models/custom_model/best.pt")
results = model.val()
