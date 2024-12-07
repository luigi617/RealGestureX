import os
from ultralytics import YOLO
import os



dataset_yaml = os.getcwd()+"/scripts/hand_tracking/dataset.yaml"
pretrained_weights = "yolov8s.pt"
model = YOLO(pretrained_weights)
epochs = 50
img_size = 640
batch_size = 16

results = model.train(
    data=dataset_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    project="models/runs",
    name="exp_hand_detection",
    exist_ok=True
)

validation_metrics = model.val()

print("Final validation metrics:")
print(f"mAP: {validation_metrics['metrics']['mAP']}")
print(f"Precision: {validation_metrics['metrics']['precision']}")
print(f"Recall: {validation_metrics['metrics']['recall']}")
print(f"F1-score: {validation_metrics['metrics']['f1']}")
