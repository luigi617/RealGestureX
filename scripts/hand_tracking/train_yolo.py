import os
from ultralytics import YOLO
import os


# Step 1: Define Paths
# Adjust these paths based on your directory structure
dataset_yaml = os.getcwd()+"/scripts/hand_tracking/dataset.yaml"
pretrained_weights = "yolov8s.pt"  # choose a YOLOv8 model variant: yolov8n.pt, yolov8s.pt, etc.

# Step 2: Initialize Model
# If you have a pretrained model, you can load it here. Otherwise, start from a yolov8 model checkpoint.
model = YOLO(pretrained_weights)

# Step 3: Set Training Parameters
epochs = 50          # how many epochs to train
img_size = 640        # input image size
batch_size = 16       # batch size depends on your GPU memory

# Step 4: Train the Model
# The 'train' method will automatically read the dataset.yaml for data paths and class mappings
results = model.train(
    data=dataset_yaml,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    project="models/runs",    # where to save training results
    name="exp_hand_detection",    # name of the training run
    exist_ok=True                 # allow overwriting existing runs
)

# Step 5: Evaluate the Model (optional)
# After training, you can validate on the validation set:
validation_metrics = model.val()

# Step 6: Inference with the Trained Model
# Load the best weights after training and run inference on a test image
model = YOLO(os.path.join("models", "parameters", "hand_tracking_model.pt"))

# # Perform inference on a single image
# results = model.predict(source="path_to_test_image.jpg", imgsz=img_size, conf=0.25)
# for box in results[0].boxes:
#     print(f"Class: {box.cls}, Confidence: {box.conf}, BoundingBox: {box.xyxy}")

# # Step 7 (Optional): Export Model
# # You can export the model to ONNX or another format if needed:
# model.export(format="onnx")
