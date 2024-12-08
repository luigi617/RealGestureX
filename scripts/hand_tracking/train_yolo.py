import os
import torch
from ultralytics import YOLO

def main():
    # Define dataset and model paths
    dataset_yaml = os.path.join(os.getcwd(), "scripts", "hand_tracking", "dataset.yaml")
    pretrained_weights = "yolov8s.pt"
    model_save_path = os.path.join(os.getcwd(), "models", "parameters", "hand_tracking_model.pth")
    
    # Initialize the YOLO model
    model = YOLO(pretrained_weights)
    
    # Training hyperparameters
    img_size = 640
    batch_size = 16
    patience = 10
    max_epochs = 1000
    model.train(
        data=dataset_yaml,
        epochs=max_epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,
        save=True,
        save_period=-1
    )
    
    
    # Validate the model after training
    validation_metrics = model.val()
    
    # Print final validation metrics with formatting
    print("Final validation metrics:")
    print(f"mAP: {validation_metrics.box.map:.4f}")
    print(f"Precision: {validation_metrics.box.precision:.4f}")
    print(f"Recall: {validation_metrics.box.recall:.4f}")
    print(f"F1-score: {validation_metrics.box.f1:.4f}")


if __name__ == "__main__":
    main()
