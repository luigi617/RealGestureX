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
    
    best_mAP = 0
    wait = 0
    patience = 10
    max_epochs = 1000

    for epoch in range(max_epochs):
        # Train for one epoch
        model.train(data=dataset_yaml, epochs=1, imgsz=img_size, batch=batch_size)

        # Validate model
        val_metrics = model.val()
        current_mAP = val_metrics.box.map

        # Check for improvement
        if current_mAP > best_mAP:
            best_mAP = current_mAP
            wait = 0
            torch.save(model.model.state_dict(), model_save_path)
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break

    
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
