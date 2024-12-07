import os
import sys
import torch  # Import torch for saving the model
from ultralytics import YOLO

def main():
    try:
        # Define dataset and model paths
        current_dir = os.getcwd()
        dataset_yaml = os.path.join(current_dir, "scripts", "hand_tracking", "dataset.yaml")
        pretrained_weights = "yolov8s.pt"
        model_save_dir = os.path.join(current_dir, "models", "parameters")
        model_save_path = os.path.join(model_save_dir, "hand_tracking_model.pth")  # Changed to .pth

        # Ensure the save directory exists
        os.makedirs(model_save_dir, exist_ok=True)

        # Check if dataset YAML exists
        if not os.path.exists(dataset_yaml):
            print(f"Dataset YAML file not found at {dataset_yaml}")
            sys.exit(1)

        # Initialize the YOLO model
        model = YOLO(pretrained_weights)

        # Training hyperparameters
        img_size = 640
        batch_size = 16
        max_epochs = 1000
        patience = 10  # Early stopping patience

        best_mAP = 0
        wait = 0

        # Determine device
        device = "cuda" if model.device.type == "cuda" else "cpu"
        print(f"Using device: {device}")

        for epoch in range(1, max_epochs + 1):
            print(f"Epoch {epoch}/{max_epochs}")

            # Train for one epoch
            model.train(data=dataset_yaml, epochs=1, imgsz=img_size, batch=batch_size, device=device, verbose=False)

            # Validate model
            val_metrics = model.val()
            current_mAP = val_metrics.box.map

            print(f"Validation mAP: {current_mAP:.4f}")

            # Check for improvement
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                wait = 0
                # Save the best model's state_dict as a .pth file
                torch.save(model.model.state_dict(), model_save_path)
                print(f"New best mAP: {best_mAP:.4f}. Model saved to {model_save_path}.")
                
            else:
                wait += 1
                print(f"No improvement. Wait count: {wait}/{patience}")
                if wait >= patience:
                    print("Early stopping triggered.")
                    break

        # Final validation
        validation_metrics = model.val()
        
        # Print final validation metrics with formatting
        print("\nFinal Validation Metrics:")
        print(f"mAP: {validation_metrics.box.map:.4f}")
        print(f"Precision: {validation_metrics.box.precision:.4f}")
        print(f"Recall: {validation_metrics.box.recall:.4f}")
        print(f"F1-score: {validation_metrics.box.f1:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
