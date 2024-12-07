import os
from ultralytics import YOLO
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main():
    # Define dataset and model paths
    dataset_yaml = os.path.join(os.getcwd(), "scripts", "hand_tracking", "dataset.yaml")
    pretrained_weights = "yolov8s.pt"
    
    # Initialize the YOLO model
    model = YOLO(pretrained_weights)
    
    # Training hyperparameters
    epochs = 50
    img_size = 640
    batch_size = 16
    
    # Define the EarlyStopping callback
    early_stop_callback = EarlyStopping(
        monitor='val/mAP',    # Metric to monitor; ensure this matches the validation metric name
        patience=10,           # Number of epochs to wait for improvement
        verbose=True,         # Enable verbose logging
        mode='max'            # 'max' since higher mAP is better
    )
    
    # Train the model with early stopping
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project="models/runs",
        name="exp_hand_detection",
        exist_ok=True,
        callbacks=[early_stop_callback]  # Include the EarlyStopping callback
    )
    
    # Validate the model after training
    validation_metrics = model.val()
    
    # Print final validation metrics with formatting
    print("Final validation metrics:")
    print(f"mAP: {validation_metrics['metrics']['mAP']:.4f}")
    print(f"Precision: {validation_metrics['metrics']['precision']:.4f}")
    print(f"Recall: {validation_metrics['metrics']['recall']:.4f}")
    print(f"F1-score: {validation_metrics['metrics']['f1']:.4f}")

if __name__ == "__main__":
    main()
