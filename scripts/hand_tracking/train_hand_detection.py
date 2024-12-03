import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision.ops import box_iou

from models.HandDetectionModel import HandDetectionModel
from models.HandLandmarkTrackingModel import HandLandmarkModel
from utils.utils import calculate_mae, compute_average_iou, split_data



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HandDetectionDataset(Dataset):
    def __init__(self, data_dir, image_transform=None):
        self.image_paths = data_dir["image"]      # Path to the images directory
        self.bbox_paths = data_dir["hand"]        # Path to the bounding boxes directory

        assert len(self.image_paths) == len(self.bbox_paths), \
            "Mismatch between the number of images and bounding boxes."
        
        self.image_transform = image_transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image
        image = cv2.imread(self.image_paths[idx])  # Read image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        # Load the bounding box
        bbox = self.load_bbox(self.bbox_paths[idx])

        if self.image_transform:
            image = Image.fromarray(image)
            image = self.image_transform(image)

        target = {}
        target["boxes"] = torch.tensor([bbox], dtype=torch.float32)  # [N, 4]
        target["labels"] = torch.tensor([1], dtype=torch.int64)     
        
        return image, target
    
    def load_bbox(self, bbox_path):
        with open(bbox_path, 'r') as f:
            # Assuming bbox format: x_min y_min x_max y_max
            bbox = list(map(int, f.read().strip().split()))
        return bbox

def train_hand_detection(model, train_loader, val_loader, optimizer, num_epochs=10, patience=5):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for images, targets in loop:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            optimizer.zero_grad()

            outputs = model(images, targets)
            print(outputs.keys())
            loss = sum(loss for loss in outputs.values())
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # Validation Loss and IoU Metric
        val_loss = 0.0
        all_pred_boxes = []
        all_gt_boxes = []
        model.eval()
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}', leave=False)
            for images, targets in loop:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

                # Compute validation loss
                outputs = model(images, targets)
                print(outputs.keys())
                loss = sum(loss for loss in outputs.values())
                val_loss += loss.item()

                # Perform inference to get predictions
                detections = model(images)  # Outputs without targets

                for det, tgt in zip(detections, targets):
                    pred_boxes = det['boxes'].cpu()
                    # Filter out predictions with low scores if necessary
                    # e.g., pred_boxes = pred_boxes[det['scores'] > 0.5]
                    all_pred_boxes.append(pred_boxes)
                    gt_boxes = tgt['boxes'].cpu()
                    all_gt_boxes.append(gt_boxes)

        avg_val_loss = val_loss / len(val_loader)
        avg_iou = compute_average_iou(all_pred_boxes, all_gt_boxes)
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_iou:.4f}")

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter if validation loss improves
            torch.save(model.state_dict(), 'models/parameters/hand_detection_model.pth')
            print(f"Model saved at epoch {epoch+1}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        model.train()  # Switch back to training mode

    print("Training completed!")


def collate_fn(batch):
    return tuple(zip(*batch))


image_transform = transforms.Compose([
    transforms.Resize((320, 320)),      # Resize image to 128x128
    transforms.ToTensor(),              # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
])



data_dir= 'hand_tracking_dataset'

num_epochs = 1000
batch_size = 16
learning_rate = 1e-3
patience = 20  # Early stopping patience

train_data, val_data, test_data = split_data(data_dir, ["image", "hand", "landmark"], [".jpg", ".txt", ".json"])
# Instantiate dataset
train_dataset = HandDetectionDataset(data_dir=train_data, image_transform=image_transform)
val_dataset = HandDetectionDataset(data_dir=val_data, image_transform=image_transform)
test_dataset = HandDetectionDataset(data_dir=test_data, image_transform=image_transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)



hand_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
num_classes = 2  # Background + hand
in_features = hand_detection_model.roi_heads.box_predictor.cls_score.in_features
hand_detection_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
hand_detection_model.to(device)

params = [p for p in hand_detection_model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)



train_hand_detection(hand_detection_model, train_loader, val_loader, optimizer, num_epochs, patience)

