import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.HandDetectionModel import HandDetectionModel
from models.HandLandmarkTrackingModel import HandLandmarkModel
from utils.utils import calculate_iou, calculate_mae, split_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class HandTrackingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = data_dir["image"]      # Path to the images directory
        self.bbox_paths = data_dir["hand"]        # Path to the bounding boxes directory
        self.landmark_paths = data_dir["landmark"] # Path to the landmarks directory

        assert len(self.image_paths) == len(self.bbox_paths) == len(self.landmark_paths), \
            "Mismatch between the number of images, bounding boxes, and landmarks files."
        
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image
        image = cv2.imread(self.image_paths[idx])  # Read image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Load the bounding box
        bbox = self.load_bbox(self.bbox_paths[idx])

        # Load the landmark
        landmarks = self.load_landmark(self.landmark_paths[idx])

        # Crop the image using the bounding box (hand detection)
        x_min, y_min, x_max, y_max = bbox
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # Resize image for consistency (e.g., 128x128)
        cropped_image = cv2.resize(cropped_image, (128, 128))

        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        return cropped_image, torch.tensor(bbox, dtype=torch.float), torch.tensor(landmarks, dtype=torch.float)
    
    def load_bbox(self, bbox_path):
        with open(bbox_path, 'r') as f:
            # Assuming bbox format: x_min y_min x_max y_max
            bbox = list(map(float, f.read().strip().split()))
        return bbox
    
    def load_landmark(self, landmark_path):
        with open(landmark_path, 'r') as f:
            # Assuming landmark format: { "landmarks": [[x1, y1, z1], [x2, y2, z2], ...] }
            data = json.load(f)
            landmarks = np.array(data).flatten()  # Flatten to a 1D array of length 63 (21 landmarks * 3 coordinates)
        return landmarks
    

def train_hand_detection(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_iou_train = 0.0  # Track total IoU for training
        total_samples = 0

        for images, bboxes, _ in train_loader:
            images = images.to(device)  # Move images to GPU
            bboxes = bboxes.to(device)  # Move bounding boxes to GPU

            optimizer.zero_grad()
            outputs = model(images)  # Predicted bounding boxes
            
            # Calculate loss (using MSE loss or other loss)
            loss = criterion(outputs, bboxes)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate IoU for each prediction in the batch
            batch_iou = 0.0
            for pred_bbox, true_bbox in zip(outputs, bboxes):
                iou = calculate_iou(pred_bbox.detach().cpu().numpy(), true_bbox.detach().cpu().numpy())
                batch_iou += iou
            total_iou_train += batch_iou
            total_samples += len(images)

        avg_train_loss = running_loss / len(train_loader)
        avg_iou_train = total_iou_train / total_samples  # Average IoU for the training set
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train IoU: {avg_iou_train:.4f}")

        # Validation Loss and IoU Metric
        val_loss = 0.0
        total_iou_val = 0.0
        val_samples = 0
        model.eval()
        with torch.no_grad():
            for images, bboxes, _ in val_loader:
                images = images.to(device)  # Move images to GPU
                bboxes = bboxes.to(device)  # Move bounding boxes to GPU

                outputs = model(images)  # Predicted bounding boxes
                loss = criterion(outputs, bboxes)
                val_loss += loss.item()

                # Calculate IoU for validation batch
                batch_iou = 0.0
                for pred_bbox, true_bbox in zip(outputs, bboxes):
                    iou = calculate_iou(pred_bbox.detach().cpu().numpy(), true_bbox.detach().cpu().numpy())
                    batch_iou += iou
                total_iou_val += batch_iou
                val_samples += len(images)

        avg_val_loss = val_loss / len(val_loader)
        avg_iou_val = total_iou_val / val_samples  # Average IoU for the validation set
        print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val IoU: {avg_iou_val:.4f}")

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter if validation loss improves
            torch.save(model.state_dict(), 'models/parameters/hand_detection_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
# Training Loop for Landmark Detection with Early Stopping and Accuracy Reporting
def train_landmark_detection(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=5):
    model.train()
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_accuracy = 0
        for images, _, landmarks in train_loader:
            images = images.to(device)  # Move images to GPU
            landmarks = landmarks.to(device)  # Move bounding boxes to GPU

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy for landmarks (Mean Absolute Error)
            accuracy = calculate_mae(outputs, landmarks)
            total_accuracy += accuracy.item()

        # Validation Loss and Early Stopping
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for images, _, landmarks in val_loader:
                images = images.to(device)  # Move images to GPU
                landmarks = landmarks.to(device)  # Move bounding boxes to GPU

                outputs = model(images)
                loss = criterion(outputs, landmarks)
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter if validation loss improves
            torch.save(model.state_dict(), 'models/parameters/hand_landmark_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


transform = transforms.Compose([
    transforms.ToTensor(),              # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Example normalization
])

# data_dir= os.getcwd() + 'hand_tracking_dataset'
data_dir= 'hand_tracking_dataset'

num_epochs = 1000
batch_size = 64
learning_rate = 1e-3
patience = 20  # Early stopping patience

train_data, val_data, test_data = split_data(data_dir, ["image", "hand", "landmark"], [".jpg", ".txt", ".json"])
# Instantiate dataset
train_dataset = HandTrackingDataset(data_dir=train_data, transform=transform)
val_dataset = HandTrackingDataset(data_dir=val_data, transform=transform)
test_dataset = HandTrackingDataset(data_dir=test_data, transform=transform)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



hand_detection_model = HandDetectionModel().to(device)
bbox_criterion = nn.MSELoss()
bbox_optimizer = optim.Adam(hand_detection_model.parameters(), lr=0.001)


hand_landmark_model = HandLandmarkModel().to(device)
landmark_criterion = nn.MSELoss()
landmark_optimizer = optim.Adam(hand_landmark_model.parameters(), lr=0.001)

train_hand_detection(hand_detection_model, train_loader, val_loader, bbox_criterion, bbox_optimizer, num_epochs, patience)
train_landmark_detection(hand_landmark_model, train_loader, val_loader, landmark_criterion, landmark_optimizer, num_epochs, patience)
