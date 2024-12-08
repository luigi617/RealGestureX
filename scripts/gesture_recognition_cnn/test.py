import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

from models.DynamicGestureCNNModel import DynamicGestureCNNModel
from utils.utils import get_device, split_dynamic_data, evaluate
from models.GestureClasses import dynamic

class DynamicGestureDataset(Dataset):
    def __init__(self, data_dir: dict, transform=None, sequence_length=15):
        self.sequence_length = sequence_length
        self.transform = transform
        self.classes = dynamic
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(dynamic)}
        self.samples = []
        
        for cls in self.classes:
            for sample_dir in data_dir[cls]:
                image_files = sorted([
                    os.path.join(sample_dir, img)
                    for img in os.listdir(sample_dir)
                    if img.endswith('.jpg') or img.endswith('.jpeg') or img.endswith('.png')
                ])
                if len(image_files) >= self.sequence_length:
                    self.samples.append((image_files[:self.sequence_length], self.class_to_idx[cls]))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_paths, label = self.samples[idx]
        images = []
        for img_path in image_paths:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.transform:
                image = self.transform(image)
            else:
                image = torch.FloatTensor(image)
            images.append(image)
        images = torch.stack(images)
        return images, label

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Normalized)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('cm.jpg')
    plt.show()


def evaluate_and_get_predictions(model, data_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    return y_true, y_pred

def train_dynamic_gesture_model():
    dynamic_dir = os.path.join(os.getcwd(), 'datasets', 'gesture_dataset_cnn', 'dynamic')
    
    batch_size = 16
    
    num_classes = len(dynamic)
    
    train_data, val_data, test_data = split_dynamic_data(dynamic_dir, dynamic, ".jpg")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = DynamicGestureDataset(train_data, transform=transform, sequence_length=15)
    val_dataset = DynamicGestureDataset(val_data, transform=transform, sequence_length=15)
    test_dataset = DynamicGestureDataset(test_data, transform=transform, sequence_length=15)

    for dataset, name in zip([train_dataset, val_dataset, test_dataset], 
                             ['train', 'val', 'test']):
        if len(dataset) == 0:
            raise ValueError(f"{name} dataset is empty. Check your data split.")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DynamicGestureCNNModel(
        num_classes=num_classes,
        hidden_size=256,
        num_layers=2,
        bidirectional=True,
        freeze_cnn=True
    ).to(device)

    
    # Load the best model and evaluate
    model.load_state_dict(torch.load('models/parameters/dynamic_gesture_cnn_model.pth', map_location=torch.device(device)))
    model.to(device)

    # Get predictions for the test data
    y_true, y_pred = evaluate_and_get_predictions(model, test_loader, device)

    # Plot the confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names=dynamic)

if __name__ == "__main__":
    device = get_device()
    train_dynamic_gesture_model()
