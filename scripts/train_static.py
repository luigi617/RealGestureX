# scripts/train_static.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.StaticGestureModel import StaticGestureModel
from utils.utils import preprocess_landmarks
import os
import json
import numpy as np
from tqdm import tqdm

static = ["pointing", "open_palm", "thumb_index_touch", "fist"]

class StaticGestureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (str): Path to the directory with gesture data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.labels = []
        self.transform = transform
        self.classes = static
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        for cls in self.classes:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            json_files = sorted([f for f in os.listdir(cls_dir) if f.endswith('.json')])
            for jf in json_files:
                jf_path = os.path.join(cls_dir, jf)
                with open(jf_path, 'r') as f:
                    landmarks = json.load(f)
                    landmarks = preprocess_landmarks(landmarks).numpy()
                    self.data.append(landmarks)
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.FloatTensor(sample)
        return sample, label

def train_static_gesture_model():
    # Paths
    train_dir = os.getcwd() + '/gesture_dataset/static/train'
    val_dir = os.getcwd() + '/gesture_dataset/static/val'
    # Hyperparameters
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    num_classes = len(static)
    
    # Datasets and Dataloaders
    train_dataset = StaticGestureDataset(train_dir)
    val_dataset = StaticGestureDataset(val_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Model, Loss, Optimizer
    model = StaticGestureModel(input_size=63, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2f}% '
              f'Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.2f}%')
        
        # Save the best model
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), 'models/static_gesture_model.pth')
            print(f'Best model saved with Val Acc: {best_val_acc:.2f}%')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_static_gesture_model()
