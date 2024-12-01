import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from models.DynamicGestureModel import DynamicGestureModel
from utils.utils import evaluate, preprocess_landmarks, split_data
import os
import json
import numpy as np
from tqdm import tqdm
from models.GestureClasses import static, dynamic


class DynamicGestureDataset(Dataset):
    def __init__(self, data_dir, sequence_length=30, transform=None):
        """
        Args:
            data_dir (str): Path to the directory with gesture data.
            sequence_length (int): Number of frames per gesture sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = []
        self.labels = []
        self.classes = list(data_dir.keys())
        self.transform = transform
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.sequence_length = sequence_length
        
        for cls in self.classes:
            for jf_path in data_dir[cls]:
                with open(jf_path, 'r') as f:
                    seq_landmarks = []
                    landmarks = json.load(f)
                    for landmark in landmarks:
                        processed_landmark = preprocess_landmarks(landmark).numpy()
                        seq_landmarks.append(processed_landmark)
                    self.data.append(seq_landmarks)
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]  # List of np.array of shape (63,)
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        else:
            sample = torch.from_numpy(np.array(sample))  # Shape: (seq_len, 63)
        return sample, label

def train_dynamic_gesture_model():
    # Paths
    static_dir = os.getcwd() + '/gesture_dataset/dynamic'
    
    # Hyperparameters
    num_epochs = 1000
    batch_size = 64
    learning_rate = 1e-3
    num_classes = len(dynamic)
    hidden_size = 128
    num_layers = 2
    patience = 20  # Early stopping patience

    train_data, val_data, test_data = split_data(static_dir, dynamic)
    
    # Datasets and Dataloaders
    transform = None
    train_dataset = DynamicGestureDataset(train_data, transform=transform)
    val_dataset = DynamicGestureDataset(val_data, transform=transform)
    test_dataset = DynamicGestureDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    model = DynamicGestureModel(num_classes=num_classes, hidden_size=hidden_size, num_layers=num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0  # Counter for early stopping

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for sequences, labels in loop:
            sequences, labels = sequences.to(device), labels.to(device)
            
            # Forward
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * sequences.size(0)
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
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = 100 * val_correct / val_total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.2f}% '
              f'Val Loss: {val_epoch_loss:.4f} Val Acc: {val_epoch_acc:.2f}%')
        
        # Early stopping check
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            epochs_without_improvement = 0  # Reset counter
            # Save the best model
            torch.save(model.state_dict(), 'models/parameters/dynamic_gesture_model.pth')
            print(f'Best model saved with Val Acc: {best_val_acc:.2f}%')
        else:
            epochs_without_improvement += 1
        
        # Check if early stopping should be triggered
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {patience} epochs with no improvement.')
            break
    
    # Load the best model and evaluate
    model.load_state_dict(torch.load('models/parameters/dynamic_gesture_model.pth'))
    model.to(device)
    train_acc = evaluate(model, train_loader, device)
    
    # Evaluate on Validation Set
    val_acc = evaluate(model, val_loader, device)
    
    # Evaluate on Test Set
    test_acc = evaluate(model, test_loader, device)

    print("\nFinal Accuracies of the Best Model:")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dynamic_gesture_model()
