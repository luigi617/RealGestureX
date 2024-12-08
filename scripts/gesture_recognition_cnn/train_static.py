import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms

from models.StaticGestureCNNModel import StaticGestureCNNModel
from utils.utils import get_device, split_data, evaluate
from models.GestureClasses import static

class StaticGestureDataset(Dataset):
    def __init__(self, data_dir:dict, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.classes = static
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(static)}
        for cls in self.classes:
            for image_path in data_dir[cls]:
                self.image_paths.append(image_path)
                self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image)
        return image, label

def train_static_gesture_model():
    static_dir = os.getcwd() + '/datasets/gesture_dataset_cnn/static'
    
    num_epochs = 1000
    batch_size = 16
    learning_rate = 1e-3
    num_classes = len(static)
    patience = 10
    
    train_data, val_data, test_data = split_data(static_dir, static, ".jpg")
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = StaticGestureDataset(train_data, transform=transform)
    val_dataset = StaticGestureDataset(val_data, transform=transform)
    test_dataset = StaticGestureDataset(test_data, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = StaticGestureCNNModel(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = 100 * correct / total
        
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
        
        # Early stopping check
        if val_epoch_acc > best_val_acc:
            best_val_acc = val_epoch_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), 'models/parameters/static_gesture_cnn_model.pth')
            print(f'Best model saved with Val Acc: {best_val_acc:.2f}%')
        else:
            epochs_without_improvement += 1
        
        # Check if early stopping should be triggered
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {patience} epochs with no improvement.')
            break
    
    model.load_state_dict(torch.load('models/parameters/static_gesture_cnn_model.pth', map_location=torch.device(device)))
    model.to(device)
    train_acc = evaluate(model, train_loader, device)
    val_acc = evaluate(model, val_loader, device)
    test_acc = evaluate(model, test_loader, device)

    print("\nFinal Accuracies of the Best Model:")
    print(f"Training Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    device = get_device()
    train_static_gesture_model()
