import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision import transforms

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


def train_dynamic_gesture_model():
    dynamic_dir = os.path.join(os.getcwd(), 'datasets', 'gesture_dataset_cnn', 'dynamic')
    
    num_epochs = 1000
    batch_size = 16
    learning_rate = 1e-3
    patience = 10
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DynamicGestureCNNModel(
        num_classes=num_classes,
        hidden_size=256,
        num_layers=2,
        bidirectional=True,
        freeze_cnn=True
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    
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
            os.makedirs('models/parameters', exist_ok=True)
            torch.save(model.state_dict(), 'models/parameters/dynamic_gesture_cnn_model.pth')
            print(f'Best model saved with Val Acc: {best_val_acc:.2f}%')
        else:
            epochs_without_improvement += 1
        
        # Check if early stopping should be triggered
        if epochs_without_improvement >= patience:
            print(f'Early stopping triggered after {patience} epochs with no improvement.')
            break
    
    # Load the best model and evaluate
    model.load_state_dict(torch.load('models/parameters/dynamic_gesture_cnn_model.pth', map_location=torch.device(device)))
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
    train_dynamic_gesture_model()
