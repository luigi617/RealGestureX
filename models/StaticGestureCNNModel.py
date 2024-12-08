
import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticGestureCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(StaticGestureCNNModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # 128x128x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 64x64x32
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 32 * 32, 128)  # After two poolings: 32x32
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))  # 128x128 -> 64x64
        
        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))  # 64x64 -> 32x32
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 32*32*32)
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
