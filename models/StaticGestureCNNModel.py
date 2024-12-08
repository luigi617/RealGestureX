import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticGestureCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(StaticGestureCNNModel, self).__init__()
        
        # Convolutional layers with Batch Normalization
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)  # 128x128x16
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # 64x64x32
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32x32x64
        self.bn3 = nn.BatchNorm2d(64)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.5)
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # Output size (1,1)
        
        # Fully connected layer
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 128x128 -> 64x64
        
        # Second convolutional block
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x64 -> 32x32
        
        # Third convolutional block
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 32x32 -> 16x16
        
        # Global Average Pooling
        x = self.global_avg_pool(x)  # 16x16 -> 1x1
        
        # Flatten
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 64)
        
        # Fully connected layer with dropout
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
