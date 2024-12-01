import torch
import torch.nn as nn

class HandDetectionModel(nn.Module):
    def __init__(self):
        super(HandDetectionModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # Pooling layer to downsample
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # Pooling layer to downsample
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # Pooling layer to downsample
        
        # Update the input size for fc1 based on the downsampling
        self.fc1 = nn.Linear(128 * 80 * 80, 1024)  # New flattened size
        self.fc2 = nn.Linear(1024, 4)  # Output (x_min, y_min, x_max, y_max)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        
        # Flatten the tensor to prepare for the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten
        
        # Apply fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
