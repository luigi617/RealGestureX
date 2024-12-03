import torch
import torch.nn as nn
import torch.nn.functional as F

class StaticGestureCNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(StaticGestureCNNModel, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)  # Input: 128x128x3, Output: 128x128x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 128x128x64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 128x128x128
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # After pooling, image is reduced to 16x16
        self.fc2 = nn.Linear(512, num_classes)
        
        # Define a dropout layer to avoid overfitting
        self.dropout = nn.Dropout(0.5)

        # Define a max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max-pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten the output from the convolutional layers
        x = x.view(-1, 128 * 16 * 16)
        
        # Apply the fully connected layers with dropout and ReLU activation
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x