import torch
import torch.nn as nn
import torch.nn.functional as F



class StaticGestureCNNModel(nn.Module):
    def __init__(self, num_classes=26):  # Adjust num_classes based on your dataset
        super(StaticGestureCNNModel, self).__init__()
        self.features = nn.Sequential(
            # Convolutional Layer 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels=3 (RGB), Output=32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 112 x 112

            # Convolutional Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Output=64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 56 x 56

            # Convolutional Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Output=128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 28 x 28

            # Convolutional Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Output=256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Output: 256 x 14 x 14

            # You can add more layers if needed
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 14 * 14, 512),  # Adjust based on the output size from conv layers
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x
    
import torch
import torch.nn as nn
import torchvision.models as models

class StaticGestureCNNModel(nn.Module):
    def __init__(self, num_classes=26, freeze_layers=True):
        super(StaticGestureCNNModel, self).__init__()

        self.resnet = models.resnet18(pretrained=True)

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False 

            for param in self.resnet.fc.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.resnet(x)

