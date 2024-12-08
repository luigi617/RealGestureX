import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision.models as models

class StaticGestureCNNModel(nn.Module):
    def __init__(self, num_classes=5, freeze_layers=True):
        super(StaticGestureCNNModel, self).__init__()

        # Load the pre-trained ResNet34 model
        self.resnet34 = models.resnet34(pretrained=True)

        # Modify the final fully connected layer to match the number of classes
        self.resnet34.fc = nn.Linear(self.resnet34.fc.in_features, num_classes)

        # Fine-tuning: Freeze earlier layers and only train the last few layers
        if freeze_layers:
            # Freeze all layers
            for param in self.resnet34.parameters():
                param.requires_grad = False
            
            # Unfreeze the final fully connected layer (classifier)
            for param in self.resnet34.fc.parameters():
                param.requires_grad = True

            # Optionally, unfreeze the deeper layers (e.g., from layer 4 onwards)
            for param in self.resnet34.layer4.parameters():  # Last residual block
                param.requires_grad = True

    def forward(self, x):
        # Forward pass through ResNet34
        return self.resnet34(x)
