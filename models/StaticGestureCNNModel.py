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

