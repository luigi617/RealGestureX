import torch
import torch.nn as nn



class StaticGestureModel(nn.Module):
    def __init__(self, input_size=63, num_classes=5):
        super(StaticGestureModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
     
    def forward(self, x):
        out = self.fc(x)
        return out
