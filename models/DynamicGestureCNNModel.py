import torch
import torch.nn as nn
import torchvision.models as models

class DynamicGestureCNNModel(nn.Module):
    def __init__(self, num_classes=5, hidden_size=64, num_layers=2, bidirectional=True, freeze_cnn=True):
        super(DynamicGestureCNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        self.freeze_cnn = freeze_cnn
        if freeze_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        self.feature_size = 512
        
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        
        self.fc1 = nn.Linear(self.hidden_size * self.num_directions, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape

        x = x.view(batch_size * seq_length, c, h, w)
        
        with torch.set_grad_enabled(not self.freeze_cnn):
            cnn_features = self.resnet(x)  
        cnn_features = cnn_features.view(batch_size, seq_length, self.feature_size)  
        
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(cnn_features, (h0, c0))  
        
        out = out[:, -1, :]  
        
        out = self.fc1(out)  
        out = self.relu(out)
        out = self.fc2(out)  
        
        return out  
