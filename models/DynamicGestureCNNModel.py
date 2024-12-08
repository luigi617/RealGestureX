import torch
import torch.nn as nn
import torchvision.models as models

class DynamicGestureCNNModel(nn.Module):
    def __init__(self, num_classes=5, hidden_size=64, num_layers=2, bidirectional=True, freeze_cnn=True):
        super(DynamicGestureCNNModel, self).__init__()
        
        # Store parameters as instance variables
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1
        
        # Pre-trained CNN for feature extraction (e.g., ResNet18)
        resnet = models.resnet18(pretrained=True)
        # Remove the fully connected layer and adapt output size for our needs
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Output: (batch_size, 512, 1, 1)
        
        # Optionally freeze CNN weights
        self.freeze_cnn = freeze_cnn
        if freeze_cnn:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # This gives us a feature map of size (batch_size, 512) for each image
        self.feature_size = 512
        
        # LSTM layer for sequential modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,  # ResNet's output feature size
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        
        # Fully connected layers for final gesture classification
        self.fc1 = nn.Linear(self.hidden_size * self.num_directions, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        """
        x shape: (batch_size, seq_length, channels, height, width)
        - batch_size: Number of sequences in the batch
        - seq_length: Number of frames in each sequence
        - channels: Number of channels in the image (usually 3 for RGB)
        - height, width: The image dimensions (e.g., 128x128)
        """
        batch_size, seq_length, c, h, w = x.shape
        
        # Reshape to (batch_size * seq_length, c, h, w) to process all frames at once
        x = x.view(batch_size * seq_length, c, h, w)
        
        # Extract features using CNN (ResNet)
        with torch.set_grad_enabled(not self.freeze_cnn):
            cnn_features = self.resnet(x)  # Shape: (batch_size * seq_length, 512, 1, 1)
        cnn_features = cnn_features.view(batch_size, seq_length, self.feature_size)  # Shape: (batch_size, seq_length, 512)
        
        # Pass the features through LSTM for temporal modeling
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(cnn_features, (h0, c0))  # Shape: (batch_size, seq_length, hidden_size * num_directions)
        
        # Use the final hidden state from the LSTM (last timestep)
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size * num_directions)
        
        # Classification layers
        out = self.fc1(out)  # (batch_size, 64)
        out = self.relu(out)
        out = self.fc2(out)  # (batch_size, num_classes)
        
        return out  # Raw logits (CrossEntropyLoss will apply softmax internally)
