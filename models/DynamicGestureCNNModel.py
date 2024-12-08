
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class DynamicGestureModel(nn.Module):
    def __init__(self, num_classes=5, hidden_size=64, num_layers=2, bidirectional=True):
        super(DynamicGestureModel, self).__init__()
        
        # Pre-trained CNN for feature extraction (e.g., ResNet18)
        resnet = models.resnet18(pretrained=True)
        # Remove the fully connected layer and adapt output size for our needs
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        # This gives us a feature map of size (batch_size, 512) for each image
        self.feature_size = 512
        
        # LSTM layer for sequential modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_size,  # ResNet's output feature size
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        
        # Fully connected layers for final gesture classification
        self.fc1 = nn.Linear(hidden_size * 2, 64)  # *2 for bidirectional LSTM
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
        cnn_features = []
        
        # Extract features using CNN (ResNet)
        for i in range(seq_length):
            frame = x[:, i, :, :, :]  # (batch_size, c, h, w) for one frame
            feature = self.resnet(frame).view(batch_size, -1)  # (batch_size, 512)
            cnn_features.append(feature)
        
        # Stack CNN features across all frames (seq_length)
        cnn_features = torch.stack(cnn_features, dim=1)  # Shape: (batch_size, seq_length, 512)
        
        # Pass the features through LSTM for temporal modeling
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Bidirectional LSTM
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(cnn_features, (h0, c0))  # (batch_size, seq_length, hidden_size * 2)
        
        # Use the final hidden state from the LSTM (last timestep)
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size * 2)
        
        # Classification layers
        out = self.fc1(out)  # (batch_size, 64)
        out = self.relu(out)
        out = self.fc2(out)  # (batch_size, num_classes)
        
        return out  # Raw logits (CrossEntropyLoss will apply softmax internally)

