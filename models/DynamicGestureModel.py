import torch
import torch.nn as nn

class DynamicGestureModel(nn.Module):
    def __init__(self, num_classes=4, hidden_size=128, num_layers=2):
        """
        Args:
            num_classes (int): Number of dynamic gesture classes.
            hidden_size (int): Number of features in the hidden state of LSTM.
            num_layers (int): Number of recurrent layers in LSTM.
        """
        super(DynamicGestureModel, self).__init__()
        
        # CNN for spatial feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: 3x224x224
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 32x112x112
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 64x56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: 128x28x28
        )
        
        # Calculate the size of CNN output
        self.feature_size = 128 * 28 * 28
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, C, H, W).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        batch_size, seq_len, C, H, W = x.size()
        # Reshape to process each frame through CNN
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn(x)  # (batch * seq_len, 128, 28, 28)
        x = x.view(x.size(0), -1)  # (batch * seq_len, feature_size)
        
        # Reshape for LSTM
        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, feature_size)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
        # Use the output from the last time step
        out = lstm_out[:, -1, :]  # (batch, hidden_size*2)
        out = self.fc(out)        # (batch, num_classes)
        return out
