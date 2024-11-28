import torch
import torch.nn as nn

# class DynamicGestureModel(nn.Module):
#     def __init__(self, num_classes=4, hidden_size=128, num_layers=2):
#         """
#         Args:
#             num_classes (int): Number of dynamic gesture classes.
#             hidden_size (int): Number of features in the hidden state of LSTM.
#             num_layers (int): Number of recurrent layers in LSTM.
#         """
#         super(DynamicGestureModel, self).__init__()
        
#         # CNN for spatial feature extraction
#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # Input: 3x224x224
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Output: 32x112x112
            
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Output: 64x56x56
            
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Output: 128x28x28
#         )
        
#         # Calculate the size of CNN output
#         self.feature_size = 128 * 28 * 28
        
#         # LSTM for temporal feature extraction
#         self.lstm = nn.LSTM(
#             input_size=self.feature_size,
#             hidden_size=hidden_size,
#             num_layers=num_layers,
#             batch_first=True,
#             bidirectional=True
#         )
        
#         # Fully connected layer for classification
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size * 2, 512),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, num_classes)
#         )
    
#     def forward(self, x):
#         """
#         Forward pass.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, seq_len, C, H, W).

#         Returns:
#             torch.Tensor: Output logits of shape (batch_size, num_classes).
#         """
#         batch_size, seq_len, C, H, W = x.size()
#         # Reshape to process each frame through CNN
#         x = x.view(batch_size * seq_len, C, H, W)
#         x = self.cnn(x)  # (batch * seq_len, 128, 28, 28)
#         x = x.view(x.size(0), -1)  # (batch * seq_len, feature_size)
        
#         # Reshape for LSTM
#         x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, feature_size)
#         lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size*2)
        
#         # Use the output from the last time step
#         out = lstm_out[:, -1, :]  # (batch, hidden_size*2)
#         out = self.fc(out)        # (batch, num_classes)
#         return out



class DynamicGestureModel(nn.Module):
    def __init__(self, input_size=63, hidden_size=64, num_layers=2, num_classes=5, bidirectional=True):
        super(DynamicGestureModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        # No need for softmax if using CrossEntropyLoss

    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(x.device)
        c0 = torch.zeros(
            self.num_layers * self.num_directions, x.size(0), self.hidden_size
        ).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size * num_directions)

        # Decode the hidden state of the last time step
        out = out[:, -1, :]  # Shape: (batch_size, hidden_size * num_directions)
        out = self.fc1(out)   # Shape: (batch_size, 64)
        out = self.relu(out)
        out = self.fc2(out)   # Shape: (batch_size, num_classes)
        return out  # Outputs raw logits