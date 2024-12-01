import torch
import torch.nn as nn



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