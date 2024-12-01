import torch
import torch.nn as nn

class HandLandmarkModel(nn.Module):
    def __init__(self, input_size=(128, 128)):
        super(HandLandmarkModel, self).__init__()
        
        self.input_size = input_size  # input size of the image (height, width)
        
        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # Calculate the output size after all the convolutions and pooling
        self._calculate_fc_input_size()

        # Define the fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, 1024)  # Adjust the size based on calculation
        self.fc2 = nn.Linear(1024, 63)  # Output 21 landmarks * 3 coordinates (x, y, z)
        
    def _calculate_fc_input_size(self):
        """
        Calculate the size of the input to the fully connected layer
        by passing a dummy tensor through the convolutional and pooling layers.
        """
        # Create a dummy input tensor with the specified input size
        dummy_input = torch.zeros(1, 3, self.input_size[0], self.input_size[1])
        
        # Pass through the conv layers
        x = torch.relu(self.conv1(dummy_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten the tensor to determine the size of the input to fc1
        self.fc_input_size = x.numel()  # Number of elements in the flattened tensor

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
