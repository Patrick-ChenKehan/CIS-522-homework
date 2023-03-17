import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """class for model"""

    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 8, 3)
        # torch.nn.init.xavier_normal_(self.conv1.weight)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        self.fc1 = nn.Linear(16 * 6 * 6, 64)
        # torch.nn.init.xavier_normal_(self.fc1.weight)
        # self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(64, num_classes)
        # torch.nn.init.xavier_normal_(self.fc3.weight)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for the model"""
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment

# Comment
