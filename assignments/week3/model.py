import torch
from typing import Callable
import torch.nn as nn


class MLP(nn.Module):
    """MLP class"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.hidden_count = hidden_count
        self.activation = activation()
        self.initializer = initializer
        self.layers = nn.ModuleList()
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(0.15)
        for _ in range(hidden_count):
            # next_num_inputs = hidden_size
            layer = nn.Linear(input_size, hidden_size)
            layer.weight = initializer(layer.weight)
            layer.bias = torch.nn.Parameter(torch.tensor(0.01))
            self.layers += [layer]
            input_size = hidden_size

        # Create final layer
        self.out = nn.Linear(input_size, num_classes)
        self.out.weight = initializer(self.out.weight)
        self.out.bias = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        # Flatten inputs to 2D (if more than that)
        x = x.flatten(end_dim=-2)

        # Get activations of each layer
        for layer in self.layers:
            x = self.bn(self.activation(layer(x)))
        # x = self.bn(x)
        # Get outputs
        x = self.out(x)

        return x
