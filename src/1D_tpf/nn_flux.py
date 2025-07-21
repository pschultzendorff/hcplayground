"""Neural network flux function to be used in homotopy continuation methods."""

import torch
from torch import nn
from torch.utils.data import DataLoader


class FluxNN(nn.Module):
    """Neural network flux function for homotopy continuation."""

    def __init__(self, input_dim, hidden_dim):
        super(FluxNN, self).__init__()
        self.input_dim = input_dim  # 2 saturations, 2 pressures, 2 permeabilities, 2 distances, entry pressure
        self.hidden_dim = hidden_dim
        self.output_dim = 2

        # Define the neural network layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvexLoss(nn.Module):
    """Partial loss function to ensure that the network is convex or concave w.r.t. all
    its outputs."""

    def __init__(self):
        super(ConvexLoss, self).__init__()

    def forward(self, output, target):
        """Compute the convex loss."""
        # Assuming output and target are of shape (batch_size, output_dim)
        loss = torch.mean((output - target) ** 2)
        return loss
