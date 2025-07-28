"""Neural network flux function to be used in homotopy continuation methods."""

import torch
from torch import nn
from torch.utils.data import DataLoader


class FluxNN(nn.Module):
    """Neural network flux function for homotopy continuation."""

    def __init__(self, input_dim, hidden_dim):
        super(FluxNN, self).__init__()
        self.input_dim = input_dim  # 2 saturations, 2 pressures
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


class DistanceLoss(nn.Module):
    """Distance loss function to ensure that the network outputs are close to the
    target values."""

    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, output, target):
        """Compute the distance loss."""
        # Assuming output and target are of shape (batch_size, output_dim)
        loss = torch.mean(torch.abs(output - target))
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function that includes both convex and distance losses."""

    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.convex_loss = ConvexLoss()
        self.distance_loss = DistanceLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, output, target):
        """Compute the combined loss."""
        convex_loss_value = self.convex_loss(output, target)
        distance_loss_value = self.distance_loss(output, target)
        return self.alpha * convex_loss_value + self.beta * distance_loss_value


class BrooksCoreyWettingFlux(nn.Module):
    """Target wetting flux function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mu_w = kwargs["mu_w"]
        self.mu_n = kwargs["mu_n"]

        self.nb = kwargs.get("nb", 2)
        self.p_e = kwargs.get("p_e", 5.0)
        self.n1 = kwargs.get("n1", 2)
        self.n2 = kwargs.get("n2", 1 + 2 / self.nb)
        self.n3 = kwargs.get("n3", 1)

    def mobility_w(self, s):
        """Mobility function for water."""
        return s ** (self.n1 + self.n2 * self.n3) / self.mu_w

    def mobility_n(self, s):
        """Mobility function for non-aqueous phase."""
        return (1 - s) ** self.n1 * (1 - s**self.n2) ** self.n3 / self.mu_n

    def forward(self, x):
        """Forward pass through the neural network."""
        return self(x)
