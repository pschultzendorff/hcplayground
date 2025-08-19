import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

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


class ConvexConcaveLoss(nn.Module):
    """Partial loss function to ensure that the network is neither convex nor concave
    w.r.t. the saturation.

    """

    def __init__(self):
        super().__init__()

    def forward(self, output):
        """Compute the convex loss."""
        # Assuming output and target are of shape (batch_size, output_dim)
        loss = torch.mean((output - target) ** 2)
        return loss


class CurvatureLoss(nn.Module):
    """Curvature loss to ensure that the ."""

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


class CoreyFlux(nn.Module):
    """Target wetting flux function."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def mobility_w(self, s):
        """Mobility function for water."""
        return s**2

    def mobility_n(self, s):
        """Mobility function for non-aqueous phase."""
        return (1 - s) ** 2

    def forward(self, x):
        """Forward pass through the neural network."""
        # Assuming x is a tensor of shape (batch_size, 2) where the first column is
        # saturation and the second column is the mobility ratio.
        s = x[:, 0]
        mobility_ratio = x[:, 1]
        lambda_w = self.mobility_w(s)
        lambda_n = self.mobility_n(s)
        return lambda_w / (lambda_w + mobility_ratio * lambda_n)


def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10):
    """Train the neural network model."""
    model.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")
    print("Training complete.")


def plot(model):
    """Plot the neural network flux function."""
    s_vals = torch.linspace(0, 1, 100)
    x = torch.stack([s_vals, s_vals], dim=1)  # Assuming two saturations

    with torch.no_grad():
        flux_values = model(x)

    plt.figure(figsize=(10, 6))
    plt.plot(s_vals.numpy(), flux_values.numpy())
    plt.xlabel("Saturation")
    plt.ylabel("Flux")
    plt.title("Neural Network Flux Function")
    plt.grid()
    plt.show()
