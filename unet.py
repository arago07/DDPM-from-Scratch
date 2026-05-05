import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeEmbedding(nn.Module):

    """
    Sinusoidal time embedding + 2-layer MLP.

    Converts a scalar timestep t (an integer) into a high-dimensional
    vector that the U-Net can use to condition each ResidualBlock.

    Inspired by Transformer's positional encoding.
    """

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = self.n_channels
        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        self.act = nn.SiLU()
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t):
        # Todo: tomorrow 
        pass