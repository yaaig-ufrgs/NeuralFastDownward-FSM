import torch
import torch.nn as nn
import random
from math import sqrt
import numpy as np


class Block(nn.Module):
    def __init__(self, hidden_size):
        """
        Args:
            in_channels (int):  Number of input channels.
            out_channels (int): Number of output channels.
            stride (int):       Controls the stride.
        """
        super(Block, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x):
        identity = x
        out = self.block(x)

        out += identity
        out = torch.relu(out)

        return out


class RSL(nn.Module):
    def __init__(self, num_atoms):
        super(RSL, self).__init__()
        self.flatten = nn.Flatten()
        self.denseLayers = nn.Sequential(
            nn.Linear(num_atoms, 250), nn.ReLU(), nn.Linear(250, 250), nn.ReLU()
        )
        self.resblock = Block(250)
        self.outLayer = nn.Sequential(nn.Linear(250, 1), nn.ReLU())

    def forward(self, x):
        x = self.flatten(x)
        x = self.denseLayers(x)
        x = self.resblock(x)
        out = self.outLayer(x)
        return out
