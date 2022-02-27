"""
Contains extra loss functions that can be used.
"""

import torch
import torch.nn as nn
import numpy as np


class RMSELoss(torch.nn.Module):
    # For now, doesn't work on `train.py`, only `eval.py`.
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class MSELossWeighted(torch.nn.Module):
    def __init__(self):
        super(MSELossWeighted, self).__init__()

    def forward(self, x, y, w):
        loss =  (x - y)**2
        loss = loss * w
        loss = torch.mean(loss)
        return loss
