"""
Contains extra loss functions that can be used.
"""

import torch
import torch.nn as nn


class RMSELoss(torch.nn.Module):
    # For now, doesn't work on `train.py`, only `eval.py`.
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
