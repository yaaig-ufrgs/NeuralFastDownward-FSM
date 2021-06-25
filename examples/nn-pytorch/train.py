#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import math
import sys

from model import HNN
from training_data import (
   load_training_state_value_tuples,
   states_to_boolean,
)


# TODO train.py

# Use CPU instead of GPU.
device = torch.device("cpu")

# TODO prepare REAL training/input data
# Ferber on the input:
# The inputs of our networks are states represented as fixed
# size Boolean vectors. We associate every entry of the input
# vector with a fact of Π. For all facts that are part of the input
# state we set their vector entries to 1. All other entries are set
# to 0. We train the network using the mean squared error as
# loss function and the adam optimizer with its default param-

## Example input just to check if things are working
#x = torch.linspace(-math.pi, math.pi, 2000, device=device)
#y = torch.sin(x)
# Prepare the input tensor (x, x^2, x^3).
#p = torch.tensor([1, 2, 3])
#xx = x.unsqueeze(-1).pow(p).to(device)

## Real training data
training_data_file = "domain_to_training_pairs_blocks.json"
state_value_pairs = load_training_state_value_tuples(training_data_file)

states = states_to_boolean(state_value_pairs, "blocksworld")
heuristics = [t[1] for t in state_value_pairs]

print(len(states))
print(len(heuristics))

# CONTINUE...


hnn = HNN().to(device)
print(hnn)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(hnn.parameters(), lr=0.001)

"""
for epoch in range(1000):
    loss = 0.0
    y_pred = hnn(xx)

    # Calculate loss.
    loss = loss_fn(y_pred, y)

    if epoch % 100 == 99:
        print(f"{epoch}\t {loss.item()}")

    # Clear gradients for the variables it will update.
    optimizer.zero_grad()

    # Compute gradient of the loss.
    loss.backward()

    # Update parameters.
    optimizer.step()
"""
