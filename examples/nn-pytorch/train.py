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

# Use CPU instead of GPU.
device = torch.device("cpu")

## Real training data
training_data_file = "domain_to_training_pairs_blocks.json"
state_value_pairs = load_training_state_value_tuples(training_data_file)

states = states_to_boolean(state_value_pairs, "blocksworld")
heuristics = [t[1] for t in state_value_pairs]

y = torch.tensor(heuristics, dtype=torch.float32)
x = torch.tensor(states, dtype=torch.float32)

print("x:", x.shape)
print("y:", y.shape)

hnn = HNN(input_size=x.shape[1], hidden_units=x.shape[1], output_size=1).to(device)
print(hnn)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(hnn.parameters(), lr=0.001)

for epoch in range(1000):
    loss = 0.0
    y_pred = hnn(x)

    # Calculate loss.
    loss = loss_fn(y_pred, y)

    if epoch % 100 == 99:
        print(f"{epoch}\t y_pred={y_pred}, y={y}\t loss={loss.item()}")

    # Clear gradients for the variables it will update.
    optimizer.zero_grad()

    # Compute gradient of the loss.
    loss.backward()

    # Update parameters.
    optimizer.step()
