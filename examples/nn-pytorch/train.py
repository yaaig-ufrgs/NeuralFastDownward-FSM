#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

import math
import sys

from model import HNN


# TODO train.py

# Use CPU instead of GPU.
device = torch.device("cpu")

# TODO prepare REAL training/input data data
# Example input just to check if things are working
x = torch.linspace(-math.pi, math.pi, 2000, device=device)
y = torch.sin(x)

# Prepare the input tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p).to(device)

hnn = HNN().to(device)
print(hnn)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(hnn.parameters(), lr=0.001)

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
