#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim

import math
import sys

# Use CPU instead of GPU.
device = torch.device("cpu")

# TODO Prepare training data
# TODO Create model (Ferber)
#     - Supervised learning
#     - Feed forward
#     - Input = state
#     - 3 hidden layers, Sigmoid activation
#     - Output = heuristic
#     - Adam optimizer, MSE, Batch size 100
#     - 10-fold cross-validation (9 as training, 1 as validation) --> low-priority
# TODO Train


# H(euristic) Neural Network
class HNN(nn.Module):
    def __init__(
        self,
        input_size: int = 3,    # set as 3 just for the example input
        hidden_units: int = 32,
        output_size: int = 1,
            
    ):
        super(HNN, self).__init__()
        self.hid1 = nn.Linear(input_size, hidden_units)
        self.hid2 = nn.Linear(hidden_units, hidden_units)
        self.hid3 = nn.Linear(hidden_units, hidden_units)
        self.opt = nn.Linear(hidden_units, output_size)
        self.opt1 = nn.Flatten(0, 1) # just for the example input

    def forward(self, x):
        z = torch.sigmoid(self.hid1(x))
        z = torch.sigmoid(self.hid2(z))
        z = torch.sigmoid(self.hid3(z))
        z = self.opt(z)
        z = self.opt1(z) # just for the example input
        return z


hnn = HNN().to(device)
print(hnn)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(hnn.parameters(), lr=0.001)

device = torch.device("cpu")

# Example input just to check if things are working
x = torch.linspace(-math.pi, math.pi, 2000, device=device)
y = torch.sin(x)

# Prepare the input tensor (x, x^2, x^3).
p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p).to(device)

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
