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
    InstanceDataset,
    load_training_state_value_tuples,
    states_to_boolean,
)


# Use CPU instead of GPU.
device = torch.device("cpu")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        if batch % 100 == 0:
            current = batch * len(X)
            print(f"Train loss:\t {loss.item():>7f}")

        # Clear gradients for the variables it will update.
        optimizer.zero_grad()

        # Compute gradient of the loss.
        loss.backward()

        # Update parameters.
        optimizer.step()

        
def val_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            diff = np.array(pred-y)
            print(f"Val diff avg:\t {np.average(diff)}")
            val_loss += loss_fn(pred, y).item()

    val_loss /= num_batches
    print(f"Avg val loss:\t {val_loss:>8f} \n")


## Real training data
dataset = InstanceDataset("domain_to_training_pairs_blocks.json", "blocksworld")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

print("len train dataset", len(train_dataset))
print("len val dataset", len(val_dataset))

train_dataloader = DataLoader(dataset=train_dataset,
                         batch_size=10,
                         shuffle=True,
                         num_workers=1)

val_dataloader = DataLoader(dataset=val_dataset,
                         batch_size=10,
                         shuffle=True,
                         num_workers=1)

x_shape = dataset.x_shape()
y_shape = dataset.y_shape()

model = HNN(input_size=x_shape[1], hidden_units=x_shape[1], output_size=1).to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    val_loop(val_dataloader, model, loss_fn)
print("Done!")


""" 
What is torch.jit.trace?
------------------------
docs: https://pytorch.org/docs/stable/generated/torch.jit.trace.html

"Trace a function and return an executable or ScriptFunction that will be optimized using 
just-in-time compilation. Tracing is ideal for code that operates only on Tensors and lists, 
dictionaries, and tuples of Tensors."

"Using torch.jit.trace and torch.jit.trace_module, you can turn an existing module or Python 
function into a TorchScript ScriptFunction or ScriptModule. You must provide example inputs, 
and we run the function, recording the operations performed on all the tensors."

In other words, "tracing" a model means transforming your PyTorch code ("eager mode") to 
TorchScript code ("script mode"). Script mode is focused on production, while eager mode is 
for prototyping and research. Script mode is performatic (JIT) and portable. 

TorchScript is a domain-specific language for ML, and it is a subset of Python.
"""

example_input = train_dataloader.dataset[0][0]

traced_model = torch.jit.trace(model, example_input)
print(f"\nTraced model:\n{traced_model}")
traced_model.save("traced.pt")
