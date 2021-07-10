#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import HNN
from train_workflow import TrainWorkflow
from training_data import InstanceDataset

# Use CPU instead of GPU.
device = torch.device("cpu")


def setup_dataloaders(dataset: InstanceDataset, train_split: float, batch_size: int, shuffle: bool):

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=10,
                            shuffle=True,
                            num_workers=1)

    val_dataloader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=1)

    return train_dataloader, val_dataloader


dataset = InstanceDataset("sas-plans/sas_plan_test_npuzzle")
train_dataloader, val_dataloader = setup_dataloaders(dataset=dataset,
                                                     train_split=0.8,
                                                     batch_size=10,
                                                     shuffle=True)

x_shape = dataset.x_shape()
y_shape = dataset.y_shape()

model = HNN(input_size=x_shape[1], hidden_units=x_shape[1], output_size=1).to(device)
print(model)

train_wf = TrainWorkflow(model=model,
                         train_dataloader=train_dataloader,
                         val_dataloader=val_dataloader,
                         max_num_epochs=100,
                         optimizer=optim.Adam(model.parameters(), lr=0.001))

train_wf.run()

train_wf.save_traced_model("traced.pt")
