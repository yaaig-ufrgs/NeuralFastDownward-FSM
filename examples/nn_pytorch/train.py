#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import HNN
from train_workflow import TrainWorkflow
from training_data import InstanceDataset
from fast_downward_api import solve_instances_with_fd

# Use CPU instead of GPU.
device = torch.device("cpu")

def setup_dataloaders(dataset: InstanceDataset, train_split: float, batch_size: int, shuffle: bool):

    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=1)

    return train_dataloader, val_dataloader


dataset = InstanceDataset("sas-plans/sas_plan_test_npuzzle")
train_dataloader, val_dataloader = setup_dataloaders(dataset=dataset,
                                                     train_split=0.8,
                                                     batch_size=10,
                                                     shuffle=True)

model = HNN(input_size=dataset.x_shape()[1], hidden_units=dataset.x_shape()[1], output_size=1).to(device)
print(model)

train_wf_blind = TrainWorkflow(model=model,
                               train_dataloader=train_dataloader,
                               val_dataloader=val_dataloader,
                               max_num_epochs=100,
                               optimizer=optim.Adam(model.parameters(), lr=0.001))

train_wf_blind.run()

blind_model_fname = "traced.pt"
train_wf_blind.save_traced_model(blind_model_fname)

# (instance_idx, exit_code)
instance_list = solve_instances_with_fd(blind_model_fname, "instances/npuzzle-domain.pddl",
                                        ["instances/n-puzzle-3x3-s44.pddl"], blind=True)

print(instance_list)
# TODO remove solved instances from the dataset and make new training workflow
