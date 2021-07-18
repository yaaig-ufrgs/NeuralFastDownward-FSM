#!/usr/bin/env python3

from sys import argv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import HNN
from train_workflow import TrainWorkflow
from k_fold_training_data import KFoldTrainingData

"""
Use as:
$ ./train <training_domain> <task_folder>
"""

domain = argv[1]
task_folder = argv[2]
domain_max_value = 1

# Use CPU instead of GPU.
device = torch.device("cpu")


# TODO other domains
if domain == "blocksworld":
    domain_max_value = 327

# TODO
domain = task_folder+"/domain.pddl"
problems = []
N_PROBLEMS = 10
for i in range(N_PROBLEMS):
    problems.append(task_folder+f"/p{i+1}.pddl")

N_FOLDS = 10
kfold = KFoldTrainingData(domain, problems, domain_max_value, batch_size=10, num_folds=N_FOLDS, shuffle=False)

for fold_idx in range(N_FOLDS):
    train_dataloader, val_dataloader = kfold.get_fold(fold_idx)

    model = HNN(input_size=train_dataloader.dataset.x_shape()[1],
                hidden_units=train_dataloader.dataset.x_shape()[1],
                output_size=train_dataloader.dataset.y_shape()[1]).to(device)

    print(model)

    train_wf_blind = TrainWorkflow(model=model,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                max_num_epochs=10,
                                optimizer=optim.Adam(model.parameters(), lr=0.001))

    train_wf_blind.run(validation=True)

    blind_model_fname = f"traced_fold_{fold_idx}.pt"
    train_wf_blind.save_traced_model(blind_model_fname)
