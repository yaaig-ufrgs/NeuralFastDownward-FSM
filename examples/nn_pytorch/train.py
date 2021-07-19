#!/usr/bin/env python3

from sys import argv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import HNN
from train_workflow import TrainWorkflow
from k_fold_training_data import KFoldTrainingData
import fast_downward_api as fd_api

"""
Use as:
$ ./train.py <training_domain> <task_folder>
e.g. $ ./train.py blocksworld ../../tasks/blocksworld_ipc/probBLOCKS-12-0
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
domain_pddl = task_folder+"/domain.pddl"
problems = []
N_PROBLEMS = 5
for i in range(N_PROBLEMS):
    problems.append(task_folder+f"/p{i+1}.pddl")

N_FOLDS = 5
kfold = KFoldTrainingData(domain_pddl, problems, domain_max_value, batch_size=10, num_folds=N_FOLDS, shuffle=False)

val_success = []
for fold_idx in range(N_FOLDS):
    train_dataloader, val_dataloader = kfold.get_fold(fold_idx)

    model = HNN(input_size=train_dataloader.dataset.x_shape()[1],
                hidden_units=train_dataloader.dataset.x_shape()[1],
                output_size=train_dataloader.dataset.y_shape()[1]).to(device)

    print(model)

    train_wf = TrainWorkflow(model=model,
                                train_dataloader=train_dataloader,
                                val_dataloader=val_dataloader,
                                max_num_epochs=100,
                                optimizer=optim.Adam(model.parameters(), lr=0.001))

    train_wf.run(validation=True)

    model_fname = f"traced_fold_{fold_idx}.pt"
    train_wf.save_traced_model(model_fname)

    """
    Test step
    """
    """
    val_problems = kfold.get_test_problems_from_fold(fold_idx)
    plans_found = 0
    for problem in val_problems:
        cost = fd_api.solve_instance_with_fd_nh(domain, problem, model_fname)
        plans_found += int(cost != None)
    success_rate = 100 * plans_found / len(val_problems)
    val_success.append(success_rate)
    print(f"Fold {fold_idx} val success: {plans_found} of {len(val_problems)} ({success_rate}%)")

print()
print(f"Max val success (fold {val_success.index(max(val_success))}): {max(val_success)}%")
print(f"Min val success (fold {val_success.index(min(val_success))}): {min(val_success)}%")
print(f"Avg val success: {sum(val_success) / len(val_success)}%")
    """
