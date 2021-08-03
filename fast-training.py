#!/usr/bin/env python3

from sys import argv
from random import shuffle
from datetime import datetime
from os import path, makedirs
import torch
import torch.optim as optim

from src.pytorch.model import HNN
from src.pytorch.train_workflow import TrainWorkflow
from src.pytorch.k_fold_training_data import KFoldTrainingData
import src.pytorch.fast_downward_api as fd_api

"""
Use: $ ./fast-training.py sas_plan test_tasks_folder
e.g. $ ./fast-training.py ../../results/sampling/sampling_blocksworld_ipc/probBLOCKS-12-0 ../../tasks/blocksworld_ipc/probBLOCKS-12-0
"""

OUTPUT_MODEL_FOLDER = f"results/train-pytorch-{datetime.now().isoformat().replace('-', '.').replace(':', '.')}"

if __name__ == "__main__":
    samples = argv[1]
    test_task_folder = argv[2]
    shuffle_tasks = False
    test_step = False

    if not path.exists(OUTPUT_MODEL_FOLDER):
        makedirs(OUTPUT_MODEL_FOLDER)

    domain_pddl = test_task_folder+"/domain.pddl"
    problems_pddl = []
    N_PROBLEMS = 200
    for i in range(N_PROBLEMS):
        problems_pddl.append(test_task_folder+f"/p{i+1}.pddl")
    if shuffle_tasks:
        shuffle(problems_pddl)


    N_FOLDS = 10
    kfold = KFoldTrainingData(samples, batch_size=100, num_folds=N_FOLDS, shuffle=shuffle_tasks)

    test_success = []
    for fold_idx in range(N_FOLDS):
        train_dataloader, val_dataloader = kfold.get_fold(fold_idx)

        model = HNN(input_size=train_dataloader.dataset.x_shape()[1],
                    hidden_units=train_dataloader.dataset.x_shape()[1],
                    output_size=train_dataloader.dataset.y_shape()[1]).to(torch.device("cpu"))

        print(f"\n---------- FOLD {fold_idx+1}/{N_FOLDS} ----------")
        print(model)
        print()

        train_wf = TrainWorkflow(model=model,
                                    train_dataloader=train_dataloader,
                                    val_dataloader=val_dataloader,
                                    max_num_epochs=7,
                                    optimizer=optim.Adam(model.parameters(), lr=0.001))

        train_wf.run(validation=True)

        model_fname = f"{OUTPUT_MODEL_FOLDER}/traced_fold_{fold_idx}.pt"
        train_wf.save_traced_model(model_fname)

        """ Test """
        if test_step:
            plans_found = 0
            num_tests = int(len(problems_pddl) / N_FOLDS)
            for problem_pddl in problems_pddl[num_tests*fold_idx : num_tests*fold_idx+num_tests]:
                cost = fd_api.solve_instance_with_fd_nh(domain_pddl, problem_pddl, model_fname)
                plans_found += int(cost != None)
            success_rate = 100 * plans_found / num_tests
            test_success.append(success_rate)
            print(f"Fold {fold_idx} test success: {plans_found} of {num_tests} ({success_rate}%)")

    if test_step:
        print()
        print(f"Max test success (fold {test_success.index(max(test_success))}): {max(test_success)}%")
        print(f"Min test success (fold {test_success.index(min(test_success))}): {min(test_success)}%")
        print(f"Avg test success: {sum(test_success) / len(test_success)}%")
