#!/usr/bin/env python3

"""
Main file used to control the evaluation process of a network.
"""

import os
import sys
import logging
import random
import numpy as np
import torch
import torch.nn as nn
from random import randint
from argparse import Namespace

from torch.utils.data import DataLoader
from src.pytorch.log import setup_full_logging
from src.pytorch.k_fold_training_data import KFoldTrainingData
from src.pytorch.utils.parse_args import get_eval_args
from src.pytorch.utils.file_helpers import save_y_pred_loss_csv
from src.pytorch.utils.timer import Timer

_log = logging.getLogger(__name__)


def eval_main(args: Namespace):
    print(args)
    if args.seed == -1:
        args.seed = randint(0, 2 ** 32 - 1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dirname = '/'.join(args.trained_model.split('/')[:2])
    print(dirname)
    setup_full_logging(dirname, log_name="eval.log")

    if args.trained_model[-2:] != "pt":
        _log.error("Invalid model path.")
        return

    _log.info("Evaluation:")
    _log.info(f"model: {args.trained_model}")
    _log.info(f"data: {args.samples}")

    model = torch.jit.load(args.trained_model)
    eval_data = KFoldTrainingData(
        args.samples,
        batch_size=1,
        seed=args.seed,
        shuffle=False,
        training_size=1.0,
        model=model,
    ).get_fold(0)[0]

    eval_timer = Timer(time_limit=None).start()
    eval_y_pred_loss = eval_model(model, eval_data, args.log_states)
    y_pred_loss_file = dirname + "/heuristic_pred_loss.csv"
    _log.info(f"Elapsed time: {eval_timer.current_time()}")
    save_y_pred_loss_csv(eval_y_pred_loss, y_pred_loss_file)
    _log.info(f"Saved (state,y,pred,loss) CSV file to: {y_pred_loss_file}")


def eval_model(model, dataloader: DataLoader, log_states):
    model.eval()
    loss_fn = RMSELoss()
    eval_loss = 0
    max_loss = float('-inf') 
    min_loss = float('inf')
    min_loss_no_goal = float('inf')
    counter = 0
    eval_y_pred = {} # { state = (y, pred, loss))}

    _log.info("Evaluating...")
    for X, y in dataloader:
        counter += 1
        pred = model(X)
        loss = loss_fn(pred, y).item()

        if loss > max_loss:
            max_loss = loss
        if loss < min_loss:
            min_loss = loss
        if loss < min_loss_no_goal and int(torch.round(y[0])) != 0:
            min_loss_no_goal = loss
        eval_loss += loss

        x_lst = X.tolist()
        x_int = [int(x) for x in x_lst[0]]
        x_str = "".join(str(e) for e in x_int)

        if log_states:
            _log.info(f"| state: {x_str}")
            _log.info(f"| y: {float(y[0])} | pred: {float(pred[0])} | loss: {loss}")

        eval_y_pred[x_str] = (int(torch.round(y[0])), int(torch.round(pred[0])), loss)


    mean_loss = eval_loss / len(dataloader)
    _log.info(f"Results:")
    _log.info(f"| mean_loss: {mean_loss}")
    _log.info(f"| min_loss: {min_loss}")
    _log.info(f"| min_loss_no_goal: {min_loss_no_goal}")
    _log.info(f"| max_loss: {max_loss}")

    return eval_y_pred


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


if __name__ == "__main__":
    eval_main(get_eval_args())
