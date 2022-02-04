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
from glob import glob
from torch.utils.data import DataLoader
from src.pytorch.log import setup_full_logging
from src.pytorch.k_fold_training_data import KFoldTrainingData
from src.pytorch.utils.parse_args import get_eval_args
from src.pytorch.utils.file_helpers import save_y_pred_loss_csv
from src.pytorch.utils.log_helpers import logging_eval_config
from src.pytorch.utils.timer import Timer

_log = logging.getLogger(__name__)


def eval_main(args: Namespace):
    if args.seed == -1:
        args.seed = randint(0, 2 ** 32 - 1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.trained_model[-2:] != "pt":
        _log.error("Invalid model path.")
        return

    if len(args.samples) == 0:
        _log.error("No data given.")
        return

    dirname = '/'.join(args.trained_model.split('/')[:2])
    eval_files = glob(f"{dirname}/eval*.log")
    log_name = "eval.log" if len(eval_files) == 0 else "eval_" + str(len(eval_files)) + ".log"
    setup_full_logging(dirname, log_name=log_name)

    cmd_line = " ".join(sys.argv[0:])
    logging_eval_config(args, dirname, cmd_line)

    eval_results_files = glob(f"{dirname}/eval_results*.csv")
    eval_results_name = "eval_results.csv" if len(eval_results_files) == 0 else "eval_results_" + str(len(eval_results_files)) + ".csv"
    eval_results_path = dirname + "/" + eval_results_name

    f_results = open(eval_results_path, 'a')
    f_results.write("sample,min_loss,min_loss_no_goal,mean_loss,max_loss,time\n")

    model = torch.jit.load(args.trained_model)
    model.eval()

    eval_timer_total = Timer(time_limit=None).start()
    for sample in args.samples:
        eval_data = KFoldTrainingData(
            sample,
            batch_size=1,
            seed=args.seed,
            shuffle=False,
            training_size=1.0,
            model=model,
        ).get_fold(0)[0]

        data_name = sample.split('/')[-1]
        _log.info(f"Evaluating {data_name}...")

        eval_timer = Timer(time_limit=None).start()
        y_pred_loss, min_loss, min_loss_no_goal, mean_loss, max_loss = eval_model(model, eval_data, args.log_states)
        curr_time = eval_timer.current_time()
        _log.info(f"Results:")
        _log.info(f"| min_loss: {min_loss}")
        _log.info(f"| min_loss_no_goal: {min_loss_no_goal}")
        _log.info(f"| mean_loss: {mean_loss}")
        _log.info(f"| max_loss: {max_loss}")
        _log.info(f"| elapsed time: {curr_time}")

        y_pred_loss_file = dirname + "/" + data_name + ".csv"
        save_y_pred_loss_csv(y_pred_loss, y_pred_loss_file)
        _log.info(f"Saved (state,y,pred,loss) CSV file to: {y_pred_loss_file}")

        f_results.write(f"{data_name},{min_loss},{min_loss_no_goal},{mean_loss},{max_loss},{curr_time}\n")
        _log.info(f"Appended results to {eval_results_path}")

    _log.info(f"Total elapsed time: {eval_timer_total.current_time()}")
    f_results.close()


def eval_model(model, dataloader: DataLoader, log_states):
    loss_fn = RMSELoss()
    eval_loss = 0
    max_loss = float('-inf') 
    min_loss = float('inf')
    min_loss_no_goal = float('inf')
    eval_y_pred = {} # { state = (y, pred, loss))}
    repeat_count = 0

    with torch.no_grad():
        for X, y in dataloader:
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

            if x_str in eval_y_pred:
                repeat_count += 1
                x_str += "_" + str(repeat_count)

            eval_y_pred[x_str] = (int(torch.round(y[0])), int(torch.round(pred[0])), loss)


    mean_loss = eval_loss / len(dataloader)

    return eval_y_pred, min_loss, min_loss_no_goal, mean_loss, max_loss


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


if __name__ == "__main__":
    eval_main(get_eval_args())
