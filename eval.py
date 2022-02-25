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
from src.pytorch.utils.plot import save_y_pred_scatter_eval, save_pred_error_bar_eval
from src.pytorch.utils.helpers import get_train_args_json
import src.pytorch.utils.default_args as default_args

_log = logging.getLogger(__name__)


def eval_main(args: Namespace):
    if args.trained_model[-2:] != "pt":
        _log.error("Invalid model path.")
        return

    if len(args.samples) == 0:
        _log.error("No data given.")
        return

    dirname = "/".join(args.trained_model.split("/")[:2])

    if args.follow_training:
        original_train_args = get_train_args_json(dirname)
        args.seed = original_train_args["seed"]
        args.shuffle_seed = original_train_args["shuffle_seed"]
        args.shuffle = original_train_args["shuffle"]
        args.training_size = original_train_args["training_size"]
        args.unique_samples = original_train_args["unique_samples"]

    set_seeds(args)

    eval_files = glob(f"{dirname}/eval*.log")
    log_name = (
        "eval.log" if len(eval_files) == 0 else "eval_" + str(len(eval_files)) + ".log"
    )
    setup_full_logging(dirname, log_name=log_name)

    cmd_line = " ".join(sys.argv[0:])
    logging_eval_config(args, dirname, cmd_line)

    eval_results_files = glob(f"{dirname}/eval_results*.csv")
    eval_results_name = (
        "eval_results.csv"
        if len(eval_results_files) == 0
        else "eval_results_" + str(len(eval_results_files)) + ".csv"
    )
    eval_results_path = dirname + "/" + eval_results_name

    f_results = open(eval_results_path, "a")
    f_results.write("sample,num_samples,misses,max_rounded_abs_error,mean_rounded_abs_error,min_rmse_loss,min_rmse_loss_no_goal,mean_rmse_loss,max_rmse_loss,time\n")

    model = torch.jit.load(args.trained_model)
    model.eval()

    eval_timer_total = Timer(time_limit=None).start()

    # EVALUATION
    for sample in args.samples:
        train_data, val_data, test_data = KFoldTrainingData(
            sample,
            batch_size=1,
            shuffle=args.shuffle,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            training_size=args.training_size,
            unique_samples=args.unique_samples,
            model=model,
        ).get_fold(0)

        if train_data != None:
            eval_workflow(model, sample, dirname, train_data, "training", f_results, args.log_states, args.save_preds, args.save_plots)
        if val_data != None:
            eval_workflow(model, sample, dirname, val_data, "validation", f_results, args.log_states, args.save_preds, args.save_plots)
        if test_data != None:
            eval_workflow(model, sample, dirname, test_data, "test", f_results, args.log_states, args.save_preds, args.save_plots)

    _log.info(f"Total elapsed time for evaluation: {eval_timer_total.current_time()}")

    f_results.close()


def eval_workflow(model, sample: str, dirname: str, dataloader: DataLoader, data_type: str, f_results, log_states: bool, save_preds: bool, save_plots: bool):
    """
    Complete model evaluation workflow for a given dataset.
    """
    data_name = sample.split("/")[-1]
    _log.info(f"Evaluating {data_name} ({data_type} dataset)...")

    eval_timer = Timer(time_limit=None).start()

    (
        y_pred_loss,
        num_samples,
        misses,
        max_abs_error,
        mean_abs_error,
        min_loss,
        min_loss_no_goal,
        mean_loss,
        max_loss,
    ) = eval_model(model, dataloader, log_states)

    curr_time = eval_timer.current_time()

    _log.info(f"Results for {data_type} dataset:")
    _log.info(f"| num_samples: {num_samples}")
    _log.info(f"| rounded_misses: {misses}")
    _log.info(f"| max_rounded_abs_error: {max_abs_error}")
    _log.info(f"| mean_rounded_abs_error: {mean_abs_error}")
    _log.info(f"| min_rmse_loss: {min_loss}")
    _log.info(f"| min_rmse_loss_no_goal: {min_loss_no_goal}")
    _log.info(f"| mean_rmse_loss: {mean_loss}")
    _log.info(f"| max_rmse_loss: {max_loss}")
    _log.info(f"| elapsed time: {curr_time}")

    if save_preds:
        y_pred_loss_file = f"{dirname}/{data_name}_{data_type}.csv"
        save_y_pred_loss_csv(y_pred_loss, y_pred_loss_file)
        _log.info(f"Saved {data_type} dataset (state,y,pred,loss) CSV file to {y_pred_loss_file}")

    if save_plots:
        plots_dir = f"{dirname}/plots"
        save_y_pred_scatter_eval(y_pred_loss, plots_dir, data_type, data_name)
        save_pred_error_bar_eval(y_pred_loss, plots_dir, data_type, data_name)
        _log.info(f"Saved {data_name} plots for {data_type} dataset to {plots_dir}")

    if f_results != None:
        f_results.write(
            f"{data_type},,,,,,,,,\n"
        )
        f_results.write(
            f"{data_name},{num_samples},{misses},{max_abs_error},{mean_abs_error},{min_loss},{min_loss_no_goal},{mean_loss},{max_loss},{curr_time}\n"
        )
        _log.info(f"Saved results to a CSV file.")


def eval_model(model, dataloader: DataLoader, log_states: bool):
    loss_fn = RMSELoss()
    eval_loss = 0
    max_loss = float("-inf")
    min_loss = float("inf")
    min_loss_no_goal = float("inf")
    eval_y_pred = []  # [[state, y, pred, abs_error, loss], ...]
    eval_abs_error = 0
    max_abs_error = 0
    misses = 0

    # Observation: RMSE and "abssolute error" here are virtually the same, except the RMSE is comparing y
    # directly with the floating-point output (prediction) of the trained model, while the "abs error"
    # is comparing y with the rounded (integer) prediction of the trained model.
    # This distinction is made because during search, the output of the network is rounded; therefore, while the
    # difference between the absolute error and RMSE here are minimum, the former directly reflects the output
    # of the network during search.

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y).item()

            rounded_y = int(torch.round(y[0]))
            rounded_pred = int(torch.round(pred[0]))
            if rounded_y != rounded_pred:
                misses += 1

            abs_error = abs(rounded_y - rounded_pred)
            if abs_error > max_abs_error:
                max_abs_error = abs_error
            eval_abs_error += abs_error

            if loss > max_loss:
                max_loss = loss
            if loss < min_loss:
                min_loss = loss
            if loss < min_loss_no_goal and rounded_y != 0:
                min_loss_no_goal = loss
            eval_loss += loss

            x_lst = X.tolist()
            x_int = [int(x) for x in x_lst[0]]
            x_str = "".join(str(e) for e in x_int)

            if log_states:
                _log.info(f"| state: {x_str}")
                _log.info(
                    f"| y: {float(y[0])} | pred: {float(pred[0])} | rmse_loss: {loss} | abs_error_round: {abs_error}"
                )

            eval_y_pred.append([x_str, float(y[0]), float(pred[0]), abs_error, loss])

    mean_loss = eval_loss / len(dataloader)
    mean_abs_error = eval_abs_error / len(dataloader)

    return (
        eval_y_pred,
        len(dataloader),
        misses,
        max_abs_error,
        mean_abs_error,
        min_loss,
        min_loss_no_goal,
        mean_loss,
        max_loss,
    )


def set_seeds(args: Namespace, shuffle_seed: bool = True):
    """
    Sets seeds to assure program reproducibility.
    """
    if args.seed == -1:
        args.seed = randint(0, 2 ** 32 - 1)
    if shuffle_seed and args.shuffle_seed == -1:
        args.shuffle_seed = args.seed
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    random.seed(args.seed)
    np.random.seed(args.seed)


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


if __name__ == "__main__":
    eval_main(get_eval_args())
