#!/usr/bin/env python3

"""
Main file used to control the training process.
"""

import os
import sys
import logging
import random
import numpy as np
import glob
import torch
import torch.nn as nn
from shutil import copyfile
from random import randint

from src.pytorch.k_fold_training_data import KFoldTrainingData
from src.pytorch.model import HNN
from src.pytorch.train_workflow import TrainWorkflow
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    get_fixed_max_epochs,
    add_train_arg,
    get_problem_by_sample_filename,
)
from src.pytorch.utils.file_helpers import (
    create_train_directory,
    save_y_pred_csv,
    remove_csv_except_best,
)
from src.pytorch.utils.log_helpers import logging_train_config
from src.pytorch.utils.plot import (
    save_h_pred_scatter,
    save_box_plot,
    save_gif_from_plots,
    remove_intermediate_plots,
)
import src.pytorch.utils.default_args as default_args
from src.pytorch.utils.parse_args import get_train_args
from src.pytorch.utils.timer import Timer
from src.pytorch.utils.loss import *
from eval import eval_workflow 
from argparse import Namespace

_log = logging.getLogger(__name__)


def train_main(args: Namespace):
    """
    Higher-level setup of the full training procedure.
    """
    if args.num_threads != -1:
        torch.set_num_threads(args.num_threads)

    set_seeds(args)

    # If forcibly changing the order that the samples are presented, disable shuffling.
    if args.standard_first or args.contrast_first or args.intercalate_samples != 0:
        args.shuffle = False

    args.domain, args.problem = get_problem_by_sample_filename(args.samples)
    args.save_git_diff = True

    dirname = create_train_directory(args)
    setup_full_logging(dirname)

    if len(args.hidden_units) not in [0, 1, args.hidden_layers]:
        _log.error("Invalid hidden_units length.")
        return
    if args.max_epochs == -1:
        args.max_epochs = get_fixed_max_epochs(args)
    if (
        args.max_epochs == default_args.MAX_EPOCHS
        and args.max_training_time == default_args.MAX_TRAINING_TIME
    ):
        args.max_epochs = default_args.FORCED_MAX_EPOCHS
        _log.warning(
            f"Neither max epoch nor max training time have been defined. "
            f"Setting maximum epochs to {default_args.FORCED_MAX_EPOCHS}."
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.use_gpu else "cpu"
    )
    if device == torch.device("cpu"):
        args.use_gpu = False

    cmd_line = " ".join(sys.argv[0:])
    logging_train_config(args, dirname, cmd_line)

    # TRAINING
    best_fold, num_retries, train_timer = train_nn(args, dirname, device)

    _log.info("Finishing training.")
    _log.info(f"Elapsed time: {train_timer.current_time()}")
    _log.info(f"Restarts needed: {num_retries}")

    h_pred_files = glob.glob(f"{dirname}/heuristic_pred*.csv")
    if len(h_pred_files) > 1:
        remove_csv_except_best(dirname, best_fold["fold"])
        os.rename(
            f"{dirname}/heuristic_pred_{best_fold['fold']}.csv",
            f"{dirname}/heuristic_pred.csv",
        )
    else:
        os.rename(
            f"{dirname}/heuristic_pred_0.csv",
            f"{dirname}/heuristic_pred.csv",
        )

    try:
        if args.training_size != 1.0:
            _log.info(
                f"Saving traced_{best_fold['fold']}.pt as best "
                f"model (by val loss = {best_fold['val_loss']})"
            )
            copyfile(
                f"{dirname}/models/traced_{best_fold['fold']}.pt",
                f"{dirname}/models/traced_best_val_loss.pt",
            )
    except:
        _log.error(f"Failed to save best fold.")


    _log.info(f"Restarts needed: {num_retries}")
    _log.info("Training complete!")

    # OTHER PLOTS
    make_extra_plots(args, dirname, best_fold)

    # POST-TRAINING EVALUATION OF THE BEST MODEL
    if args.post_train_eval:
        _log.info("Performing post-training evaluation...")
        post_training_evaluation(f"{dirname}/models/traced_best_val_loss.pt", args, dirname)
        _log.info("Finished!")


def train_nn(args: Namespace, dirname: str, device: torch.device) -> (dict, int, Timer):
    """
    Manages the training procedure.
    """
    num_retries = 0
    born_dead = True
    _log.warning(f"ATTENTION: Training will be performed on device '{device}'.")

    losses = {"mse": (nn.MSELoss(), False), "mse_weighted": (MSELossWeighted(), True), "rmse": (RMSELoss(), False)}
    chosen_loss_function, is_weighted = losses[args.loss_function]

    while born_dead:
        kfold = KFoldTrainingData(
            args.samples,
            device=device,
            batch_size=args.batch_size,
            num_folds=args.num_folds,
            output_layer=args.output_layer,
            shuffle=args.shuffle,
            seed=args.seed,
            shuffle_seed=args.shuffle_seed,
            training_size=args.training_size,
            data_num_workers=args.data_num_workers,
            normalize=args.normalize_output,
            clamping=args.clamping,
            remove_goals=args.remove_goals,
            standard_first=args.standard_first,
            contrast_first=args.contrast_first,
            intercalate_samples=args.intercalate_samples,
            cut_non_intercalated_samples=args.cut_non_intercalated_samples,
            sample_percentage=args.sample_percentage,
            unique_samples=args.unique_samples,
            unique_states=args.unique_states,
            model=args.model,
        )

        if args.normalize_output:
            # Add the reference value in train_args.json to denormalize in the test
            add_train_arg(dirname, "max_h", kfold.domain_max_value)

        train_timer = Timer(args.max_training_time).start()
        best_fold = {"fold": -1, "val_loss": float("inf")}

        for fold_idx in range(args.num_folds):
            _log.info(
                f"Running training workflow for fold {fold_idx+1} out "
                f"of {args.num_folds}"
            )

            train_dataloader, val_dataloader, test_dataloader = kfold.get_fold(fold_idx)

            model = HNN(
                input_units=train_dataloader.dataset.x_shape()[1],
                hidden_units=args.hidden_units,
                output_units=train_dataloader.dataset.y_shape()[1],
                hidden_layers=args.hidden_layers,
                activation=args.activation,
                output_layer=args.output_layer,
                dropout_rate=args.dropout_rate,
                linear_output=args.linear_output,
                use_bias=args.bias,
                use_bias_output=args.bias_output,
                weights_method=args.weights_method,
            ).to(device)

            if fold_idx == 0:
                _log.info(model)

            train_wf = TrainWorkflow(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                device=device,
                max_epochs=args.max_epochs,
                plot_n_epochs=args.plot_n_epochs,
                save_best=args.save_best_epoch_model,
                dirname=dirname,
                optimizer=torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                ),
                scatter_plot=args.scatter_plot,
                loss_fn=chosen_loss_function,
                is_weighted_loss_fn=is_weighted,
                restart_no_conv=args.restart_no_conv,
                patience=args.patience,
            )

            fold_val_loss, born_dead = train_wf.run(fold_idx, train_timer)

            if born_dead and args.num_folds == 1:
                args.seed += args.seed_increment_when_born_dead
                _log.info(f"Updated seed: {args.seed}")
                set_seeds(args, False)
                num_retries += 1
                add_train_arg(dirname, "updated_seed", args.seed)
                break

            heuristic_pred_file = f"{dirname}/heuristic_pred_{fold_idx}.csv"

            if fold_val_loss != None:
                if fold_val_loss < best_fold["val_loss"]:
                    save_y_pred_csv(train_wf.train_y_pred_values, heuristic_pred_file)
                    _log.info(f"New best val loss at fold {fold_idx} = {fold_val_loss}")
                    best_fold["fold"] = fold_idx
                    best_fold["val_loss"] = fold_val_loss
                else:
                    _log.info(
                        f"Val loss at fold {fold_idx} = {fold_val_loss} (best = {best_fold['val_loss']})"
                    )
            else:  # Only using training data
                save_y_pred_csv(train_wf.train_y_pred_values, heuristic_pred_file)
                best_fold["fold"] = fold_idx
                best_fold["train_loss"] = train_wf.cur_train_loss

            train_wf.save_traced_model(
                f"{dirname}/models/traced_{fold_idx}.pt", args.model
            )

            if train_timer.check_timeout():
                _log.info(f"Maximum training time reached. Stopping training.")
                break

    return best_fold, num_retries, train_timer


def post_training_evaluation(trained_model: str, args: Namespace, dirname: str) -> (dict, int, Timer):
    """
    Manages the training procedure.
    """
    model = torch.jit.load(trained_model)
    model.eval()

    train_data, val_data, test_data = KFoldTrainingData(
        args.samples,
        device=torch.device("cpu"),
        batch_size=1,
        num_folds=args.num_folds,
        output_layer=args.output_layer,
        shuffle=args.shuffle,
        seed=args.seed,
        shuffle_seed=args.shuffle_seed,
        training_size=args.training_size,
        data_num_workers=args.data_num_workers,
        normalize=args.normalize_output,
        clamping=args.clamping,
        remove_goals=args.remove_goals,
        standard_first=args.standard_first,
        contrast_first=args.contrast_first,
        intercalate_samples=args.intercalate_samples,
        cut_non_intercalated_samples=args.cut_non_intercalated_samples,
        sample_percentage=args.sample_percentage,
        unique_samples=args.unique_samples,
        model=args.model,
    ).get_fold(0)

    if train_data != None:
        eval_workflow(model, args.samples, dirname, train_data, "training", None, False, args.save_heuristic_pred, args.scatter_plot)
    if val_data != None:
        eval_workflow(model, args.samples, dirname, val_data, "validation", None, False, args.save_heuristic_pred, args.scatter_plot)
    if test_data != None:
        eval_workflow(model, args.samples, dirname, test_data, "test", None, False, args.save_heuristic_pred, args.scatter_plot)


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


def make_extra_plots(args: Namespace, dirname: str, best_fold: dict):
    """
    Manages extra plots, suchs as:
    - h vs predicted h scatter plot animation.
    - hnn vs chosen heuristic scatter plot.
    - boxplot with hnn, h* and goalcount.
    """
    plots_dir = f"{dirname}/plots"

    if args.scatter_plot and args.plot_n_epochs != -1:
        try:
            _log.info(f"Saving scatter plot GIF.")
            save_gif_from_plots(plots_dir, best_fold["fold"])
            remove_intermediate_plots(plots_dir, best_fold["fold"])
        except:
            _log.error(f"Failed making plot GIF.")

    heuristic_pred_file = f"{dirname}/heuristic_pred.csv"
    dir_split = dirname.split("/")[-1].split("_")
    problem_name = "_".join(dir_split[2:4])
    sample_seed = args.samples.split("_")[-1]
    data = {}

    if args.compare_csv_dir != "" and os.path.isfile(heuristic_pred_file):
        csv_dir = args.compare_csv_dir
        if csv_dir[-1] != "/":
            csv_dir += "/"
        problem_name = "_".join(dirname.split("/")[-1].split("_")[2:4])
        csv_h = glob.glob(csv_dir + problem_name + ".csv")

        if len(csv_h) > 0 and args.scatter_plot:
            try:
                _log.info(f"Saving h^nn vs. h scatter plot.")
                data = save_h_pred_scatter(plots_dir, heuristic_pred_file, csv_h[0])
            except:
                _log.error(f"Failed making hnn vs. h scatter plot.")

    if len(data) > 0 and args.hstar_csv_dir != "":
        csv_dir = args.hstar_csv_dir
        if csv_dir[-1] != "/":
            csv_dir += "/"
        csv_hstar = glob.glob(f"{csv_dir}*{problem_name}**{sample_seed}*.csv")

        if len(csv_hstar) > 0:
            try:
                _log.info(f"Saving box plot.")
                save_box_plot(plots_dir, data, csv_hstar[0])
            except:
                _log.error(f"Failed making box plot.")

    if args.save_heuristic_pred == False:
        os.remove(heuristic_pred_file)


if __name__ == "__main__":
    train_main(get_train_args())
