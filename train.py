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
from src.pytorch.model_rtdl import ResNetRTDL
from src.pytorch.train_workflow import TrainWorkflow
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    get_fixed_max_epochs,
    add_train_arg,
    get_problem_by_sample_filename,
    get_memory_usage_mb,
    create_fake_samples,
    count_parameters,
)
from src.pytorch.utils.file_helpers import (
    create_train_directory,
    save_y_pred_csv,
    remove_csv_except_best,
)
from src.pytorch.utils.log_helpers import logging_train_config
from src.pytorch.utils.plot import (
    save_gif_from_plots,
    remove_intermediate_plots,
)
import src.pytorch.utils.default_args as default_args
from src.pytorch.utils.parse_args import get_train_args
from src.pytorch.utils.timer import Timer
from src.pytorch.utils.loss import RMSELoss
from eval import eval_workflow
from argparse import Namespace

_log = logging.getLogger(__name__)

def train_main(args: Namespace):
    """
    Higher-level setup of the full training procedure.
    """
    if args.num_cores != -1:
        torch.set_num_cores(args.num_cores)

    if args.samples.startswith("fake_"):
        _, samples_domain, samples_problem, samples_num = args.samples.split("_")
        args.samples = create_fake_samples(samples_domain, samples_problem, int(samples_num))
        if not args.samples:
            _log.error("Fake samples failed.")
            exit(0)

    set_seeds(args)

    args.domain, args.problem = get_problem_by_sample_filename(args.samples)

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
    _log.info(f"Elapsed time: {round(train_timer.current_time(), 4)}s")
    if num_retries:
        _log.info(f"Restarts needed: {num_retries}")

    if best_fold['fold'] != -1:
        if args.save_heuristic_pred:
            try:
                h_pred_files = glob.glob(f"{dirname}/heuristic_pred*.csv")
                if len(h_pred_files) > 1:
                    remove_csv_except_best(dirname, best_fold["fold"])
                    os.rename(
                        f"{dirname}/heuristic_pred_train_{best_fold['fold']}.csv",
                        f"{dirname}/heuristic_pred_train.csv",
                    )
                    os.rename(
                        f"{dirname}/heuristic_pred_val_{best_fold['fold']}.csv",
                        f"{dirname}/heuristic_pred_val.csv",
                    )
                else:
                    os.rename(
                        f"{dirname}/heuristic_pred_train_0.csv",
                        f"{dirname}/heuristic_pred_train.csv",
                    )
                    os.rename(
                        f"{dirname}/heuristic_pred_val_0.csv",
                        f"{dirname}/heuristic_pred_val.csv",
                    )
            except:
                _log.error(f"Failed to save heuristic_pred.csv.")
        try:
            if args.training_size != 1.0 and args.num_folds > 1:
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

        _log.info(f"Peak memory usage: {get_memory_usage_mb(True)} MB")
        _log.info("Training complete!")
    else:
        _log.error("Training incomplete! No trained networks.")

    if args.samples.startswith("fake_"):
        os.remove(args.samples)

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

    losses = {"mse": nn.MSELoss(), "rmse": RMSELoss()}
    chosen_loss_function = losses[args.loss_function]

    train_timer = Timer(args.max_training_time).start()
    while born_dead:
        starting_time = train_timer.current_time()
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
            sample_percentage=args.sample_percentage,
            unique_samples=args.unique_samples,
            unique_states=args.unique_states,
            model=args.model,
        )
        _log.info(f"Loading the training data took {round(train_timer.current_time() - starting_time, 4)}s.")

        if args.normalize_output:
            # Add the reference value in train_args.json to denormalize in the test
            add_train_arg(dirname, "max_h", kfold.domain_max_value)

        best_fold = {"fold": -1, "val_loss": float("inf")}

        for fold_idx in range(args.num_folds):
            _log.info(
                f"Running training workflow for fold {fold_idx+1} out "
                f"of {args.num_folds}"
            )

            train_dataloader, val_dataloader, test_dataloader = kfold.get_fold(fold_idx)

            if args.model == "resnet_rtdl":
                model = ResNetRTDL(
                    input_units=train_dataloader.dataset.x_shape()[1],
                    hidden_units=args.hidden_units[0],
                    output_units=train_dataloader.dataset.y_shape()[1],
                    num_layers=args.hidden_layers,
                    activation=args.activation,
                    output_layer=args.output_layer,
                    hidden_dropout=args.dropout_rate,
                    residual_dropout=args.dropout_rate,
                    linear_output=args.linear_output,
                    use_bias=args.bias,
                    use_bias_output=args.bias_output,
                    weights_method=args.weights_method,
                    model=args.model,
                ).to(device)
            else:
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
                    model=args.model,
                ).to(device)

            if fold_idx == 0:
                _log.info(f"\n{model}")
                count_parameters(model)

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
                check_dead_once=args.check_dead_once,
                loss_fn=chosen_loss_function,
                restart_no_conv=args.restart_no_conv,
                patience=args.patience,
            )

            fold_val_loss = train_wf.run(fold_idx, train_timer)

            born_dead = fold_val_loss is None
            if born_dead and args.num_folds == 1:
                args.seed += args.seed_increment_when_born_dead
                _log.info(f"Updated seed: {args.seed}")
                set_seeds(args, False)
                num_retries += 1
                add_train_arg(dirname, "updated_seed", args.seed)
            else:
                if args.save_heuristic_pred:
                    heuristic_pred_file_train = f"{dirname}/heuristic_pred_train_{fold_idx}.csv"
                    heuristic_pred_file_val = f"{dirname}/heuristic_pred_val_{fold_idx}.csv"

                if fold_val_loss is not None:
                    if fold_val_loss < best_fold["val_loss"]:
                        if args.save_heuristic_pred:
                            save_y_pred_csv(train_wf.train_y_pred_values, heuristic_pred_file_train)
                            save_y_pred_csv(train_wf.val_y_pred_values, heuristic_pred_file_val)
                        _log.info(f"New best val loss at fold {fold_idx} = {fold_val_loss}")
                        best_fold["fold"] = fold_idx
                        best_fold["val_loss"] = fold_val_loss
                    else:
                        _log.info(
                            f"Val loss at fold {fold_idx} = {fold_val_loss} (best = {best_fold['val_loss']})"
                        )
                else:  # Only using training data
                    if args.save_heuristic_pred:
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
        sample_percentage=args.sample_percentage,
        unique_samples=args.unique_samples,
        model=args.model,
    ).get_fold(0)

    if train_data is not None:
        eval_workflow(model, args.samples, dirname, train_data, "train", None, False, args.save_heuristic_pred, args.scatter_plot)
    if val_data is not None:
        eval_workflow(model, args.samples, dirname, val_data, "val", None, False, args.save_heuristic_pred, args.scatter_plot)
    if test_data is not None:
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

    if not args.save_heuristic_pred and os.path.exists(heuristic_pred_file):
        os.remove(heuristic_pred_file)


if __name__ == "__main__":
    train_main(get_train_args())
