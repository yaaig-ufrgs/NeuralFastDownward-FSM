#!/usr/bin/env python3

import os
import logging
import random
import numpy as np
import glob
import torch
from shutil import copyfile

from src.pytorch.k_fold_training_data import KFoldTrainingData
from src.pytorch.model import HNN
from src.pytorch.train_workflow import TrainWorkflow
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    logging_train_config,
    create_train_directory,
    get_fixed_max_epochs,
    save_y_pred_csv,
    remove_csv_except_best,
   )
from src.pytorch.utils.plot import (
    save_y_pred_scatter,
    save_h_pred_scatter,
    save_box_plot,
    save_gif_from_plots,
    remove_intermediate_plots,
)
from src.pytorch.utils.parse_args import get_train_args
from src.pytorch.utils.timer import Timer

_log = logging.getLogger(__name__)

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)

def train_main(args):
    if args.seed != -1:
        set_seeds(args.seed)

    dirname = create_train_directory(args)
    setup_full_logging(dirname)

    if len(args.hidden_units) not in [0, 1, args.hidden_layers]:
        _log.error("Invalid hidden_units length.")
        return
    if args.max_epochs == -1:
        args.max_epochs = get_fixed_max_epochs(dirname)

    logging_train_config(args, dirname)

    kfold = KFoldTrainingData(
        args.samples,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        output_layer=args.output_layer,
        shuffle=args.shuffle,
        seed=args.seed,
    )

    train_timer = Timer(args.max_training_time).start()
    best_fold = {"fold": -1, "val_loss": float("inf")}

    num_retries = 0
    for fold_idx in range(args.num_folds):
        _log.info(
            f"Running training workflow for fold {fold_idx+1} out "
            f"of {args.num_folds}"
        )

        train_dataloader, val_dataloader = kfold.get_fold(fold_idx)

        need_restart = True
        while need_restart:
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
                weights_seed=args.weights_seed,
            ).to(torch.device("cpu"))

            if fold_idx == 0:
                _log.info(model)

            train_wf = TrainWorkflow(
                model=model,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                max_epochs=args.max_epochs,
                plot_n_epochs=args.plot_n_epochs,
                max_epochs_not_improving = args.max_epochs_not_improving,
                max_epochs_no_convergence = args.restart_no_conv,
                dirname=dirname,
                optimizer=torch.optim.Adam(
                    model.parameters(),
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                ),
            )

            fold_val_loss, restart_flag = train_wf.run(fold_idx, train_timer, validation=True)
            need_restart = restart_flag
            if need_restart == True:
                # ????
                # In case of non-convergence, what makes more sense to restart:
                # - The _whole_ training setup, including data splitting in kfold?
                # - Or only restart the current fold?
                args.seed += 100
                print(args.seed)
                set_seeds(args.seed)
                num_retries += 1

        heuristic_pred_file = f"{dirname}/heuristic_pred_{fold_idx}.csv"

        if fold_val_loss < best_fold["val_loss"]:
            save_y_pred_csv(train_wf.y_pred_values, heuristic_pred_file)
            _log.info(f"New best val loss at fold {fold_idx} = {fold_val_loss}")
            best_fold["fold"] = fold_idx
            best_fold["val_loss"] = fold_val_loss
        else:
            _log.info(
                f"Val loss at fold {fold_idx} = {fold_val_loss} (best = {best_fold['val_loss']})"
            )

        train_wf.save_traced_model(f"{dirname}/models/traced_{fold_idx}.pt")
        if train_timer.check_timeout():
            _log.info(f"Maximum training time reached. Stopping training.")
            break

    _log.info("Finishing training.")
    _log.info(f"Elapsed time: {train_timer.current_time()}")
    _log.info(f"Restarts needed: {num_retries}")

    remove_csv_except_best(dirname, best_fold['fold'])
    os.rename(f"{dirname}/heuristic_pred_{best_fold['fold']}.csv", f"{dirname}/heuristic_pred.csv")

    try:
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

    ### PLOTTING
    plots_dir = f"{dirname}/plots"

    if args.scatter_plot and args.plot_n_epochs != -1:
        try:
            _log.info(
                f"Saving scatter plot GIF."
            )
            save_gif_from_plots(plots_dir, best_fold['fold'])
            remove_intermediate_plots(plots_dir, best_fold['fold'])
        except:
            _log.error(f"Failed making plot GIF.")
       
    heuristic_pred_file = f"{dirname}/heuristic_pred.csv"
    dir_split = dirname.split('/')[-1].split('_')
    problem_name = '_'.join(dir_split[2:4])
    sample_seed = dir_split[-1].split('.')[0]
    data = {}

    if args.compare_csv_dir != "" and os.path.isfile(heuristic_pred_file):
        csv_dir = args.compare_csv_dir
        if csv_dir[-1] != "/":
            csv_dir += "/"
        problem_name = '_'.join(dirname.split('/')[-1].split('_')[2:4])
        csv_h = glob.glob(csv_dir + problem_name + ".csv")

        if len(csv_h) > 0:
            try:
                _log.info(
                    f"Saving h^nn vs. h scatter plot."
                )
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
                _log.info(
                    f"Saving box plot."
                )
                save_box_plot(plots_dir, data, csv_hstar[0])
            except:
                _log.error(f"Failed making box plot.")


    _log.info("Training complete!")


if __name__ == "__main__":
    train_main(get_train_args())
