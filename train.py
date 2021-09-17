#!/usr/bin/env python3

import logging
import random
import numpy as np
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
   )
from src.pytorch.utils.plot import (
    save_y_pred_scatter,
    save_gif_from_plots,
    remove_intermediate_plots,
)
from src.pytorch.utils.parse_args import get_train_args
from src.pytorch.utils.timer import Timer

_log = logging.getLogger(__name__)


def train_main(args):
    if args.seed != -1:
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        random.seed(args.seed)
        np.random.seed(args.seed)

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

    for fold_idx in range(args.num_folds):
        _log.info(
            f"Running training workflow for fold {fold_idx+1} out "
            f"of {args.num_folds}"
        )
        train_dataloader, val_dataloader = kfold.get_fold(fold_idx)

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
            dirname=dirname,
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
            ),
        )

        fold_val_loss = train_wf.run(train_timer, validation=True)

        if fold_val_loss < best_fold["val_loss"]:
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

    try:
        _log.info(
            f"Saving post-training state,pred,y csv file to {dirname}/heuristic_pred.csv"
        )
        save_y_pred_csv(train_wf.y_pred_values, f"{dirname}/heuristic_pred.csv")
    except:
        _log.error(f"Failed to save csv file.")

    if args.scatter_plot and args.plot_n_epochs != -1:
        try:
            plots_dir = f"{dirname}/plots"
            _log.info(
                f"Saving plot GIF to {plots_dir}"
            )
            save_gif_from_plots(plots_dir)
            remove_intermediate_plots(plots_dir)
        except:
            _log.error(f"Failed making plot GIF.")
       
    _log.info("Training complete!")


if __name__ == "__main__":
    train_main(get_train_args())
