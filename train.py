#!/usr/bin/env python3

import logging
import torch
from shutil import copyfile

from src.pytorch.k_fold_training_data import KFoldTrainingData
from src.pytorch.model import HNN
from src.pytorch.train_workflow import TrainWorkflow
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import logging_train_config, create_train_directory
from src.pytorch.utils.parse_args import get_train_args
from src.pytorch.utils.timer import Timer

# TODO:
#   - Output layer arg

_log = logging.getLogger(__name__)

def train_main(args):
    dirname = create_train_directory(args)
    setup_full_logging(dirname)
    logging_train_config(args, dirname)

    kfold = KFoldTrainingData(args.samples,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        shuffle=args.shuffle)

    train_timer = Timer(args.max_training_time).start()
    best_fold = {"fold" : -1, "val_loss" : -1}

    for fold_idx in range(args.num_folds):
        _log.info(
            f"Running training workflow for fold {fold_idx+1} out "
            f"of {args.num_folds}"
        )
        train_dataloader, val_dataloader = kfold.get_fold(fold_idx)

        model = HNN(
            input_units=train_dataloader.dataset.x_shape()[1],
            output_units=train_dataloader.dataset.y_shape()[1],
            hidden_layers=args.hidden_layers,
            activation=args.activation,
            dropout_rate=args.dropout_rate
        ).to(torch.device("cpu"))

        if fold_idx == 0:
            _log.info(model)

        train_wf = TrainWorkflow(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_epochs=args.max_epochs,
            optimizer=torch.optim.Adam(
                model.parameters(),
                lr=args.learning_rate,
                weight_decay=args.weight_decay)
        )

        fold_val_loss = train_wf.run(train_timer, validation=True)

        if fold_val_loss < best_fold["val_loss"] or fold_idx == 0:
            _log.info(f"New best val loss at fold {fold_idx} = {fold_val_loss}")
            best_fold["fold"] = fold_idx
            best_fold["val_loss"] = fold_val_loss
        else:
            _log.info(
                f"Val loss at fold {fold_idx} = {fold_val_loss} (best = {best_fold['val_loss']})"
            )

        if train_timer.check_timeout():
            _log.info(
                f"Maximum training time reached. Stopping training."
            )
            break

        train_wf.save_traced_model(f"{dirname}/models/traced_fold{fold_idx}.pt")

    _log.info("Finishing training.")
    _log.info(f"Elapsed time: {train_timer.current_time()}")

    copyfile(f"{dirname}/models/traced_fold{best_fold['fold']}.pt", f"{dirname}/traced_best.pt")
    _log.info(
        f"Saving traced_fold{best_fold['fold']}.pt as best model (val loss = {best_fold['val_loss']})"
    )


    _log.info("Training complete!")


if __name__ == "__main__":
    train_main(get_train_args())
