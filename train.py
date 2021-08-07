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
#   - Save best model, save top epochs?
#   - Debug logs

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

        train_wf.run(train_timer, validation=True)

        if train_timer.check_timeout():
            _log.info(
                f"Maximum training time reached. Stopping training."
            )
            break

        train_wf.save_traced_model(f"{dirname}/models/traced_fold{fold_idx}.pt")

    # TODO: get the best fold
    best_fold = 0
    copyfile(f"{dirname}/models/traced_fold{best_fold}.pt", f"{dirname}/traced_best.pt")
    _log.info(f"Saving traced_fold{best_fold}.pt as best model.")

    _log.info("Training complete!")


if __name__ == "__main__":
    train_main(get_train_args())
