#!/usr/bin/env python3

import logging
import torch

from src.pytorch.k_fold_training_data import KFoldTrainingData
from src.pytorch.model import HNN
from src.pytorch.train_workflow import TrainWorkflow
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import logging_train_config, create_directory
from src.pytorch.utils.parse_args import get_train_args

# TODO: output_layer, max_training_time args

_log = logging.getLogger(__name__)

def train_main(args):
    dirname = create_directory(args)
    setup_full_logging(dirname)
    logging_train_config(args, dirname)

    kfold = KFoldTrainingData(args.samples,
        batch_size=args.batch_size,
        num_folds=args.num_folds,
        shuffle=args.shuffle)

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
            _log.info(
                f"{model}"
            )

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

        train_wf.run(validation=True)

        model_fname = f"dirname/models/traced_fold{fold_idx}.pt"
        train_wf.save_traced_model(str(model_fname))

if __name__ == "__main__":
    train_main(get_train_args())
