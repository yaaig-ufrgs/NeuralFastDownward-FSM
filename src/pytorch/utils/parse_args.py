import argparse
from pathlib import Path

# TODO: add help message for each argument

from default_args import (
    DEFAULT_OUTPUT_LAYER,
    DEFAULT_NUM_FOLDS,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ACTIVATION,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_SHUFFLE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_TRAINING_TIME
)

def get_train_parser():
    parser = argparse.ArgumentParser(description="train description")
    parser.add_argument(
        'samples',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        "-o",
        "--output-layer",
        choices=["regression", "unary", "one-hot"],
        default=DEFAULT_OUTPUT_LAYER,
    )
    parser.add_argument(
        "-f",
        "--num-folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
    )
    parser.add_argument(
        "-hid",
        "--hidden-layers",
        type=int,
        default=DEFAULT_HIDDEN_LAYERS,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
    )
    parser.add_argument(
        "-t",
        "--max-training-time",
        type=int,
        default=DEFAULT_MAX_TRAINING_TIME,
    )
    parser.add_argument(
        "-a",
        "--activation",
        choices=["sigmoid", "relu"],
        default=DEFAULT_ACTIVATION,
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        "--regularization",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "-d",
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
    )
    parser.add_argument(
        "-s",
        "--shuffle",
        type=int,
        default=DEFAULT_SHUFFLE,
    )
    return parser.parse_args()
