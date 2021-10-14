import argparse
from pathlib import Path

from src.pytorch.utils.default_args import (
    DEFAULT_OUTPUT_LAYER,
    DEFAULT_LINEAR_OUTPUT,
    DEFAULT_NUM_FOLDS,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_HIDDEN_UNITS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ACTIVATION,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_SHUFFLE,
    DEFAULT_BIAS,
    DEFAULT_BIAS_OUTPUT,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_EPOCHS_NOT_IMPROVING,
    DEFAULT_MAX_TRAINING_TIME,
    DEFAULT_DOMAIN_PDDL,
    DEFAULT_SEARCH_ALGORITHM,
    DEFAULT_HEURISTIC,
    DEFAULT_UNARY_THRESHOLD,
    DEFAULT_MAX_SEARCH_TIME,
    DEFAULT_MAX_SEARCH_MEMORY,
    DEFAULT_MAX_EXPANSIONS,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_MODEL,
    DEFAULT_SCATTER_PLOT,
    DEFAULT_SCATTER_PLOT_N_EPOCHS,
    DEFAULT_HEURISTIC_MULTIPLIER,
    DEFAULT_WEIGHTS_METHOD,
    DEFAULT_WEIGHTS_SEED,
    DEFAULT_COMPARED_HEURISTIC_CSV_DIR,
    DEFAULT_HSTAR_CSV_DIR,
    DEFAULT_FACTS_FILE,
    DEFAULT_DEF_VALUES_FILE,
    DEFAULT_RESTART_NO_CONV,
)


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples",
        type=str,
        help="Path to file with samples to be used in training.",
    )
    parser.add_argument(
        "-o",
        "--output-layer",
        choices=["regression", "prefix", "one-hot"],
        default=DEFAULT_OUTPUT_LAYER,
        help="Network output layer type. (default: %(default)s)",
    )
    parser.add_argument(
        "-lo",
        "--linear-output",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=DEFAULT_LINEAR_OUTPUT,
        help="Use linear output in the output layer (True) or use an activation (False). (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--num-folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help="Number of folds to split training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        type=int,
        default=DEFAULT_HIDDEN_LAYERS,
        help="Number of hidden layers of the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-hu",
        "--hidden-units",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_UNITS,
        help='Number of units in each hidden layers. For all hidden layers with same size enter \
              only one value; for different size between layers enter "hidden_layers" values. \
              (default: scalable according to the input and output units.)',
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of samples used in each step of training. (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Network learning rate. (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Maximum number of epochs to train each fold (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-en",
        "--max-epochs-not-improving",
        type=int,
        default=DEFAULT_MAX_EPOCHS_NOT_IMPROVING,
        help="Stop training if loss does not improve after n epochs (-1 to disable). (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--max-training-time",
        type=int,
        default=DEFAULT_MAX_TRAINING_TIME,
        help="Maximum network training time (all folds). (default: %(default)ss)",
    )
    parser.add_argument(
        "-a",
        "--activation",
        choices=["sigmoid", "relu"],
        default=DEFAULT_ACTIVATION,
        help="Activation function for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        "--regularization",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
        help="Weight decay (L2 regularization) to use in network training. (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dropout-rate",
        type=float,
        default=DEFAULT_DROPOUT_RATE,
        help="Dropout rate for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-sh",
        "--shuffle",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=DEFAULT_SHUFFLE,
        help="Shuffle the training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-bi",
        "--bias",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=DEFAULT_BIAS,
        help="Use bias or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-biout",
        "--bias-output",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=DEFAULT_BIAS,
        help="Use bias or not in the output layer. (default: %(default)s)",
    )
    parser.add_argument(
        "-of",
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help="Path where the training folder will be saved. (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed to be used. Defaults to no seed. (default: random)",
    )
    parser.add_argument(
        "-sp",
        "--scatter-plot",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=DEFAULT_SCATTER_PLOT_N_EPOCHS,
        help="Create a scatter plot with y, predicted values. (default: %(default)s)",
    )
    parser.add_argument(
        "-spn",
        "--plot-n-epochs",
        type=int,
        default=DEFAULT_SCATTER_PLOT_N_EPOCHS,
        help="Do a scatter plot every n epochs. If -1, plot only after training. (default: %(default)s)",
    )
    parser.add_argument(
        "-wm",
        "--weights-method",
        choices=["none", "sqrt_k", "1", "xavier_uniform", "xavier_normal"],
        default=DEFAULT_WEIGHTS_METHOD,
        help="Inicialization of network weights. (default: %(default)s)",
    )
    parser.add_argument(
        "-ws",
        "--weights-seed",
        type=int,
        default=DEFAULT_WEIGHTS_SEED,
        help="Random seed to be used. Defaults to no seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-cdir",
        "--compare-csv-dir",
        type=str,
        default=DEFAULT_COMPARED_HEURISTIC_CSV_DIR,
        help="Directory with CSV data to compare h^nn against; used for plotting. (default: %(default)s)",
    )
    parser.add_argument(
        "-hdir",
        "--hstar-csv-dir",
        type=str,
        default=DEFAULT_COMPARED_HEURISTIC_CSV_DIR,
        help="Directory with h* CSV data; used for box plot. (default: %(default)s)",
    )
    parser.add_argument(
        "-rst",
        "--restart-no-conv",
        type=int,
        default=DEFAULT_RESTART_NO_CONV,
        help="Restart after n epochs of non-convergence. (default: %(default)s)",
    )

    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_folder", type=Path, help="Path to training folder with trained model."
    )
    parser.add_argument(
        "problem_pddls", type=str, nargs="+", help="Path to problems PDDL."
    )
    parser.add_argument(
        "-d",
        "--domain_pddl",
        type=str,
        default=DEFAULT_DOMAIN_PDDL,
        help="Path to domain PDDL. (default: problem_folder/domain.pddl)",
    )
    parser.add_argument(
        "-a",
        "--search-algorithm",
        choices=["astar", "eager_greedy"],
        default=DEFAULT_SEARCH_ALGORITHM,
        help="Algorithm to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-heu",
        "--heuristic",
        choices=["nn", "add", "blind", "ff", "goalcount", "hmax", "lmcut"],
        default=DEFAULT_HEURISTIC,
        help="Heuristic to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-hm",
        "--heuristic-multiplier",
        type=int,
        default=DEFAULT_HEURISTIC_MULTIPLIER,
        help="Value to multiply the output heuristic with. (default: %(default)s)",
    )
    parser.add_argument(
        "-u",
        "--unary-threshold",
        type=float,
        default=DEFAULT_UNARY_THRESHOLD,
        help="Unary threshold to be used if output layer is unary prefix. (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--max-search-time",
        type=int,
        default=DEFAULT_MAX_SEARCH_TIME,
        help="Time limit for each search. (default: %(default)ss)",
    )
    parser.add_argument(
        "-m",
        "--max-search-memory",
        type=int,
        default=DEFAULT_MAX_SEARCH_MEMORY,
        help="Memory limit for each search. (default: %(default)sMB)",
    )
    parser.add_argument(
        "-e",
        "--max-expansions",
        type=int,
        default=DEFAULT_MAX_EXPANSIONS,
        help="Maximum expanded states for each search (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-pt",
        "--test-model",
        choices=["all", "best"],
        default=DEFAULT_TEST_MODEL,
        help="Model(s) used for testing. (default: %(default)s)",
    )
    parser.add_argument(
        "-ffile",
        "--facts-file",
        type=str,
        default=DEFAULT_FACTS_FILE,
        help="Order of facts during sampling. (default: %(defaults)s)",
    )
    parser.add_argument(
        "-dfile",
        "--defaults-file",
        type=str,
        default=DEFAULT_DEF_VALUES_FILE,
        help="Default values for facts given with `ffile`. (default: %(defaults)s)",
    )

    return parser.parse_args()
