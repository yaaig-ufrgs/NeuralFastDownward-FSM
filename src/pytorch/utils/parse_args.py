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
    DEFAULT_SHUFFLE_SEED,
    DEFAULT_BIAS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_THREADS,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_TRAINING_TIME,
    DEFAULT_DOMAIN_PDDL,
    DEFAULT_SEARCH_ALGORITHM,
    DEFAULT_HEURISTIC,
    DEFAULT_UNARY_THRESHOLD,
    DEFAULT_MAX_SEARCH_TIME,
    DEFAULT_MAX_SEARCH_MEMORY,
    DEFAULT_MAX_EXPANSIONS,
    DEFAULT_OUTPUT_FOLDER,
    DEFAULT_MODEL,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_TEST_MODEL,
    DEFAULT_SCATTER_PLOT,
    DEFAULT_SCATTER_PLOT_N_EPOCHS,
    DEFAULT_HEURISTIC_MULTIPLIER,
    DEFAULT_WEIGHTS_METHOD,
    DEFAULT_WEIGHTS_SEED,
    DEFAULT_COMPARED_HEURISTIC_CSV_DIR,
    DEFAULT_FACTS_FILE,
    DEFAULT_DEF_VALUES_FILE,
    DEFAULT_NORMALIZE_OUTPUT,
    DEFAULT_SEED_INCREMENT_WHEN_BORN_DEAD,
    DEFAULT_RESTART_NO_CONV,
    DEFAULT_SAVE_HEURISTIC_PRED,
    DEFAULT_AUTO_TASKS_N,
    DEFAULT_AUTO_TASKS_FOLDER,
    DEFAULT_AUTO_TASKS_SEED,
    DEFAULT_SAMPLES_FOLDER,
)


def get_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "samples",
        type=str,
        help="Path to file with samples to be used in training.",
    )
    parser.add_argument(
        "-mdl",
        "--model",
        choices=["hnn", "resnet"],
        default=DEFAULT_MODEL,
        help="Network model to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-pat",
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stop patience. (default: %(default)s)",
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
        type=str2bool,
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
        "-t",
        "--max-training-time",
        type=int,
        default=DEFAULT_MAX_TRAINING_TIME,
        help="Maximum network training time (all folds). (default: %(default)ss)",
    )
    parser.add_argument(
        "-a",
        "--activation",
        choices=["sigmoid", "relu", "leakyrelu"],
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
        "-shs",
        "--shuffle-seed",
        type=int,
        default=DEFAULT_SHUFFLE_SEED,
        help="Seed to be used for separating training and validation data. Defaults to network seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-sh",
        "--shuffle",
        type=str2bool,
        default=DEFAULT_SHUFFLE,
        help="Shuffle the training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-bi",
        "--bias",
        type=str2bool,
        default=DEFAULT_BIAS,
        help="Use bias or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-biout",
        "--bias-output",
        type=str2bool,
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
        type=str2bool,
        default=DEFAULT_SCATTER_PLOT,
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
        choices=[
            "default",
            "sqrt_k",
            "1",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "rai",
        ],
        default=DEFAULT_WEIGHTS_METHOD,
        help="Inicialization of network weights. (default: %(default)s)",
    )
    parser.add_argument(
        "-ws",
        "--weights-seed",
        type=int,
        default=DEFAULT_WEIGHTS_SEED,
        help="Random seed to be used. Defaults to equal to main seed. (default: %(default)s)",
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
        "-no",
        "--normalize-output",
        type=str2bool,
        default=DEFAULT_NORMALIZE_OUTPUT,
        help="Normalizes the output neuron. (default: %(default)s)",
    )
    parser.add_argument(
        "-rst",
        "--restart-no-conv",
        type=str2bool,
        default=DEFAULT_RESTART_NO_CONV,
        help="Restarts the network if it won't converge. (default: %(default)s)",
    )
    parser.add_argument(
        "-sibd",
        "--seed-increment-when-born-dead",
        type=int,
        default=DEFAULT_SEED_INCREMENT_WHEN_BORN_DEAD,
        help="Seed increment when the network needs to restart due to born dead. (default: %(default)s)",
    )
    parser.add_argument(
        "-trd",
        "--num-threads",
        type=int,
        default=DEFAULT_NUM_THREADS,
        help="Number of threads used for intra operations on CPU (PyTorch). (default: %(default)s)",
    )
    parser.add_argument(
        "-hpred",
        "--save-heuristic-pred",
        type=str2bool,
        default=DEFAULT_SAVE_HEURISTIC_PRED,
        help="Save a csv file with the expected and network-predicted heuristics for all training samples. (default: %(default)s)",
    )
    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_folder",
        type=Path,
        help="Path to training folder with trained model."
    )
    parser.add_argument(
        "-tfc",
        "--train_folder_compare",
        type=str,
        default="",
        help="Trained folder to be used for comparison agains the main model."
    )
    parser.add_argument(
        "problem_pddls",
        type=str,
        nargs="*",
        default=[],
        help="Path to problems PDDL."
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
        "-sdir",
        "--samples-dir",
        type=str,
        default=DEFAULT_SAMPLES_FOLDER,
        help="Default samples directory to automatically get facts and defaults files. (default: %(default)s)",
    )
    parser.add_argument(
        "-ffile",
        "--facts-file",
        type=str,
        default=DEFAULT_FACTS_FILE,
        help="Order of facts during sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "-dfile",
        "--defaults-file",
        type=str,
        default=DEFAULT_DEF_VALUES_FILE,
        help="Default values for facts given with `ffile`. (default: %(default)s)",
    )
    parser.add_argument(
        "-atn",
        "--auto-tasks-n",
        type=int,
        default=DEFAULT_AUTO_TASKS_N,
        help="Number of tasks taken automatically. (default: %(default)s)",
    )
    parser.add_argument(
        "-atf",
        "--auto-tasks-folder",
        type=str,
        default=DEFAULT_AUTO_TASKS_FOLDER,
        help="Base folder to search for tasks automatically. (default: %(default)s)"
    )
    parser.add_argument(
        "-ats",
        "--auto-tasks-seed",
        type=int,
        default=DEFAULT_AUTO_TASKS_SEED,
        help="Seed to shuffle the tasks taken automatically. (default: %(default)s)",
    )
    return parser.parse_args()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
