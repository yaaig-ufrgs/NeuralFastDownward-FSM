import argparse
from pathlib import Path

import src.pytorch.utils.default_args as default_args


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
        default=default_args.MODEL,
        help="Network model to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-sb",
        "--save-best-epoch-model",
        type=str2bool,
        default=default_args.SAVE_BEST_EPOCH_MODEL,
        help="Saves the best model from the best epoch instead of the last one. (default: %(default)s)",
    )
    parser.add_argument(
        "-diff",
        "--save-git-diff",
        type=str2bool,
        default=default_args.SAVE_GIT_DIFF,
        help="Saves git diff or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-pte",
        "--post-train-eval",
        type=str2bool,
        default=default_args.POST_TRAIN_EVAL,
        help="Perform a post-training evaluation on the trained model with the same dataset used in training. (default: %(default)s)",
    )
    parser.add_argument(
        "-pat",
        "--patience",
        type=int,
        default=default_args.PATIENCE,
        help="Early-stop patience. (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-layer",
        choices=["regression", "prefix", "one-hot"],
        default=default_args.OUTPUT_LAYER,
        help="Network output layer type. (default: %(default)s)",
    )
    parser.add_argument(
        "-lo",
        "--linear-output",
        type=str2bool,
        default=default_args.LINEAR_OUTPUT,
        help="Use linear output in the output layer (True) or use an activation (False). (default: %(default)s)",
    )
    parser.add_argument(
        "-f",
        "--num-folds",
        type=int,
        default=default_args.NUM_FOLDS,
        help="Number of folds to split training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-hl",
        "--hidden-layers",
        type=int,
        default=default_args.HIDDEN_LAYERS,
        help="Number of hidden layers of the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-hu",
        "--hidden-units",
        type=int,
        nargs="+",
        default=default_args.HIDDEN_UNITS,
        help='Number of units in each hidden layers. For all hidden layers with same size enter \
              only one value; for different size between layers enter "hidden_layers" values. \
              Use 0 to make it scalable according to the input and output units. (default: 250)',
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=default_args.BATCH_SIZE,
        help="Number of samples used in each step of training. (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=default_args.LEARNING_RATE,
        help="Network learning rate. (default: %(default)s)",
    )
    parser.add_argument(
        "-e",
        "--max-epochs",
        type=int,
        default=default_args.MAX_EPOCHS,
        help="Maximum number of epochs to train each fold (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--max-training-time",
        type=int,
        default=default_args.MAX_TRAINING_TIME,
        help="Maximum network training time (all folds). (default: %(default)ss)",
    )
    parser.add_argument(
        "-a",
        "--activation",
        choices=["sigmoid", "relu", "leakyrelu"],
        default=default_args.ACTIVATION,
        help="Activation function for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-w",
        "--weight-decay",
        "--regularization",
        type=float,
        default=default_args.WEIGHT_DECAY,
        help="Weight decay (L2 regularization) to use in network training. (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--dropout-rate",
        type=float,
        default=default_args.DROPOUT_RATE,
        help="Dropout rate for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-shs",
        "--shuffle-seed",
        type=int,
        default=default_args.SHUFFLE_SEED,
        help="Seed to be used for separating training and validation data. Defaults to network seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-sh",
        "--shuffle",
        type=str2bool,
        default=default_args.SHUFFLE,
        help="Shuffle the training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-gpu",
        "--use-gpu",
        type=str2bool,
        default=default_args.USE_GPU,
        help="Use GPU during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-bi",
        "--bias",
        type=str2bool,
        default=default_args.BIAS,
        help="Use bias or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-tsize",
        "--training-size",
        type=float,
        default=default_args.TRAINING_SIZE,
        help="Training data size in relation to validation data. (default: %(default)s)",
    )
    parser.add_argument(
        "-spt",
        "--sample-percentage",
        type=float,
        default=default_args.SAMPLE_PERCENTAGE,
        help="Sample percentage to be used during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-us",
        "--unique-samples",
        type=str2bool,
        default=default_args.UNIQUE_SAMPLES,
        help="Remove repeated samples from data. (default: %(default)s)",
    )
    parser.add_argument(
        "-ust",
        "--unique-states",
        type=str2bool,
        default=default_args.UNIQUE_STATES,
        help="Remove repeated states (only x) from data. (default: %(default)s)",
    )
    parser.add_argument(
        "-biout",
        "--bias-output",
        type=str2bool,
        default=default_args.BIAS,
        help="Use bias or not in the output layer. (default: %(default)s)",
    )
    parser.add_argument(
        "-of",
        "--output-folder",
        type=Path,
        default=default_args.OUTPUT_FOLDER,
        help="Path where the training folder will be saved. (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=default_args.RANDOM_SEED,
        help="Random seed to be used. Defaults to no seed. (default: random)",
    )
    parser.add_argument(
        "-sp",
        "--scatter-plot",
        type=str2bool,
        default=default_args.SCATTER_PLOT,
        help="Create a scatter plot with y, predicted values. (default: %(default)s)",
    )
    parser.add_argument(
        "-spn",
        "--plot-n-epochs",
        type=int,
        default=default_args.SCATTER_PLOT_N_EPOCHS,
        help="Do a scatter plot every n epochs. If -1, plot only after training. (default: %(default)s)",
    )
    parser.add_argument(
        "-wm",
        "--weights-method",
        choices=[
            "default",
            "sqrt_k",
            "1",
            "01",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "rai",
        ],
        default=default_args.WEIGHTS_METHOD,
        help="Initialization of network weights. (default: %(default)s)",
    )
    parser.add_argument(
        "-lf",
        "--loss-function",
        choices=[
            "mse",
            "rmse",
        ],
        default=default_args.LOSS_FUNCTION,
        help="Loss function to be used during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-no",
        "--normalize-output",
        type=str2bool,
        default=default_args.NORMALIZE_OUTPUT,
        help="Normalizes the output neuron. (default: %(default)s)",
    )
    parser.add_argument(
        "-rst",
        "--restart-no-conv",
        type=str2bool,
        default=default_args.RESTART_NO_CONV,
        help="Restarts the network if it won't converge. (default: %(default)s)",
    )
    parser.add_argument(
        "-cdead",
        "--check-dead-once",
        type=str2bool,
        default=default_args.CHECK_DEAD_ONCE,
        help="Only check if network is dead once, at the start of the first epoch. (default: %(default)s)",
    )
    parser.add_argument(
        "-sibd",
        "--seed-increment-when-born-dead",
        type=int,
        default=default_args.SEED_INCREMENT_WHEN_BORN_DEAD,
        help="Seed increment when the network needs to restart due to born dead. (default: %(default)s)",
    )
    parser.add_argument(
        "-trd",
        "--num-cores",
        type=int,
        default=default_args.NUM_CORES,
        help="Number of cores used for intra operations on CPU (PyTorch). (default: %(default)s)",
    )
    parser.add_argument(
        "-dnw",
        "--data-num-workers",
        type=int,
        default=default_args.DATALOADER_NUM_WORKERS,
        help="Number of workers for multi-process data loading. (default: %(default)s)",
    )
    parser.add_argument(
        "-hpred",
        "--save-heuristic-pred",
        type=str2bool,
        default=default_args.SAVE_HEURISTIC_PRED,
        help="Save a csv file with the expected and network-predicted heuristics for all training samples. (default: %(default)s)",
    )
    parser.add_argument(
        "-addfn",
        "--additional-folder-name",
        nargs="*",
        choices=[
            "patience",
            "output-layer",
            "num-folds",
            "hidden-layers",
            "hidden-units",
            "batch-size",
            "learning-rate",
            "max-epochs",
            "max-training-time",
            "activation",
            "weight-decay",
            "dropout-rate",
            "shuffle-seed",
            "shuffle",
            "use-gpu",
            "bias",
            "bias-output",
            "normalize-output",
            "restart-no-conv",
            "sample-percentage",
            "training-size",
        ],
        default=default_args.ADDITIONAL_FOLDER_NAME,
        help="Allows to add parameters to the folder name. (default: %(default)s)",
    )

    return parser.parse_args()


def get_test_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train_folder", type=Path, help="Path to training folder with trained model."
    )
    parser.add_argument(
        "-tfc",
        "--train-folder-compare",
        type=str,
        default="",
        help="Trained folder to be used for comparison agains the main model.",
    )
    parser.add_argument(
        "problem_pddls", type=str, nargs="*", default=[], help="Path to problems PDDL."
    )
    parser.add_argument(
        "-diff",
        "--save-git-diff",
        type=str2bool,
        default=default_args.SAVE_GIT_DIFF,
        help="Saves git diff or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--domain_pddl",
        type=str,
        default=default_args.DOMAIN_PDDL,
        help="Path to domain PDDL. (default: problem_folder/domain.pddl)",
    )
    parser.add_argument(
        "-a",
        "--search-algorithm",
        choices=["astar", "eager_greedy"],
        default=default_args.SEARCH_ALGORITHM,
        help="Algorithm to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-heu",
        "--heuristic",
        choices=["nn", "add", "blind", "ff", "goalcount", "hmax", "lmcut", "hstar"],
        default=default_args.HEURISTIC,
        help="Heuristic to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-hm",
        "--heuristic-multiplier",
        type=int,
        default=default_args.HEURISTIC_MULTIPLIER,
        help="Value to multiply the output heuristic with. (default: %(default)s)",
    )
    parser.add_argument(
        "-u",
        "--unary-threshold",
        type=float,
        default=default_args.UNARY_THRESHOLD,
        help="Unary threshold to be used if output layer is unary prefix. (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--max-search-time",
        type=int,
        default=default_args.MAX_SEARCH_TIME,
        help="Time limit for each search. (default: %(default)ss)",
    )
    parser.add_argument(
        "-m",
        "--max-search-memory",
        type=int,
        default=default_args.MAX_SEARCH_MEMORY,
        help="Memory limit for each search. (default: %(default)sMB)",
    )
    parser.add_argument(
        "-e",
        "--max-expansions",
        type=int,
        default=default_args.MAX_EXPANSIONS,
        help="Maximum expanded states for each search (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-pt",
        "--test-model",
        choices=["all", "best", "epochs"],
        default=default_args.TEST_MODEL,
        help="Model(s) used for testing. (default: %(default)s)",
    )
    parser.add_argument(
        "-sdir",
        "--samples-dir",
        type=str,
        default=default_args.SAMPLES_FOLDER,
        help="Default samples directory to automatically get facts and defaults files. (default: get from the samples file)",
    )
    parser.add_argument(
        "-ffile",
        "--facts-file",
        type=str,
        default=default_args.FACTS_FILE,
        help="Order of facts during sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "-dfile",
        "--defaults-file",
        type=str,
        default=default_args.DEF_VALUES_FILE,
        help="Default values for facts given with `ffile`. (default: %(default)s)",
    )
    parser.add_argument(
        "-atn",
        "--auto-tasks-n",
        type=int,
        default=default_args.AUTO_TASKS_N,
        help="Number of tasks taken automatically. (default: %(default)s)",
    )
    parser.add_argument(
        "-atf",
        "--auto-tasks-folder",
        type=str,
        default=default_args.AUTO_TASKS_FOLDER,
        help="Base folder to search for tasks automatically. (default: %(default)s)",
    )
    parser.add_argument(
        "-ats",
        "--auto-tasks-seed",
        type=int,
        default=default_args.AUTO_TASKS_SEED,
        help="Seed to shuffle the tasks taken automatically. (default: %(default)s)",
    )
    parser.add_argument(
        "-dlog",
        "--downward-logs",
        type=str2bool,
        default=default_args.SAVE_DOWNWARD_LOGS,
        help="Save each instance's Fast-Downward log or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-unit-cost",
        "--unit-cost",
        type=str2bool,
        default=default_args.UNIT_COST,
        help="Test with unit cost instead of operator cost. (default: %(default)s)",
    )

    return parser.parse_args()


def get_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trained_model",
        type=str,
        help="Path to an already trained model to be evaluated.",
    )
    parser.add_argument(
        "samples",
        type=str,
        nargs="*",
        default=[],
        help="Path to file with states to be evaluated on",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=default_args.EVAL_SEED,
        help="Random seed to be used. Defaults to no seed. (default: random)",
    )
    parser.add_argument(
        "-shs",
        "--shuffle-seed",
        type=int,
        default=default_args.EVAL_SHUFFLE_SEED,
        help="Seed to be used for separating training and validation data. (default: %(default)s)",
    )
    parser.add_argument(
        "-sh",
        "--shuffle",
        type=str2bool,
        default=default_args.EVAL_SHUFFLE,
        help="Shuffle the training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-tsize",
        "--training-size",
        type=float,
        default=default_args.EVAL_TRAINING_SIZE,
        help="Training data size in relation to validation data. Change it in case you want to separate training data from validation data. (default: %(default)s)",
    )
    parser.add_argument(
        "-ft",
        "--follow-training",
        type=str2bool,
        default=default_args.FOLLOW_TRAIN,
        help="Follow original training config when it comes to training data size, shuffling and seeds. (default: %(default)s)",
    )
    parser.add_argument(
        "-us",
        "--unique-samples",
        type=str2bool,
        default=default_args.UNIQUE_SAMPLES,
        help="Remove repeated samples (x and y) from data. (default: %(default)s)",
    )
    parser.add_argument(
        "-ls",
        "--log-states",
        type=str2bool,
        default=default_args.LOG_STATES,
        help="Detailed logging of states with their predictions. (default: %(default)s)",
    )
    parser.add_argument(
        "-sp",
        "--save-preds",
        type=str2bool,
        default=default_args.SAVE_PREDS,
        help="Save heuristic prediction files for every state. (default: %(default)s)",
    )
    parser.add_argument(
        "-plt",
        "--save-plots",
        type=str2bool,
        default=default_args.SAVE_PLOTS,
        help="Save plots related to the evaluation. (default: %(default)s)",
    )

    return parser.parse_args()


def get_exp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp-type",
        "--exp-type",
        choices=[
            "single",
            "all",
            "combined",
        ],
        default=default_args.EXP_TYPE,
        help="Experiment type according to seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-ns",
        "--exp-net-seed",
        type=str,
        default=default_args.EXP_NET_SEED,
        help="Inclusive network seed interval to be used in experiments. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-ss",
        "--exp-sample-seed",
        type=str,
        default=default_args.EXP_SAMPLE_SEED,
        help="Inclusive sample seed interval to be used in experiments. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-cores",
        "--exp-cores",
        type=int,
        default=default_args.EXP_CORES,
        help="Number of cores to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-train",
        "--exp-only-train",
        type=str2bool,
        default=default_args.EXP_ONLY_TRAIN,
        help="Only train instead of training and testing. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-test",
        "--exp-only-test",
        type=str2bool,
        default=default_args.EXP_ONLY_TEST,
        help="Only test instead of training and testing. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-eval",
        "--exp-only-eval",
        type=str2bool,
        default=default_args.EXP_ONLY_EVAL,
        help="Only evaluate instead of training and testing. (default: %(default)s)",
    )
    parser.add_argument(
        "samples",
        type=str,
        help="Path to sample files.",
    )
    parser.add_argument(
        "-trn-mdl",
        "--train-model",
        choices=["hnn", "resnet"],
        default=default_args.MODEL,
        help="Network model to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-diff",
        "--train-save-git-diff",
        type=str2bool,
        default=default_args.SAVE_GIT_DIFF,
        help="Saves git diff or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-pte",
        "--train-post-train-eval",
        type=str2bool,
        default=default_args.POST_TRAIN_EVAL,
        help="Perform a post-training evaluation on the trained model with the same dataset used in training. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-sb",
        "--train-save-best-epoch-model",
        type=str2bool,
        default=default_args.SAVE_BEST_EPOCH_MODEL,
        help="Saves the best model from the best epoch instead of the last one. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-pat",
        "--train-patience",
        type=int,
        default=default_args.PATIENCE,
        help="Early-stop patience. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-hl",
        "--train-hidden-layers",
        type=int,
        default=default_args.HIDDEN_LAYERS,
        help="Number of hidden layers of the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-hu",
        "--train-hidden-units",
        type=int,
        nargs="+",
        default=default_args.HIDDEN_UNITS,
        help='Number of units in each hidden layers. For all hidden layers with same size enter \
              only one value; for different size between layers enter "hidden_layers" values. \
              (default: scalable according to the input and output units.)',
    )
    parser.add_argument(
        "-train-t",
        "--train-max-training-time",
        type=int,
        default=default_args.MAX_TRAINING_TIME,
        help="Maximum network training time (all folds). (default: %(default)ss)",
    )
    parser.add_argument(
        "-trn-b",
        "--train-batch-size",
        type=int,
        default=default_args.BATCH_SIZE,
        help="Number of samples used in each step of training. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-o",
        "--train-output-layer",
        choices=["regression", "prefix", "one-hot"],
        default=default_args.OUTPUT_LAYER,
        help="Network output layer type. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-lo",
        "--train-linear-output",
        type=str2bool,
        default=default_args.LINEAR_OUTPUT,
        help="Use linear output in the output layer (True) or use an activation (False). (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-f",
        "--train-num-folds",
        type=int,
        default=default_args.NUM_FOLDS,
        help="Number of folds to split training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-lr",
        "--train-learning-rate",
        type=float,
        default=default_args.LEARNING_RATE,
        help="Network learning rate. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-e",
        "--train-max-epochs",
        type=int,
        default=default_args.MAX_EPOCHS,
        help="Maximum number of epochs to train each fold (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-w",
        "--train-weight-decay",
        "--regularization",
        type=float,
        default=default_args.WEIGHT_DECAY,
        help="Weight decay (L2 regularization) to use in network training. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-d",
        "--train-dropout-rate",
        type=float,
        default=default_args.DROPOUT_RATE,
        help="Dropout rate for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-shs",
        "--train-shuffle-seed",
        type=int,
        default=default_args.SHUFFLE_SEED,
        help="Seed to be used for separating training and validation data. Defaults to network seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-sh",
        "--train-shuffle",
        type=str2bool,
        default=default_args.SHUFFLE,
        help="Shuffle the training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-tsize",
        "--train-training-size",
        type=float,
        default=default_args.TRAINING_SIZE,
        help="Training data size in relation to validation data. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-spt",
        "--train-sample-percentage",
        type=float,
        default=default_args.SAMPLE_PERCENTAGE,
        help="Sample percentage to be used during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-us",
        "--train-unique-samples",
        type=str2bool,
        default=default_args.UNIQUE_SAMPLES,
        help="Remove repeated samples from data. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-ust",
        "--train-unique-states",
        type=str2bool,
        default=default_args.UNIQUE_STATES,
        help="Remove repeated states (only x) from data. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-lf",
        "--train-loss-function",
        choices=[
            "mse",
            "rmse",
        ],
        default=default_args.LOSS_FUNCTION,
        help="Loss function to be used during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-wm",
        "--train-weights-method",
        choices=[
            "default",
            "sqrt_k",
            "1",
            "01",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
            "rai",
        ],
        default=default_args.WEIGHTS_METHOD,
        help="Inicialization of network weights. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-bi",
        "--train-bias",
        type=str2bool,
        default=default_args.BIAS,
        help="Use bias or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-biout",
        "--train-bias-output",
        type=str2bool,
        default=default_args.BIAS,
        help="Use bias or not in the output layer. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-of",
        "--train-output-folder",
        type=Path,
        default=default_args.OUTPUT_FOLDER,
        help="Path where the training folder will be saved. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-a",
        "--train-activation",
        choices=["sigmoid", "relu", "leakyrelu"],
        default=default_args.ACTIVATION,
        help="Activation function for hidden layers. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-s",
        "--train-seed",
        type=int,
        default=default_args.RANDOM_SEED,
        help="Random seed to be used. Defaults to no seed. (default: random)",
    )
    parser.add_argument(
        "-trn-rst",
        "--train-restart-no-conv",
        type=str2bool,
        default=default_args.RESTART_NO_CONV,
        help="Restarts the network if it won't converge. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-cdead",
        "--train-check-dead-once",
        type=str2bool,
        default=default_args.CHECK_DEAD_ONCE,
        help="Only check if network is dead once, at the start of the first epoch. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-sibd",
        "--train-seed-increment-when-born-dead",
        type=int,
        default=default_args.SEED_INCREMENT_WHEN_BORN_DEAD,
        help="Seed increment when the network needs to restart due to born dead. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-trd",
        "--train-num-cores",
        type=int,
        default=default_args.NUM_CORES,
        help="Number of cores used for intra operations on CPU (PyTorch). (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-dnw",
        "--train-data-num-workers",
        type=int,
        default=default_args.DATALOADER_NUM_WORKERS,
        help="Number of workers for multi-process data loading. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-hpred",
        "--train-save-heuristic-pred",
        type=str2bool,
        default=default_args.SAVE_HEURISTIC_PRED,
        help="Save a csv file with the expected and network-predicted heuristics for all training samples. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-gpu",
        "--train-use-gpu",
        type=str2bool,
        default=default_args.USE_GPU,
        help="Use GPU during training. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-sp",
        "--train-scatter-plot",
        type=str2bool,
        default=default_args.SCATTER_PLOT,
        help="Create a scatter plot with y, predicted values. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-spn",
        "--train-plot-n-epochs",
        type=int,
        default=default_args.SCATTER_PLOT_N_EPOCHS,
        help="Do a scatter plot every n epochs. If -1, plot only after training. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-no",
        "--train-normalize-output",
        type=str2bool,
        default=default_args.NORMALIZE_OUTPUT,
        help="Normalizes the output neuron. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-addfn",
        "--train-additional-folder-name",
        nargs="*",
        choices=[
            "patience",
            "output-layer",
            "num-folds",
            "hidden-layers",
            "hidden-units",
            "batch-size",
            "learning-rate",
            "max-epochs",
            "max-training-time",
            "activation",
            "weight-decay",
            "dropout-rate",
            "shuffle-seed",
            "shuffle",
            "use-gpu",
            "bias",
            "bias-output",
            "normalize-output",
            "restart-no-conv",
        ],
        default=default_args.ADDITIONAL_FOLDER_NAME,
        help="Allows to add parameters to the folder name. (default: %(default)s)",
    )
    parser.add_argument(
        "problem_pddls", type=str, nargs="*", default=[], help="Path to problems PDDL."
    )
    parser.add_argument(
        "-tst-modeldir",
        "--test-model-dir",
        type=Path,
        help="Path to training folder with trained model. Only used if only testing.",
    )
    parser.add_argument(
        "-tst-pddl",
        "--test-instance-pddl",
        type=str,
        default="",
        help="Instance PDDL used for testing. (default: auto)",
    )
    parser.add_argument(
        "-tst-diff",
        "--test-save-git-diff",
        type=str2bool,
        default=default_args.SAVE_GIT_DIFF,
        help="Saves git diff or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-pt",
        "--test-test-model",
        choices=["all", "best", "epochs"],
        default=default_args.TEST_MODEL,
        help="Model(s) used for testing. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-a",
        "--test-search-algorithm",
        choices=["astar", "eager_greedy"],
        default=default_args.SEARCH_ALGORITHM,
        help="Algorithm to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-heu",
        "--test-heuristic",
        choices=["nn", "add", "blind", "ff", "goalcount", "hmax", "lmcut"],
        default=default_args.HEURISTIC,
        help="Heuristic to be used in the search. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-t",
        "--test-max-search-time",
        type=int,
        default=default_args.MAX_SEARCH_TIME,
        help="Time limit for each search. (default: %(default)ss)",
    )
    parser.add_argument(
        "-tst-m",
        "--test-max-search-memory",
        type=int,
        default=default_args.MAX_SEARCH_MEMORY,
        help="Memory limit for each search. (default: %(default)sMB)",
    )
    parser.add_argument(
        "-tst-e",
        "--test-max-expansions",
        type=int,
        default=default_args.MAX_EXPANSIONS,
        help="Maximum expanded states for each search (or -1 for fixed value). (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-sdir",
        "--test-samples-dir",
        type=str,
        default=default_args.SAMPLES_FOLDER,
        help="Default samples directory to automatically get facts and defaults files. (default: get from the samples file)",
    )
    parser.add_argument(
        "-tst-atn",
        "--test-auto-tasks-n",
        type=int,
        default=default_args.AUTO_TASKS_N,
        help="Number of tasks taken automatically. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-atf",
        "--test-auto-tasks-folder",
        type=str,
        default=default_args.AUTO_TASKS_FOLDER,
        help="Base folder to search for tasks automatically. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-ats",
        "--test-auto-tasks-seed",
        type=int,
        default=default_args.AUTO_TASKS_SEED,
        help="Seed to shuffle the tasks taken automatically. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-dlog",
        "--test-downward-logs",
        type=str2bool,
        default=default_args.SAVE_DOWNWARD_LOGS,
        help="Save each instance's Fast-Downward log or not. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-unit-cost",
        "--test-unit-cost",
        type=str2bool,
        default=default_args.UNIT_COST,
        help="Test with unit cost instead of operator cost. (default: %(default)s)",
    )
    parser.add_argument(
        "-evl-mdl",
        "--eval-trained-models",
        type=str,
        nargs="*",
        default=[],
        help="Path to already trained models to be evaluated on.",
    )
    parser.add_argument(
        "-evl-smp",
        "--eval-sample",
        type=str,
        default=default_args.EXP_EVAL_SAMPLE,
        help="Sample used to evaluate the network over.",
    )
    parser.add_argument(
        "-evl-s",
        "--eval-seed",
        type=int,
        default=default_args.EVAL_SEED,
        help="Random seed to be used. Defaults to no seed. (default: random)",
    )
    parser.add_argument(
        "-evl-shs",
        "--eval-shuffle-seed",
        type=int,
        default=default_args.EVAL_SHUFFLE_SEED,
        help="Seed to be used for separating training and validation data. (default: %(default)s)",
    )
    parser.add_argument(
        "-eval-sh",
        "--eval-shuffle",
        type=str2bool,
        default=default_args.EVAL_SHUFFLE,
        help="Shuffle the training data. (default: %(default)s)",
    )
    parser.add_argument(
        "-evl-tsize",
        "--eval-training-size",
        type=float,
        default=default_args.EVAL_TRAINING_SIZE,
        help="Training data size in relation to validation data. Change it in case you want to separate training data from validation data. (default: %(default)s)",
    )
    parser.add_argument(
        "-eval-ft",
        "--eval-follow-training",
        type=str2bool,
        default=default_args.FOLLOW_TRAIN,
        help="Follow original training config when it comes to training data size, shuffling and seeds. (default: %(default)s)",
    )
    parser.add_argument(
        "-eval-us",
        "--eval-unique-samples",
        type=str2bool,
        default=default_args.UNIQUE_SAMPLES,
        help="Remove repeated samples (x and y) from data. (default: %(default)s)",
    )
    parser.add_argument(
        "-evl-ls",
        "--eval-log-states",
        type=str2bool,
        default=default_args.LOG_STATES,
        help="Detailed logging of states with their predictions. (default: %(default)s)",
    )
    parser.add_argument(
        "-evl-sp",
        "--eval-save-preds",
        type=str2bool,
        default=default_args.SAVE_PREDS,
        help="Save heuristic prediction files for every state. (default: %(default)s)",
    )
    parser.add_argument(
        "-evl-plt",
        "--eval-save-plots",
        type=str2bool,
        default=default_args.SAVE_PLOTS,
        help="Save plots related to the evaluation. (default: %(default)s)",
    )

    return parser.parse_args()


def get_sample_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "instance",
        type=str,
        help="Path to pddl or directory with instances to be used for sampling.",
    )
    parser.add_argument(
        "method",
        choices=["ferber", "yaaig"],
        default=default_args.SAMPLE_METHOD,
        help="Sampling base method to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-tst-dir",
        "--test-tasks-dir",
        type=str,
        default=default_args.SAMPLE_TEST_TASKS_DIR,
        help="Path to the directory where the test instances are located. Only used if `bound=max_task_hstar`. (default: %(default)s)",
    )
    parser.add_argument(
        "-stp",
        "--statespace",
        type=str,
        default=default_args.SAMPLE_STATESPACE,
        help="Path to the full statespace sampling file. (default: %(default)s)",
    )
    parser.add_argument(
        "-tech",
        "--technique",
        choices=["rw", "dfs", "bfs", "dfs_rw", "bfs_rw", "countBoth", "countAdds", "countDels"],
        default=default_args.SAMPLE_TECHNIQUE,
        help="Sample technique to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-stech",
        "--subtechnique",
        choices=["round_robin", "random_leaf", "percentage"],
        default=default_args.SAMPLE_SUBTECHNIQUE,
        help="Subtechique to use in dfs_rw or bfs_rw. (default: %(default)s)",
    )
    parser.add_argument(
        "-search",
        "--search-algorithm",
        choices=["greedy", "astar"],
        default=default_args.SAMPLE_SEARCH_ALGORITHM,
        help="Search algorithm to use when sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "-heur",
        "--search-heuristic",
        choices=["ff", "lmcut"],
        default=default_args.SAMPLE_SEARCH_HEURISTIC,
        help="Search heuristic used in the search algorithm. (default: %(default)s)",
    )
    parser.add_argument(
        "-ftech",
        "--ferber-technique",
        choices=["forward", "backward"],
        default=default_args.SAMPLE_FERBER_TECHNIQUE,
        help="Forward or backward search (Ferber). (default: %(default)s)",
    )
    parser.add_argument(
        "-fst",
        "--ferber-select-state",
        choices=["random_state", "entire_plan", "init_state"],
        default=default_args.SAMPLE_FERBER_SELECT_STATE,
        help="Forward or backward search (Ferber). (default: %(default)s)",
    )
    parser.add_argument(
        "-fn",
        "--ferber-num-tasks",
        type=int,
        default=default_args.SAMPLE_FERBER_NUM_TASKS,
        help="Number of tasks to generate (Ferber). (default: %(default)s)",
    )
    parser.add_argument(
        "-fmin",
        "--ferber-min-walk-len",
        type=int,
        default=default_args.SAMPLE_FERBER_MIN_WALK_LENGTH,
        help="Minimum random walk length (Ferber). (default: %(default)s)",
    )
    parser.add_argument(
        "-fmax",
        "--ferber-max-walk-len",
        type=int,
        default=default_args.SAMPLE_FERBER_MAX_WALK_LENGTH,
        help="Maximum random walk length (Ferber). (default: %(default)s)",
    )
    parser.add_argument(
        "-st",
        "--state-representation",
        choices=["fs", "fs-nomutex", "ps", "us", "au", "uc", "vs"],  # full state, full state no mutex, partial state, undefined, assign undefined, undefined char, valid state
        default=default_args.SAMPLE_STATE_REPRESENTATION,
        help="Output state representation. (default: %(default)s)",
    )
    parser.add_argument(
        "-rst",
        "--random-sample-state-representation",
        choices=["fs", "fs-nomutex", "ps", "us"],  # full state, full state no mutex, partial state, undefined
        default=default_args.SAMPLE_RANDOM_SAMPLE_STATE_REPRESENTATION,
        help="Random sample state representation. (default: %(default)s)",
    )
    parser.add_argument(
        "-uss",
        "--us-assignments",
        type=int,
        default=default_args.SAMPLE_ASSIGNMENTS_US,
        help="Number of assignments done with undefined state. (default: %(default)s)",
    )
    parser.add_argument(
        "-max",
        "--max-samples",
        type=int,
        default=default_args.SAMPLE_MAX_SAMPLES,
        help="Number of max samples to generate. (default: %(default)s)",
    )
    parser.add_argument(
        "-scs",
        "--searches",
        type=int,
        default=default_args.SAMPLE_SEARCHES,
        help="Number of searches. (default: %(default)s)",
    )
    parser.add_argument(
        "-sscs",
        "--samples-per-search",
        type=int,
        default=default_args.SAMPLE_PER_SEARCH,
        help="Number of samples per search. (default: %(default)s)",
    )
    parser.add_argument(
        "-b",
        "--bound",
        type=str,
        default=default_args.SAMPLE_BOUND,
        help="How to bound each rollout. Choices=[default, facts, facts_per_avg_effects, max_task_hstar, digit] (default: %(default)s)",
    )
    parser.add_argument(
        "-bm",
        "--bound-multiplier",
        type=float,
        default=default_args.SAMPLE_BOUND_MULTIPLIER,
        help="Multiplies the bound of each rollout by the given value. (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=str,
        default=default_args.SAMPLE_SEED,
        help="Sample seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-dups",
        "--allow-dups",
        choices=["all", "interrollout", "none"],  # full state, full state no mutex, partial state, undefined, assign undefined
        default=default_args.SAMPLE_ALLOW_DUPLICATES,
        help="Allow duplicate samples. (default: %(default)s)",
    )
    parser.add_argument(
        "-ms",
        "--mult-seed",
        type=str,
        default=default_args.SAMPLE_MULT_SEEDS,
        help="Sample mult seeds (1..n). (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--random-percentage",
        type=int,
        default=default_args.SAMPLE_RANDOM_PERCENTAGE,
        help="Percentage of random samples. (default: %(default)s)",
    )
    parser.add_argument(
        "-ce",
        "--random-estimates",
        type=str,
        default=default_args.SAMPLE_RANDOM_ESTIMATES,
        help="Estimates of the random samples. (default: %(default)s)",
    )
    parser.add_argument(
        "-rhg",
        "--restart-h-when-goal-state",
        type=str2bool,
        default=default_args.SAMPLE_RESTART_H_WHEN_GOAL_STATE,
        help="Restart h value when goal state is sampled (only random walk). (default: %(default)s)",
    )
    parser.add_argument(
        "-sf",
        "--state-filtering",
        type=str,
        default=default_args.SAMPLE_STATE_FILTERING,
        help="Filtering of applicable operators (none, mutex, statespace). (default: %(default)s)",
    )
    parser.add_argument(
        "-bfsp",
        "--bfs-percentage",
        type=int,
        default=default_args.SAMPLE_BFS_PERCENTAGE,
        help="Percentage of samples per BFS when technique=bfs_rw. (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_args.SAMPLE_DIR,
        help="Directory where the samples will be saved. (default: %(default)s)",
    )
    parser.add_argument(
        "-sai",
        "--sample-improvement",
        choices=["none", "partial", "complete", "both"],
        default=default_args.SAMPLE_IMPROVEMENT,
        help="Sample h-value improvement (SAI) strategy to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-sorth",
        "--sort-h",
        type=str2bool,
        default=default_args.SAMPLE_SORT_H,
        help="Sort sampling by increasign h-values before performing SUI. (default: %(default)s)",
    )
    parser.add_argument(
        "-sui",
        "--successor-improvement-k",
        type=int,
        default=default_args.SAMPLE_SUI,
        help="Successor Improvement (SUI) lookahead. If 0, no SUI is performed. (default: %(default)s)",
    )
    parser.add_argument(
        "-suieps",
        "--sui-eps",
        type=int,
        default=default_args.SAMPLE_SUI_EPSILON,
        help="RMSE no-improvement threshold for SUI early stop. (default: %(default)s)",
    )
    parser.add_argument(
        "-suirule",
        "--sui-rule",
        choices=["vu_u", "v_vu"], # v -> v | u,  u -> u :::: v -> v,  u -> v | u
        default=default_args.SAMPLE_SUI_RULE,
        help="Rule used to check if a state s' is compatible with s'' during SUI. (default: %(default)s)",
    )
    parser.add_argument(
        "-kd",
        "--k-depth",
        type=int,
        default=default_args.SAMPLE_K_DEPTH,
        help="Depth `k` for DFS or BFS. (default: %(default)s)",
    )
    parser.add_argument(
        "-unit",
        "--unit-cost",
        type=str2bool,
        default=default_args.SAMPLE_UNIT_COST,
        help="Increments h by unit cost instead of operator cost. (default: %(default)s)",
    )
    parser.add_argument(
        "-cores",
        "--cores",
        type=int,
        default=default_args.SAMPLE_CORES,
        help="Cores to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-t",
        "--max-time",
        type=float,
        default=default_args.SAMPLE_MAX_TIME,
        help="Max time to consider when doing sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--mem-limit",
        type=int,
        default=default_args.SAMPLE_MEM_LIMIT_MB,
        help="Memory limit to consider when sampling. (default: %(default)s)",
    )
    parser.add_argument(
        "-eval",
        "--evaluator",
        type=str,
        default=default_args.SAMPLE_EVALUATOR,
        help="Evaluator to use to estimate the h-values. (default: sampling)",
    )

    return parser.parse_args()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
