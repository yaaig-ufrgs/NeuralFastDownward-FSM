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
        "-biout",
        "--bias-output",
        type=str2bool,
        default=default_args.BIAS,
        help="Use bias or not in the output layer. (default: %(default)s)",
    )
    parser.add_argument(
        "-clp",
        "--clamping",
        type=int,
        default=default_args.CLAMPING,
        help="Value to clamp heuristics with h=value-cl. (default: %(default)s)",
    )
    parser.add_argument(
        "-rmg",
        "--remove-goals",
        type=str2bool,
        default=default_args.REMOVE_GOALS,
        help="Remove goals from the sampling data (h = 0). (default: %(default)s)",
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
        help="Inicialization of network weights. (default: %(default)s)",
    )
    parser.add_argument(
        "-cdir",
        "--compare-csv-dir",
        type=str,
        default=default_args.COMPARED_HEURISTIC_CSV_DIR,
        help="Directory with CSV data to compare h^nn against; used for plotting. (default: %(default)s)",
    )
    parser.add_argument(
        "-hdir",
        "--hstar-csv-dir",
        type=str,
        default=default_args.COMPARED_HEURISTIC_CSV_DIR,
        help="Directory with h* CSV data; used for box plot. (default: %(default)s)",
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
        "-sibd",
        "--seed-increment-when-born-dead",
        type=int,
        default=default_args.SEED_INCREMENT_WHEN_BORN_DEAD,
        help="Seed increment when the network needs to restart due to born dead. (default: %(default)s)",
    )
    parser.add_argument(
        "-trd",
        "--num-threads",
        type=int,
        default=default_args.NUM_THREADS,
        help="Number of threads used for intra operations on CPU (PyTorch). (default: %(default)s)",
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
        "-sfst",
        "--standard-first",
        type=str2bool,
        default=default_args.STANDARD_FIRST,
        help="Show firstly the default samples to the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-cfst",
        "--contrast-first",
        type=str2bool,
        default=default_args.CONTRAST_FIRST,
        help="Show firstly the contrasting samples to the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-itc",
        "--intercalate-samples",
        type=int,
        default=default_args.INTERCALATE_SAMPLES,
        help="Intercalate by n the sampling data with contrasting and standard data. (default: %(default)s)",
    )
    parser.add_argument(
        "-cut",
        "--cut-non-intercalated-samples",
        type=str2bool,
        default=default_args.CUT_NON_INTERCALATED_SAMPLES,
        help="Remove leftover samples from the data. (default: %(default)s)",
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
        choices=["nn", "add", "blind", "ff", "goalcount", "hmax", "lmcut"],
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

    return parser.parse_args()


def get_exp_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-exp-type",
        choices=[
            "single",
            "fixed_net_seed",
            "fixed_sample_seed",
            "change_all",
            "all",
            "combined",
        ],
        default=default_args.EXP_TYPE,
        help="Experiment type according to seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-fs",
        "--exp-fixed-seed",
        type=int,
        default=default_args.EXP_FIXED_SEED,
        help="Fixed seed for fixed seed experiments. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-ns",
        "--exp-net-seed",
        type=int,
        default=default_args.EXP_NET_SEED,
        help="Network seed for fixed network experiments. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-ss",
        "--exp-sample-seed",
        type=int,
        default=default_args.EXP_SAMPLE_SEED,
        help="Sample seed for fixed sample seed experiments. (default: %(default)s)",
    )
    parser.add_argument(
        "-exp-threads",
        type=int,
        default=default_args.EXP_THREADS,
        help="Number of threads to use. (default: %(default)s)",
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
        "-trn-of",
        "--train-output-folder",
        type=Path,
        default=default_args.OUTPUT_FOLDER,
        help="Path where the training folder will be saved. (default: %(default)s)",
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
        "-trn-rmg",
        "--train-remove-goals",
        type=str2bool,
        default=default_args.REMOVE_GOALS,
        help="Remove goals from the sampling data (h = 0). (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-sfst",
        "--train-standard-first",
        type=str2bool,
        default=default_args.STANDARD_FIRST,
        help="Show firstly the default samples to the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-cfst",
        "--train-contrast-first",
        type=str2bool,
        default=default_args.CONTRAST_FIRST,
        help="Show firstly the contrasting samples to the network. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-itc",
        "--train-intercalate-samples",
        type=int,
        default=default_args.INTERCALATE_SAMPLES,
        help="Intercalate by n the sampling data with contrasting and standard data. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-cut",
        "--train-cut-non-intercalated-samples",
        type=str2bool,
        default=default_args.CUT_NON_INTERCALATED_SAMPLES,
        help="Remove leftover samples from the data. (default: %(default)s)",
    )
    parser.add_argument(
        "-trn-gpu",
        "--train-use-gpu",
        type=str2bool,
        default=default_args.USE_GPU,
        help="Use GPU during training. (default: %(default)s)",
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
        "--tst-model-dir",
        type=Path,
        help="Path to training folder with trained model. Only used if only testing.",
    )
    parser.add_argument(
        "-tst-a",
        "--test-search-algorithm",
        choices=["astar", "eager_greedy"],
        default=default_args.SEARCH_ALGORITHM,
        help="Algorithm to be used in the search. (default: %(default)s)",
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
        "-tst-dlog",
        "--test-downward-logs",
        type=str2bool,
        default=default_args.SAVE_DOWNWARD_LOGS,
        help="Save each instance's Fast-Downward log or not. (default: %(default)s)",
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


    return parser.parse_args()


def get_sample_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "instances_dir",
        type=str,
        help="Path to directory with instances to be used for sampling.",
    )
    parser.add_argument(
        "method",
        choices=["ferber", "yaaig", "rsl"],
        default=default_args.SAMPLE_METHOD,
        help="Sampling base method to use. (default: %(default)s)",
    )
    parser.add_argument(
        "-tech",
        "--technique",
        choices=["rw", "dfs", "countBoth", "countAdds", "countDels"],
        default=default_args.SAMPLE_TECHNIQUE,
        help="Sample technique to use. (default: %(default)s)",
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
        choices=["fs", "ps", "us"],  # full state, partial state and undefined
        default=default_args.SAMPLE_STATE_REPRESENTATION,
        help="Output state representation. (default: %(default)s)",
    )
    parser.add_argument(
        "-uss",
        "--us-assignments",
        type=int,
        default=default_args.SAMPLE_ASSIGNMENTS_US,
        help="Number of assignments done with undefined state. (default: %(default)s)",
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
        default=default_args.SAMPLES_PER_SEARCH,
        help="Number of samples per search. (default: %(default)s)",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=default_args.SAMPLE_SEED,
        help="Sample seed. (default: %(default)s)",
    )
    parser.add_argument(
        "-ms",
        "--mult-seed",
        type=int,
        default=default_args.SAMPLE_MULT_SEEDS,
        help="Sample mult seeds (1..n). (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--contrasting",
        type=int,
        default=default_args.SAMPLE_CONTRASTING,
        help="Percentage of contrasting samples. (default: %(default)s)",
    )
    parser.add_argument(
        "-rhg",
        "--restart_h_when_goal_state",
        type=str2bool,
        default=default_args.SAMPLE_RESTART_H_WHEN_GOAL_STATE,
        help="Restart h value when goal state is sampled (only random walk). (default: %(default)s)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=default_args.SAMPLE_DIR,
        help="Directory where the samples will be saved. (default: %(default)s)",
    )
    parser.add_argument(
        "-min",
        "--minimization",
        type=str2bool,
        default=default_args.SAMPLE_MINIMIZATION,
        help="Match exact samples with the min heuristic value between them. (default: %(default)s)",
    )
    parser.add_argument(
        "-avi",
        "--avi_k",
        type=int,
        default=default_args.SAMPLE_AVI,
        help="Approximate Value Iteration (AVI) lookahead. If 0, no AVI is performed. (default: %(default)s)",
    )
    parser.add_argument(
        "-rsl-states",
        "--rsl-num-states",
        type=int,
        default=default_args.SAMPLE_RSL_NUM_TRAIN_STATES,
        help="Number of samples. (default: %(default)s)",
    )
    parser.add_argument(
        "-rsl-demos",
        "--rsl-num-demos",
        type=int,
        default=default_args.SAMPLE_RSL_NUM_DEMOS,
        help="Number of demos. (default: %(default)s)",
    )
    parser.add_argument(
        "-rsl-len-demo",
        "--rsl-max-len-demo",
        type=int,
        default=default_args.SAMPLE_RSL_MAX_LEN_DEMO,
        help="Max len of each demo. (default: %(default)s)",
    )
    parser.add_argument(
        "-rsl-inv",
        "--rsl-check-invars",
        type=str2bool,
        default=default_args.SAMPLE_RSL_STATE_INVARS,
        help="Check for state invariants. (default: %(default)s)",
    )
    parser.add_argument(
        "-threads",
        type=int,
        default=default_args.SAMPLE_THREADS,
        help="Threads to use. (default: %(default)s)",
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
