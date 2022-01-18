"""
Simple auxiliary functions.
"""

import logging
import glob
import os
from random import Random
from json import dump, load
from datetime import datetime, timezone
from subprocess import check_output
from src.pytorch.utils.default_args import (
    DEFAULT_AUTO_TASKS_FOLDER,
    DEFAULT_AUTO_TASKS_N,
    DEFAULT_AUTO_TASKS_SEED,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MAX_EXPANSIONS,
)
from argparse import Namespace

_log = logging.getLogger(__name__)


def to_prefix(n: int, max_value: int) -> [int]:
    """
    Convert value `n` to prefix encoding.
    """
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]


def to_onehot(n: int, max_value: int) -> [int]:
    """
    Convert value `n` to onehot encoding.
    """
    max_value += 1
    return [1 if i == n else 0 for i in range(max_value)]


def prefix_to_h(prefix: [float], threshold: float = 0.01) -> int:
    """
    Convert prefix encoding to a value, respecting the given threshold value.
    """
    last_h = len(prefix) - 1
    for i in range(len(prefix)):
        if prefix[i] < threshold:
            last_h = i - 1
            break
    return last_h


def get_datetime() -> str:
    return datetime.now(timezone.utc).strftime("%d %B %Y %H:%M:%S UTC")


def get_fixed_max_epochs(args, model="resnet_ferber21", time="1800") -> int:
    """
    If argument `-e` equals -1, returns a default number of training epochs
    for the given problem. This is used to avoid time-based training.
    """
    with open(f"reference/{model}.csv", "r") as f:
        lines = [l.replace("\n", "").split(",") for l in f.readlines()]
        header = lines[0]
        for line in lines[1:]:
            if (
                args.domain == line[header.index("domain")]
                and args.problem == line[header.index("problem")]
            ):
                return int(line[header.index(f"epochs_{time}s")])
    _log.warning(
        f"Fixed number of epochs not found. "
        f"Setting to default value ({DEFAULT_MAX_EPOCHS})."
    )
    return DEFAULT_MAX_EPOCHS


def get_fixed_max_expansions(
    args: Namespace, model="resnet_ferber21", time="600"
) -> int:
    """
    Gets reference values for max expansions until stop seeking for a solution
    when trying to solve a problem.
    """
    with open(f"reference/{model}.csv", "r") as f:
        lines = [l.replace("\n", "").split(",") for l in f.readlines()]
        header = lines[0]
        for line in lines[1:]:
            if (
                args.domain == line[header.index("domain")]
                and args.problem == line[header.index("problem")]
            ):
                return int(line[header.index(f"expansions_{time}s")])
    _log.warning(
        f"Fixed maximum expansions not found. "
        f"Setting to default value ({DEFAULT_MAX_EXPANSIONS})."
    )
    return DEFAULT_MAX_EXPANSIONS


def get_git_commit() -> str:
    return check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_hostname() -> str:
    return check_output(["cat", "/proc/sys/kernel/hostname"]).decode("ascii").strip()


def add_train_arg(dirname: str, key, value):
    """
    Adds/updates a key-value pair from the `train_args.json` file.
    """
    with open(f"{dirname}/train_args.json", "r") as f:
        data = load(f)
    data[key] = value
    with open(f"{dirname}/train_args.json", "w") as f:
        dump(data, f, indent=4)


def get_problem_by_sample_filename(sample_filename: str) -> str:
    return sample_filename.split("/")[-1].split("_")[1:3]


def get_test_tasks_from_problem(
    train_folder: str,
    tasks_folder: str = DEFAULT_AUTO_TASKS_FOLDER,
    n: int = DEFAULT_AUTO_TASKS_N,
    shuffle_seed: int = DEFAULT_AUTO_TASKS_SEED,
) -> [str]:
    """
    From the given training training problem, automatically return `n` random test instances
    from `tasks_folder`.
    """
    with open(
        f"{train_folder}/train_args.json",
    ) as f:
        data = load(f)

    domain = data["domain"]
    problem = data["problem"]
    possible_parent_dirs = ["", domain, f"{domain}/moderate", f"{domain}/hard"]
    dir = None

    for parent_dir in possible_parent_dirs:
        candidate_dir = f"{tasks_folder}/{parent_dir}/{problem}"
        if os.path.isdir(candidate_dir):
            dir = candidate_dir

    if dir == None:
        _log.error(
            f"No tasks were automatically found from {tasks_folder}. "
            "Enter tasks manually from the command line or enter the path to the tasks folder (-atf)."
        )
        return []

    pddls = [
        f"{dir}/{f}"
        for f in os.listdir(dir)
        if f[-5:] == ".pddl" and f != "domain.pddl"
    ]

    if shuffle_seed != -1:
        Random(shuffle_seed).shuffle(pddls)

    if len(pddls) < n:
        _log.warning(f"Not found {n} tasks in {dir}. {len(pddls)} were selected.")
        return pddls

    return pddls[:n]


def get_defaults_and_facts_files(
    train_folder: str,
    problem_pddl: str,
    facts_filename_format: str = "tasks/ferber21/training_tasks/{domain}/{problem}_facts.txt",
) -> (str, str):
    """
    From the given samples directory and sample file, return its `facts` and `defaults` files.

    * facts must be in the training pddls folder
    * defaults must be in the testing pddls folder
    """
    with open(f"{train_folder}/train_args.json", "r") as f:
        train_args = load(f)
    
    # facts
    if "domain" in train_args and "problem" in train_args:
        facts_file = facts_filename_format.format(
            domain=train_args["domain"], problem=train_args["problem"]
        )
        if not os.path.exists(facts_file):
            facts_file = ""
    else:
        facts_file = ""
    
    # defaults
    defaults_file = problem_pddl.replace(".pddl", "_defaults.txt")
    if not os.path.exists(defaults_file):
        defaults_file = ""

    if not facts_file:
        _log.warning("No `facts` file found for the given sample.")
    if not defaults_file:
        _log.warning("No `default` file found for the given sample.")
    return facts_file, defaults_file


def get_models_from_train_folder(train_folder: str, test_model: str) -> [str]:
    """
    Returns the required trained network models to be used for testing, according to the
    `test_model` chosen.
    """
    models = []

    if train_folder == "":
        return models

    models_folder = f"{train_folder}/models"

    if test_model == "best":
        best_fold_path = f"{models_folder}/traced_best_val_loss.pt"
        if os.path.exists(best_fold_path):
            models.append(best_fold_path)
        else:
            _log.error(f"Best val loss model does not exists!")
    elif test_model == "all":
        i = 0
        while os.path.exists(f"{models_folder}/traced_{i}.pt"):
            models.append(f"{models_folder}/traced_{i}.pt")
            i += 1
    elif test_model == "epochs":
        i = 0
        while os.path.exists(f"{models_folder}/traced_0-epoch-{i}.pt"):
            models.append(f"{models_folder}/traced_0-epoch-{i}.pt")
            i += 1

    return models


def get_samples_folder_from_train_folder(train_folder: str) -> [str]:
    try:
        with open(f"{train_folder}/train_args.json", "r") as f:
            l = load(f)["samples"].split("/")
            return l[-2] if len(l) > 1 else l[0]
    except:
        return "samples"  # default
