"""
Simple auxiliary functions.
"""

import logging
import os
from random import Random, sample
from json import dump, load
from datetime import datetime, timezone
from subprocess import check_output
from argparse import Namespace
from prettytable import PrettyTable
from glob import glob
from natsort import natsorted

import src.pytorch.utils.default_args as default_args
from src.pytorch.utils.file_helpers import (
    create_defaults_file,
)

_log = logging.getLogger(__name__)

def count_parameters(model):
    """
    Get trainable parameters of a network model.
    https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    _log.info(f"\n{table}")
    _log.info(f"Total Trainable Params: {total_params}")
    return total_params


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


def get_memory_usage_mb(peak: bool = False):
    """
    Get current memory in MB.

    Peak memory if `peak` is true, otherwise current memory
    """

    field = 'VmPeak:' if peak else 'VmSize:'
    with open('/proc/self/status') as f:
        memusage = f.read().split(field)[1].split('\n')[0][:-3]

    return round(int(memusage.strip()) / 1024)

def get_fixed_max_epochs(args, reference_file="reference/large_tasks.csv") -> int:
    """
    If argument `-e` equals -1, returns a default number of training epochs
    for the given problem. This is used to avoid time-based training.
    """
    with open(reference_file, "r") as f:
        lines = [l.replace("\n", "").split(",") for l in f.readlines()]
        header = lines[0]
        for line in lines[1:]:
            if (args.domain == line[header.index("domain")]
                    and args.problem == line[header.index("problem")]):
                return int(line[header.index(f"epochs")])
    _log.warning(
        f"Fixed number of epochs not found. "
        f"Setting to default value ({default_args.MAX_EPOCHS})."
    )
    return default_args.MAX_EPOCHS


def get_fixed_max_expansions(domain: str, problem: str, reference_file="reference/large_tests_expansions_nn_yaaig.csv") -> int:
    """
    Gets reference values for max expansions until stop seeking for a solution
    when trying to solve a problem.
    """
    map = {}
    with open(reference_file, "r") as f:
        for d, p, t in [x.strip().split(",", 2) for x in f.readlines()[1:]]:
            if d == domain and p == problem:
                for i, pX in enumerate(t.split(",")):
                    map[f"p{i+1}"] = pX
    if len(map) != 50:
        _log.error(f"Fixed maximum expansions not found.")
        exit()
    return map


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


def get_problem_by_sample_filename(sample_filename: str, train_folder: str = None) -> (str, str):
    domain, problem = sample_filename.split("/")[-1].split("_")[1:3]
    unitary = False
    for suffix in ["-unit", "unit"]:
        if domain.endswith(suffix):
            domain = domain[:-len(suffix)]
            unitary = True
    if train_folder:
        with open(f"{train_folder}/train_args.json") as f:
            data = load(f)
            assert domain == data["domain"] or (unitary and domain+"-unit" == data["domain"]) \
                or (unitary and domain+"-" == data["domain"]) or (unitary and domain.replace("-opt", "unit-opt") == data["domain"]) # hack, remove later
            assert problem == data["problem"]
    return domain, problem


def get_test_tasks_from_problem(
    domain: str,
    problem: str,
    tasks_folder: str = default_args.AUTO_TASKS_FOLDER,
    n: int = default_args.AUTO_TASKS_N,
    shuffle_seed: int = default_args.AUTO_TASKS_SEED,
) -> [str]:
    """
    From the given training training problem, automatically return `n` random test instances
    from `tasks_folder`.
    """
    possible_parent_dirs = [
        f"experiments/{domain}",
        "ferber21/test_states",
        f"ferber21/test_states/{domain}",
        f"ferber21/test_states/{domain}/easy",
        f"ferber21/test_states/{domain}/moderate",
        f"ferber21/test_states/{domain}/hard",
    ]
    dir = None

    for parent_dir in possible_parent_dirs:
        candidate_dir = f"{tasks_folder}/{parent_dir}/{problem}"
        if os.path.isdir(candidate_dir):
            dir = candidate_dir
            break

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
        if n == default_args.AUTO_TASKS_N:
            _log.info(f"All {len(pddls)} tasks found were selected.")
        else:
            _log.warning(f"Not found {n} tasks in {dir}. {len(pddls)} were selected.")
        return pddls

    return pddls[:n]


def get_defaults_and_facts_files(
    args: Namespace,
    test_folder: str,
    problem_pddl: str
) -> (str, str):
    """
    Return its `facts` and `defaults` files.
    User input preferred. If not, it tries to find it automatically.

    Facts file is obtained from `facts_filename_format`.
    Defaults file is created (temporary file) based on `problem_pddl` and `facts_file`.
    """
    facts_file = args.facts_file
    if facts_file and not os.path.exists(facts_file):
        _log.warning("The `fact_file` arg doesn't exist. Getting it automatically...")
        facts_file = ""
    defaults_file = args.defaults_file
    if defaults_file and not os.path.exists(defaults_file):
        _log.warning(
            "The `defaults_file` arg doesn't exist. Getting it automatically..."
        )
        defaults_file = ""

    if not facts_file:
        candidate_filename = problem_pddl.replace(".pddl", "_facts.txt")
        if os.path.exists(candidate_filename):
            facts_file = candidate_filename
        else:
            facts_filename_format = [
                "tasks/ferber21/training_tasks/{domain}/{problem}_facts.txt",
                "tasks/experiments/{domain}/{problem}_facts.txt"
            ]
            for filename in facts_filename_format:
                candidate_filename = filename.format(domain=args.domain, problem=args.problem)
                if os.path.exists(candidate_filename):
                    facts_file = candidate_filename
                    break

    if not defaults_file and facts_file:
        defaults_file = create_defaults_file(problem_pddl, facts_file, test_folder)
        if not os.path.exists(defaults_file):
            defaults_file = ""

    if not facts_file:
        _log.error("No `facts` file found for the given sample.")
    if not defaults_file:
        _log.error("The `defaults` file could not be created.")
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
    elif test_model == "all":
        models = glob(f"{models_folder}/*.pt")

    if not models:
        _log.error(f"No trained models found!")
        exit(1)

    return natsorted(models)


def get_random_samples(lst: list, percent: float) -> list:
    """
    Gets unique random samples from `lst`.
    """
    if percent >= 1.0:
        return lst
    n = round(len(lst) * percent)
    return sample(lst, n)


def get_samples_folder_from_train_folder(train_folder: str) -> [str]:
    try:
        with open(f"{train_folder}/train_args.json", "r") as f:
            l = load(f)["samples"].split("/")
            return l[-2] if len(l) > 1 else l[0]
    except:
        return "samples"  # default


def get_train_args_json(train_folder: str) -> dict:
    with open(train_folder + "/train_args.json") as json_file:
        return load(json_file)
