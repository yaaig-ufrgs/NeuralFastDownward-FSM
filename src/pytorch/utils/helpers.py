"""
Simple auxiliary functions.
"""

import logging
import glob
import os
from random import Random, sample, random, randint
from json import dump, load
from datetime import datetime, timezone
from subprocess import check_output
from argparse import Namespace
from prettytable import PrettyTable

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
        f"Setting to default value ({default_args.MAX_EPOCHS})."
    )
    return default_args.MAX_EPOCHS


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
        f"Setting to default value ({default_args.MAX_EXPANSIONS})."
    )
    return default_args.MAX_EXPANSIONS


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
    if train_folder:
        with open(f"{train_folder}/train_args.json") as f:
            data = load(f)
            assert domain == data["domain"]
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
        facts_filename_format = [
            "tasks/ferber21/training_tasks/{domain}/{problem}_facts.txt",
            "tasks/experiments/{domain}/{problem}_facts.txt",
        ]
        for filename in facts_filename_format:
            candidate_filename = filename.format(domain=args.domain, problem=args.problem)
            if os.path.exists(candidate_filename):
                facts_file = candidate_filename
                break

    if (not defaults_file) and facts_file:
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


def create_fake_samples(domain: str, problem: str, n_samples: int) -> str:
    try:
        with open("reference/large_tasks.csv", "r") as f:
            pddl = None
            for line in [x.strip() for x in f.readlines()[1:]]:
                if line.startswith(f"{domain},{problem}"):
                    pddl = line.split(",")[3]
            if not pddl:
                return None

        search_command = "sampling_search_yaaig(eager_greedy([ff(transform=sampling_transform())], transform=sampling_transform()), "\
            "techniques=[gbackward_yaaig(searches=1, samples_per_search=-1, max_samples=1000, bound_multiplier=1.0, technique=rw, "\
            "subtechnique=percentage, bound=default, depth_k=99999, random_seed=0, restart_h_when_goal_state=true, mutex=true, "\
            "allow_duplicates=interrollout, unit_cost=false, max_time=1200.0, mem_limit_mb=2048)], state_representation=complete, "\
            "random_seed=0, minimization=none, avi_k=0, avi_epsilon=-1, avi_unit_cost=false, avi_rule=vu_u, sort_h=false, "\
            "mse_hstar_file=, mse_result_file={sample_file}, assignments_by_undefined_state=10, contrasting_samples=0, evaluator=blind())"

        unique_id = str(random())[2:]
        samples_file = f"fake_{domain}_{problem}_{n_samples}_s{unique_id}"
        cl = [
            "./fast-downward.py",
            "--sas-file", f"{unique_id}-output.sas",
            "--plan-file", f"{unique_id}_sas_plan",
            "--build", "release",
            pddl,
            "--search", search_command.format(sample_file=samples_file)
        ]
        output = check_output(cl).decode("utf-8")
        if "Total samples: " not in output:
            return None

        with open(samples_file) as f:
            samples = [x.strip() for x in f.readlines()]
        with open(samples_file, "w") as f:
            for i in range(int(n_samples)):
                f.write(samples[randint(0, len(samples)-1)]+"\n")
        return samples_file
    except Exception as e:
        print(e)
        return None
