"""
Management of file creation and deletion.
"""

import os
import glob
from argparse import Namespace
from json import dump, load
from subprocess import check_output


def save_json(filename: str, data: list):
    """
    Saves a file with the given filename and data as a JSON file.
    """
    with open(filename, "w") as f:
        dump(data, f, indent=4)


def save_git_diff(dirname: str):
    """
    Saves the current git diff to help verifying tests.
    """
    filename = f"{dirname}/git.diff"
    with open(filename, "w") as f:
        f.write(check_output(["git", "diff"]).decode("ascii").strip())


def create_train_directory(args: Namespace) -> str:
    """
    Creates training directory according to current configuration.
    """

    print(getattr(args, "additional_folder_name"))

    sep = "."
    dirname = f"{args.output_folder}/nfd_train{sep}{args.samples.split('/')[-1]}"
    if args.seed != -1:
        dirname += f"{sep}ns{args.seed}"

    # Additional folder name
    abbrev = {
        "patience" : "pat", "output-layer" : "o",
        "num-folds" : "f", "hidden-layers" : "hl",
        "hidden-units" : "hu", "batch-size" : "b",
        "learning-rate" : "lr", "max-epochs" : "e",
        "max-training-time" : "t", "activation" : "a",
        "weight-decay" : "w", "dropout-rate" : "d",
        "shuffle-seed" : "shs", "shuffle" : "s",
        "use-gpu" : "gpu", "bias" : "bi",
        "bias-output" : "biout", "normalize-output" : "no",
        "restart-no-conv" : "rst",
    }
    for a in args.additional_folder_name:
        value = getattr(args, a.replace("-", "_"))
        if isinstance(value, list):
            value = "-".join([str(v) for v in value])
        else:
            value = str(value)
        dirname += f"-{abbrev[a]}_{value}"

    if os.path.exists(dirname):
        i = 2
        while os.path.exists(f"{dirname}{sep}{i}"):
            i += 1
        dirname = dirname + f"{sep}{i}"
    os.makedirs(dirname)
    os.makedirs(f"{dirname}/models")
    if args.save_git_diff:
        save_git_diff(dirname)

    return dirname


def create_test_directory(args):
    """
    Creates testing directory according to current configuration.
    """
    sep = "."
    tests_folder = args.train_folder / "tests"
    if not os.path.exists(tests_folder):
        os.makedirs(tests_folder)
    dirname = f"{tests_folder}/nfd_test"
    if os.path.exists(dirname):
        i = 2
        while os.path.exists(f"{dirname}{sep}{i}"):
            i += 1
        dirname = dirname + f"{sep}{i}"
    os.makedirs(dirname)
    if args.save_git_diff:
        save_git_diff(dirname)
    return dirname


def remove_temporary_files(directory: str):
    """
    Removes `output.sas` and `defaults.txt` files.
    """
    def remove_file(file: str):
        if os.path.exists(file):
            os.remove(file)
    remove_file(f"{directory}/output.sas")
    remove_file(f"{directory}/defaults.txt")


def save_y_pred_csv(data: dict, csv_filename: str):
    """
    Saves the {state: (value, predicted_value)} set to a CSV file.
    """
    with open(csv_filename, "w") as f:
        f.write("state,y,pred\n")
        for key in data.keys():
            f.write("%s,%s,%s\n" % (key, data[key][0], data[key][1]))


def remove_csv_except_best(directory: str, fold_idx: int):
    """
    Removes the recorded CSVs of each fold except the best one (less error).
    """
    csv_files = glob.glob(directory + "/*.csv")
    for f in csv_files:
        f_split = f.split("_")
        idx = int(f_split[-1].split(".")[0])
        if idx != fold_idx:
            os.remove(f)

def create_defaults_file(pddl_file: str, facts_file: str, output_file: str = "defaults.txt") -> str:
    """
    Create defaults file for `pddl_file`.
    For all fact in facts_file, 1 if fact \in initial_state(pddl_file) else 0.
    """

    init = None
    with open(pddl_file, "r") as f:
        pddl_text = f.read().lower()
        init = pddl_text.split(":init")[1].split(":goal")[0]

    with open(facts_file, "r") as f:
        facts = f.read().strip().split(";")
    
    # Atom on(i, a) -> (on i a)
    modified_facts = []
    for fact in facts:
        f = fact.replace("Atom ", "")               # Atom on(i, a) -> on(i, a)
        f = f.replace(", ", ",").replace(",", " ")  # on(i, a) -> on(i a)
        f = f"({f.split('(')[0]} {f.split('(')[1]}" # on(i a) -> (on i a)
        f = f.replace(" )", ")") # facts without objects (handempty ) -> (handempty)
        modified_facts.append(f)

    defaults = []
    for fact in modified_facts:
        value = "1" if fact in init else "0"
        defaults.append(value)

    if not defaults:
        raise Exception("get_defaults: defaults is empty")

    with open(output_file, "w") as f:
        f.write(";".join(defaults) + "\n")
    return output_file
