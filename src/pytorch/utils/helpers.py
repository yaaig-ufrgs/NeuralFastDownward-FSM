"""
Simple auxiliary functions.
"""

import logging
from json import dump
from os import path, makedirs
from datetime import datetime

_log = logging.getLogger(__name__)

def to_unary(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]

def get_domain_from_samples_filename(samples):
    # TODO
    return ""

def get_problem_from_samples_filename(samples):
    return "_".join(samples.split("/")[-1].split("_")[-2:])

def get_datetime():
    return datetime.now().isoformat().replace('-', '.').replace(':', '.')

def create_train_directory(args, config_in_foldername = False):
    dirname = get_problem_from_samples_filename(args.samples.name)
    if config_in_foldername:
        dirname += f"_{args.activation}_{args.output_layer}_" + \
            f"hid{args.hidden_layers}_w{args.weight_decay}_d{args.dropout_rate}"

    dirname = args.output_folder/f"{dirname}_{get_datetime()}"
    if path.exists(dirname):
        raise RuntimeError(f"Directory {dirname} already exists")
    makedirs(dirname)
    makedirs(dirname/"models")

    return dirname

def create_test_directory(args):
    tests_folder = args.model_folder/"tests"
    if not path.exists(tests_folder):
        makedirs(tests_folder)
    dirname = tests_folder/f"test_{get_datetime()}"
    if path.exists(dirname):
        raise RuntimeError(f"Directory {dirname} already exists")
    makedirs(dirname)
    return dirname

def save_json(filename: str, data: list):
    with open(filename, "w") as f:
            dump(data, f, indent=4)

def logging_train_config(args, dirname, json=True):
    args_dic = {
        "Domain" : get_domain_from_samples_filename(args.samples.name),
        "Samples" : args.samples.name,
        "Output layer" : args.output_layer,
        "Num folds" : args.num_folds,
        "Hidden layers" : args.hidden_layers,
        "Batch size" : args.batch_size,
        "Learning rate" : args.learning_rate,
        "Max epochs" : args.max_epochs,
        "Max training time" : args.max_training_time,
        "Activation" : args.activation,
        "Weight decay" : args.weight_decay,
        "Dropout rate" : args.dropout_rate,
        "Shuffle" : args.shuffle,
        "Output folder" : str(args.output_folder),
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if json:
        save_json(f"{dirname}/train_args.json", args_dic)

def logging_test_config(args, dirname, json=True):
    args_dic = {
        "Domain" : get_domain_from_samples_filename(dirname),
        "Model" : str(args.model_folder),
        "Domain PDDL" : args.domain_pddl,
        "Problems PDDL" : args.problem_pddls,
        "Search algorithm" : args.search_algorithm,
        "Max search time" : f"{args.max_search_time}s",
        "Max search memory" : f"{args.max_search_memory} MB",
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if json:
        save_json(f"{dirname}/test_args.json", args_dic)
