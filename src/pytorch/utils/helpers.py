"""
Simple auxiliary functions.
"""

import logging
from json import dump
from os import path, makedirs
from datetime import datetime

_log = logging.getLogger(__name__)

def to_prefix(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]

def to_onehot(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i == n else 0 for i in range(max_value)]

def get_domain_from_samples_filename(samples):
    # TODO
    return ""

def get_problem_from_samples_filename(samples):
    # TODO
    return samples.split("/")[-1]

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
        "domain" : get_domain_from_samples_filename(args.samples.name),
        "samples" : args.samples.name,
        "output_layer" : args.output_layer,
        "num_folds" : args.num_folds,
        "hidden_layers" : args.hidden_layers,
        "hidden_units": args.hidden_units,
        "batch_size" : args.batch_size,
        "learning_rate" : args.learning_rate,
        "max_epochs" : args.max_epochs,
        "max_training_time" : args.max_training_time,
        "activation" : args.activation,
        "weight_decay" : args.weight_decay,
        "dropout_rate" : args.dropout_rate,
        "shuffle" : args.shuffle,
        "random_seed" : args.random_seed,
        "output_folder" : str(args.output_folder),
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if json:
        save_json(f"{dirname}/train_args.json", args_dic)

def logging_test_config(args, dirname, json=True):
    args_dic = {
        "domain" : get_domain_from_samples_filename(dirname),
        "model" : str(args.model_folder),
        "domain_pddl" : args.domain_pddl,
        "problems_pddl" : args.problem_pddls,
        "search_algorithm" : args.search_algorithm,
        "max_search_time" : f"{args.max_search_time}s",
        "max_search_memory" : f"{args.max_search_memory} MB",
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if json:
        save_json(f"{dirname}/test_args.json", args_dic)

