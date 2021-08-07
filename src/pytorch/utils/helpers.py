"""
Simple auxiliary functions.
"""

import logging
import json
from os import path, makedirs
from datetime import datetime

SAMPLE_INIT_STATE = 1
SAMPLE_RANDOM_STATE = 2
SAMPLE_ENTIRE_PLAN = 3

_log = logging.getLogger(__name__)

def to_unary(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]

def create_directory(args, config_in_foldername = False):
    problem = "_".join(args.samples.name.split("/")[-1].split("_")[-2:])

    dirname = problem
    if config_in_foldername:
        dirname += f"_{args.activation}_{args.output_layer}_" + \
            f"hid{args.hidden_layers}_w{args.weight_decay}_d{args.dropout_rate}"

    dirname = args.output_folder/f"{dirname}_{datetime.now().isoformat().replace('-', '.').replace(':', '.')}"

    if path.exists(dirname):
        raise RuntimeError(f"Directory {dirname} already exists")
    makedirs(dirname)
    makedirs(dirname/"models")

    return dirname

def get_domain_from_samples_filename(samples):
    # TODO
    return ""

def logging_train_config(args, dirname, save_json=True):
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

    if save_json:
        with open(f"{dirname}/train_args.json", "w") as f:
            json.dump(args_dic, f, indent=4)


def logging_test_config(args, dirname, save_json=True):
    args_dic = {
        "Domain" : get_domain_from_samples_filename(dirname),
        "Model" : args.model,
        "Domain PDDL" : args.domain_pddl,
        "Problems PDDL" : args.problems_pddl, # TODO: problems list []
        "Search algorithm" : args.search_algorithm,
        "Max search time" : args.max_search_time,
        "Max search memory" : args.max_search_memory, # TODO: unit
        "Shuffle" : args.shuffle,
        "Output folder" : args.output_folder,
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if save_json:
        with open(f"{dirname}/test_args.json", "w") as f:
            json.dump(args_dic, f, indent=4)
