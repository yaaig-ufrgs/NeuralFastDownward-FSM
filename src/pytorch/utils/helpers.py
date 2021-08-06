"""
Simple auxiliary functions.
"""

import logging
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

def get_domain_from_dirname(dirname):
    # TODO
    return ""

def logging_train_config(args, dirname):
    _log.info(f"Training for domain {get_domain_from_dirname(dirname)}")
    _log.info(f"Configuration")
    _log.info(f" | Samples: {args.samples.name}")
    _log.info(f" | Output layer: {args.output_layer}")
    _log.info(f" | Num folds: {args.num_folds}")
    _log.info(f" | Hidden layers: {args.hidden_layers}")
    _log.info(f" | Batch size: {args.batch_size}")
    _log.info(f" | Learning rate: {args.learning_rate}")
    _log.info(f" | Max epochs: {args.max_epochs}")
    _log.info(f" | Max training time: {args.max_training_time}s")
    _log.info(f" | Activation: {args.activation}")
    _log.info(f" | Weight decay: {args.weight_decay}")
    _log.info(f" | Dropout rate: {args.dropout_rate}")
    _log.info(f" | Shuffle: {args.shuffle}")
    _log.info(f" | Output folder: {args.output_folder}")

def logging_test_config(args, dirname):
    _log.info(f"Testing for domain {get_domain_from_dirname(dirname)}")
    _log.info(f"Configuration")
    _log.info(f" | Model: {args.model}")
    _log.info(f" | Domain PDDL: {args.domain_pddl}")
    _log.info(f" | Problems PDDL: {args.problems_pddl}")
    _log.info(f" | Search algorithm: {args.search_algorithm}")
    _log.info(f" | Max search time: {args.max_search_time}s")
    _log.info(f" | Max search memory: {args.max_search_memory}") # TODO: unit
    _log.info(f" | Shuffle: {args.shuffle}")
    _log.info(f" | Output folder: {args.output_folder}")
