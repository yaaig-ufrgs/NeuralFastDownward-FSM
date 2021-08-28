"""
Simple auxiliary functions.
"""

import logging
from json import dump, load
from os import path, makedirs
from datetime import datetime
from src.pytorch.utils.default_args import DEFAULT_RANDOM_SEED

_log = logging.getLogger(__name__)

def to_prefix(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i < n else 0 for i in range(max_value)]

def to_onehot(n: int, max_value: int) -> [int]:
    max_value += 1
    return [1 if i == n else 0 for i in range(max_value)]

def get_datetime():
    return datetime.now().isoformat().replace('-', '.').replace(':', '.')

def create_train_directory(args, config_in_foldername = False):
    dirname = args.samples.name.split("/")[-1]
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
    tests_folder = args.train_folder/"tests"
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
        "samples" : args.samples.name,
        "output_layer" : args.output_layer,
        "num_folds" : args.num_folds,
        "hidden_layers" : args.hidden_layers,
        "hidden_units": args.hidden_units if len(args.hidden_units) > 1
            else (args.hidden_units[0] if len(args.hidden_units) == 1
            else "scalable"),
        "batch_size" : args.batch_size,
        "learning_rate" : args.learning_rate,
        "max_epochs" : args.max_epochs,
        "max_training_time" : args.max_training_time,
        "activation" : args.activation,
        "weight_decay" : args.weight_decay,
        "dropout_rate" : args.dropout_rate,
        "shuffle" : args.shuffle,
        "seed" : args.seed if args.seed != DEFAULT_RANDOM_SEED
            else "random",
        "output_folder" : str(args.output_folder)
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if json:
        save_json(f"{dirname}/train_args.json", args_dic)

def logging_test_config(args, dirname, save_file=True):
    args_dic = {
        "train_folder" : str(args.train_folder),
        "domain_pddl" : args.domain_pddl,
        "problems_pddl" : args.problem_pddls,
        "search_algorithm" : args.search_algorithm,
        "max_search_time" : f"{args.max_search_time}s",
        "max_search_memory" : f"{args.max_search_memory} MB",
        "test_model" : args.test_model
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if save_file:
        save_json(f"{dirname}/test_args.json", args_dic)

def logging_test_statistics(args, dirname, model, output, decimal_places=4, save_file=True):
    test_results_filename = f"{dirname}/test_results.json"
    if path.exists(test_results_filename):
        with open(test_results_filename) as f:
            results = load(f)
    else:
        results = {
            "configuration" : {
                "search_algorithm" : args.search_algorithm,
                "max_search_time" : f"{args.max_search_time}s",
                "max_search_memory" : f"{args.max_search_memory} MB"
            },
            "results" : {},
            "statistics" : {}
        }

    results["results"][model] = output
    results["statistics"][model] = {}
    rlist = {}
    for x in results["results"][model][args.problem_pddls[0]]:
        rlist[x] = [results["results"][model][p][x] for p in results["results"][model] \
            if x in results["results"][model][p]]
        if x == "search_state":
            rlist[x] = [results["results"][model][p][x] for p in results["results"][model]]
            results["statistics"][model]["plans_found"] = rlist[x].count("success")
            results["statistics"][model]["total_problems"] = len(rlist[x])
            results["statistics"][model]["coverage"] = \
                round(
                    results["statistics"][model]["plans_found"] / results["statistics"][model]["total_problems"],
                    decimal_places
                )
        elif x == "plan_length":
            for i in range(len(rlist[x])):
                rlist[x][i] = int(rlist[x][i])
            results["statistics"][model]["max_plan_length"] = max(rlist[x])
            results["statistics"][model]["min_plan_length"] = min(rlist[x])
            results["statistics"][model]["avg_plan_length"] = round(
                sum(rlist[x]) / len(rlist[x]),
                decimal_places
            )
        elif x == "total_time":
            for i in range(len(rlist[x])):
                rlist[x][i] = float(rlist[x][i])
            results["statistics"][model]["total_accumulated_time"] = round(sum(rlist[x]), decimal_places)
        elif x == "search_time":
            for i in range(len(rlist[x])):
                rlist[x][i] = float(rlist[x][i])
            results["statistics"][model]["avg_search_time"] = round(
                sum(rlist[x]) / len(rlist[x]),
                decimal_places
            )
        else:
            for i in range(len(rlist[x])):
                rlist[x][i] = int(rlist[x][i])
            results["statistics"][model][f"avg_{x}"] = round(sum(rlist[x]) / len(rlist[x]), decimal_places)

    _log.info(f"Training statistics for model {model}")
    for x in results["statistics"][model]:
        _log.info(f" | {x}: {results['statistics'][model][x]}")

    if save_file:
        save_json(test_results_filename, results)
