"""
Management of bulk logging.
"""

import logging
import os
from json import load
from statistics import median, mean, pstdev
from glob import glob
import src.pytorch.utils.default_args as default_args
from src.pytorch.utils.helpers import (
    get_hostname,
    get_datetime,
    get_git_commit,
)
from src.pytorch.utils.file_helpers import save_json
from argparse import Namespace

_log = logging.getLogger(__name__)


def logging_train_config(
    args: Namespace, dirname: str, cmd_line: str, json: bool = True
):
    """
    Saves the full training configuration parameters as a JSON file.
    """
    args_dic = {
        "hostname": get_hostname(),
        "date": get_datetime(),
        "commit": get_git_commit(),
        "save_git_diff": args.save_git_diff,
        "command_line": cmd_line,
        "domain": args.domain,
        "problem": args.problem,
        "samples": args.samples,
        "unique_samples": args.unique_samples,
        "unique_states": args.unique_states,
        "model": args.model,
        "post_train_eval": args.post_train_eval,
        "loss_function": args.loss_function,
        "save_best_epoch_model": args.save_best_epoch_model,
        "patience": args.patience,
        "output_layer": args.output_layer,
        "linear_output": args.linear_output,
        "num_folds": args.num_folds,
        "hidden_layers": args.hidden_layers,
        "hidden_units": args.hidden_units
        if len(args.hidden_units) > 1
        else (args.hidden_units[0] if len(args.hidden_units) == 1 else "scalable"),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs
        if args.max_epochs != default_args.MAX_EPOCHS
        else "inf",
        "max_training_time": f"{args.max_training_time}s",
        "activation": args.activation,
        "weight_decay": args.weight_decay,
        "dropout_rate": args.dropout_rate,
        "training_size": args.training_size,
        "sample_percentage": args.sample_percentage,
        "shuffle": args.shuffle,
        "shuffle_seed": args.shuffle_seed,
        "dataloader_num_workers": args.data_num_workers,
        "gpu": args.use_gpu,
        "bias": args.bias,
        "bias_output": args.bias_output,
        "normalize_output": args.normalize_output,
        "check_dead_once": args.check_dead_once,
        "seed_increment_when_born_dead": args.seed_increment_when_born_dead,
        "weights_method": args.weights_method,
        "seed": args.seed if args.seed != -1 else "random",
        "scatter_plot": args.scatter_plot if args.scatter_plot != -1 else None,
        "plot_n_epochs": args.plot_n_epochs if args.plot_n_epochs != -1 else None,
        "num_threads": args.num_threads if args.num_threads != -1 else None,
        "output_folder": str(args.output_folder),
        "additional_folder_name": args.additional_folder_name
        if args.additional_folder_name != ""
        else None,
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if json:
        save_json(f"{dirname}/train_args.json", args_dic)


def logging_test_config(
    args: Namespace, dirname: str, cmd_line: str, save_file: bool = True
):
    """
    Saves the full test configuration parameters as a JSON file.
    """
    args_dic = {
        "hostname": get_hostname(),
        "date": get_datetime(),
        "commit": get_git_commit(),
        "command": cmd_line,
        "train_folder": str(args.train_folder),
        "domain_pddl": args.domain_pddl,
        "problems_pddl": args.problem_pddls,
        "search_algorithm": args.search_algorithm,
        "heuristic": args.heuristic,
        "heuritic_multiplier": args.heuristic_multiplier,
        "max_search_time": f"{args.max_search_time}s",
        "max_search_memory": f"{args.max_search_memory} MB",
        "max_expansions": args.max_expansions,
        "unary_threshold": args.unary_threshold,
        "test_model": args.test_model,
        "facts_file": args.facts_file if args.facts_file != "" else None,
        "defaults_file": args.defaults_file if args.defaults_file != "" else None,
        "train_folder_compare": args.train_folder_compare
        if args.train_folder_compare != ""
        else None,
    }
    if args.heuristic == "nn":
        args_dic["test_model"] = args.test_model
    if len(args.problem_pddls) == 0 or (
        args.auto_tasks_folder in args.problem_pddls[0]
        and len(args.problem_pddls) <= args.auto_tasks_n
    ):
        args_dic["auto_tasks_n"] = args.auto_tasks_n
        args_dic["auto_tasks_folder"] = args.auto_tasks_folder
        args_dic["auto_tasks_seed"] = args.auto_tasks_seed

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    if save_file:
        save_json(f"{dirname}/test_args.json", args_dic)


def logging_eval_config(
    args: Namespace, dirname: str, cmd_line: str, save_file: bool = True
):
    """
    Saves the full eval configuration parameters as a JSON file.
    """

    args_dic = {
        "hostname": get_hostname(),
        "date": get_datetime(),
        "commit": get_git_commit(),
        "command": cmd_line,
        "trained_model": str(args.trained_model),
        "data": args.samples,
        "follow_training": args.follow_training,
        "seed:": args.seed,
        "shuffle_seed": args.shuffle_seed,
        "shuffle": args.shuffle,
        "training_size": args.training_size, 
        "unique_samples": args.unique_samples,
    }

    _log.info(f"Configuration")
    for a in args_dic:
        _log.info(f" | {a}: {args_dic[a]}")

    args_files = glob(f"{dirname}/eval_args*.json")
    args_name = (
        "eval_args.json"
        if len(args_files) == 0
        else "eval_args_" + str(len(args_files)) + ".json"
    )

    if save_file:
        save_json(f"{dirname}/{args_name}", args_dic)


def logging_test_statistics(
    args: Namespace,
    dirname: str,
    model: str,
    output: dict,
    decimal_places: int = 4,
    save_file: bool = True,
):
    """
    Saves the test results to a file.
    """
    test_results_filename = f"{dirname}/test_results.json"
    if os.path.exists(test_results_filename):
        with open(test_results_filename) as f:
            results = load(f)
    else:
        results = {
            "configuration": {
                "search_algorithm": args.search_algorithm,
                "heuristic": args.heuristic,
                "max_search_time": f"{args.max_search_time}s",
                "max_search_memory": f"{args.max_search_memory} MB",
                "max_expansions": str(args.max_expansions),
            },
            "results": {},
            "statistics": {},
        }

    results["results"][model] = output
    results["statistics"][model] = {}
    rlist = {}
    if len(args.problem_pddls) > 0:
        model_stats = {}
        stats = []
        for problem in results["results"][model]:
            for s in results["results"][model][problem]:
                if s not in stats:
                    stats.append(s)
        for x in stats:
            rlist[x] = [
                results["results"][model][p][x]
                for p in results["results"][model]
                if x in results["results"][model][p]
            ]

            if x == "search_state":
                rlist[x] = [
                    results["results"][model][p][x] for p in results["results"][model]
                ]
                model_stats["plans_found"] = rlist[x].count("success")
                model_stats["total_problems"] = len(rlist[x])
                model_stats["coverage"] = round(
                    model_stats["plans_found"]
                    / model_stats["total_problems"],
                    decimal_places,
                )
            else:
                float_stats = ["search_time", "expansion_rate", "total_time"]
                rlist[x] = [float(i) if x in float_stats else int(i) for i in rlist[x]]
                if x == "total_time":
                    model_stats["total_accumulated_time"] = round(sum(rlist[x]), decimal_places)
                else:
                    model_stats[f"avg_{x}"] = round(mean(rlist[x]), decimal_places)
                    if len(rlist[x]) > 1:
                        if x == "plan_length":
                            model_stats["max_plan_length"] = max(rlist[x])
                            model_stats["min_plan_length"] = min(rlist[x])
                        model_stats[f"mdn_{x}"] = round(median(rlist[x]), decimal_places)
                        model_stats[f"pstdev_{x}"] = round(pstdev(rlist[x]), decimal_places)

        results["statistics"][model] = model_stats

    _log.info(f"Testing statistics for model {model}")
    for x in results["statistics"][model]:
        _log.info(f" | {x}: {results['statistics'][model][x]}")

    if save_file:
        save_json(test_results_filename, results)
