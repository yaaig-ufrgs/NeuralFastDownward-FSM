#!/usr/bin/env python3

import logging
import sys
from os import path

from src.pytorch.fast_downward_api import solve_instance_with_fd_nh
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    logging_test_config,
    create_test_directory,
    logging_test_statistics,
    get_fixed_max_expansions,
    remove_temporary_files,
    get_test_tasks_from_problem,
    get_defaults_and_facts_files,
    get_problem_by_sample_filename,
    get_models_from_train_folder,
    get_samples_folder_from_train_folder,
)
from src.pytorch.utils.default_args import (
    DEFAULT_MAX_EXPANSIONS,
    DEFAULT_MAX_SEARCH_TIME,
    DEFAULT_FORCED_MAX_SEARCH_TIME,
)
from src.pytorch.utils.parse_args import get_test_args

_log = logging.getLogger(__name__)


def test_main(args):
    args.domain, args.problem = get_problem_by_sample_filename(str(args.train_folder).split(".")[1])
    args.save_git_diff = True

    if len(args.train_folder_compare) > 0:
       domain_cmp, problem_cmp =  get_problem_by_sample_filename(str(args.train_folder_compare).split(".")[1])
       if domain_cmp != args.domain or problem_cmp != args.problem:
            _log.error("Invalid comparison folder. Must be in the same domain and instance.")
            return

    dirname = create_test_directory(args)
    setup_full_logging(dirname)

    if args.samples_dir == None:
        args.samples_dir = get_samples_folder_from_train_folder(args.train_folder)
    if args.max_expansions == -1:
        args.max_expansions = get_fixed_max_expansions(args)
    if args.max_expansions == DEFAULT_MAX_EXPANSIONS and args.max_search_time == DEFAULT_MAX_SEARCH_TIME:
        args.max_search_time = DEFAULT_FORCED_MAX_SEARCH_TIME
        _log.warning(f"Neither max expansions nor max search time have been defined. "
                     f"Setting maximum search time to {DEFAULT_FORCED_MAX_SEARCH_TIME}s.")
    if args.samples_dir[-1] != "/":
        args.samples_dir += "/"

    if args.heuristic == "nn":
        models = get_models_from_train_folder(args.train_folder, args.test_model)
        models_cmp = get_models_from_train_folder(args.train_folder_compare, args.test_model)
        if len(models) == 0 or (len(args.train_folder_compare) > 0 and len(models_cmp) == 0):
            _log.error("No models found for testing.")
            return
    else:
        models = [""]
        models_cmp = [""]

    sample_file = str(args.train_folder).split('/')[-1].split('.')[1]
    if args.facts_file == "" and args.defaults_file == "":
        args.facts_file, args.defaults_file = get_defaults_and_facts_files(args.samples_dir, sample_file)

    if args.problem_pddls == []:
        args.problem_pddls = get_test_tasks_from_problem(
            train_folder=args.train_folder,
            tasks_folder=args.auto_tasks_folder,
            n=args.auto_tasks_n,
            shuffle_seed=args.auto_tasks_seed
        )
        if args.problem_pddls == []:
            return

    cmd_line = " ".join(sys.argv[0:])
    logging_test_config(args, dirname, cmd_line)

    for i in range(len(models)):
        model_path = models[i]
        model_cmp_path = models_cmp[i] if len(models_cmp) > 0 else ""
        output = {}
        for j, problem_pddl in enumerate(args.problem_pddls):
            _log.info(
                f'Solving instance "{problem_pddl}" ({j+1}/{len(args.problem_pddls)})'
            )
            output[problem_pddl] = solve_instance_with_fd_nh(
                domain_pddl=args.domain_pddl,
                problem_pddl=problem_pddl,
                traced_model=model_path,
                traced_model_cmp=model_cmp_path,
                search_algorithm=args.search_algorithm,
                heuristic=args.heuristic,
                heuristic_multiplier=args.heuristic_multiplier,
                unary_threshold=args.unary_threshold,
                time_limit=args.max_search_time,
                memory_limit=args.max_search_memory,
                max_expansions=args.max_expansions,
                facts_file=args.facts_file,
                defaults_file=args.defaults_file,
                save_log_to=dirname,
            )
            _log.info(problem_pddl)
            for var in output[problem_pddl]:
                _log.info(f" | {var}: {output[problem_pddl][var]}")

        model_file = model_path.split("/")[-1]
        logging_test_statistics(args, dirname, model_file, output)
        _log.info(f"Test on model {model_file} complete!")

    remove_temporary_files(dirname)
    _log.info("Test complete!")


if __name__ == "__main__":
    test_main(get_test_args())
