#!/usr/bin/env python3

import logging
from os import path

from src.pytorch.fast_downward_api import solve_instance_with_fd_nh
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    logging_test_config,
    create_test_directory,
    logging_test_statistics,
    get_fixed_max_expansions
)
from src.pytorch.utils.parse_args import get_test_args

_log = logging.getLogger(__name__)

def test_main(args):
    dirname = create_test_directory(args)
    setup_full_logging(dirname)

    if args.max_expansions == -1:
        args.max_expansions = get_fixed_max_expansions(dirname)

    logging_test_config(args, dirname)

    if args.heuristic == "nn":
        models = []
        models_folder = f"{args.train_folder}/models"
        if args.test_model == "best":
            best_fold_path = f"{models_folder}/traced_best_val_loss.pt"
            if path.exists(best_fold_path):
                models.append(best_fold_path)
            else:
                _log.error(f"Best val loss model does not exists!")
        elif args.test_model == "all":
            i = 0
            while path.exists(f"{models_folder}/traced_{i}.pt"):
                models.append(f"{models_folder}/traced_{i}.pt")
                i += 1
        if len(models) == 0:
            _log.error("No models found for testing.")
            return
    else:
        models = [""]

    for model_path in models:
        output = {}
        for i, problem_pddl in enumerate(args.problem_pddls):
            _log.info(f"Solving instance \"{problem_pddl}\" ({i+1}/{len(args.problem_pddls)})")
            output[problem_pddl] = solve_instance_with_fd_nh(
                domain_pddl=args.domain_pddl,
                problem_pddl=problem_pddl,
                traced_model=model_path,
                search_algorithm=args.search_algorithm,
                heuristic=args.heuristic,
                unary_threshold=args.unary_threshold,
                time_limit=args.max_search_time,
                memory_limit=args.max_search_memory,
                max_expansions=args.max_expansions,
                save_log_to=dirname
            )
            _log.info(f"{output[problem_pddl]}")

        model_file = model_path.split("/")[-1]
        logging_test_statistics(args, dirname, model_file, output)
        _log.info(f"Test on model {model_file} complete!")

    _log.info("Test complete!")

if __name__ == "__main__":
    test_main(get_test_args())
