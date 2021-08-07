#!/usr/bin/env python3

import logging

from src.pytorch.fast_downward_api import solve_instance_with_fd_nh
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    logging_test_config,
    create_test_directory,
    save_json,
    get_unary_threshold
)
from src.pytorch.utils.parse_args import get_test_args

_log = logging.getLogger(__name__)

def test_main(args):
    dirname = create_test_directory(args)
    setup_full_logging(dirname)
    logging_test_config(args, dirname)

    results = {
        "configuration" : {
            "search_algorithm" : args.search_algorithm,
            "max_search_time" : args.max_search_time,
            "max_search_memory" : args.max_search_memory
        },
        "results" : {},
        "stats" : {}
    }

    model_path = f"{args.model_folder}/traced_best.pt"
    for i, problem_pddl in enumerate(args.problem_pddls):
        _log.info(f"Solving instance \"{problem_pddl}\" ({i+1}/{len(args.problem_pddls)})")
        results["results"][problem_pddl] = solve_instance_with_fd_nh(
            domain_pddl=args.domain_pddl,
            problem_pddl=problem_pddl,
            traced_model=model_path,
            unary_threshold=get_unary_threshold(args.model_folder),
            time_limit=args.max_search_time,
            memory_limit=args.max_search_memory
        )
        _log.info(results["results"][problem_pddl])

    # Compute stats
    # TODO

    save_json(f"{dirname}/test_results.json", results)
    _log.info("Test complete!")

if __name__ == "__main__":
    test_main(get_test_args())
