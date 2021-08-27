#!/usr/bin/env python3

import logging
from os import path

from src.pytorch.fast_downward_api import solve_instance_with_fd_nh
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    logging_test_config,
    create_test_directory,
    logging_test_statistics,
)
from src.pytorch.utils.parse_args import get_test_args

_log = logging.getLogger(__name__)

def test_main(args):
    dirname = create_test_directory(args)
    setup_full_logging(dirname)
    logging_test_config(args, dirname)

    model_path = f"{args.model_folder}/traced_best.pt"
    assert path.exists(model_path)

    problems_output = {}
    for i, problem_pddl in enumerate(args.problem_pddls):
        _log.info(f"Solving instance \"{problem_pddl}\" ({i+1}/{len(args.problem_pddls)})")
        problems_output[problem_pddl] = solve_instance_with_fd_nh(
            domain_pddl=args.domain_pddl,
            problem_pddl=problem_pddl,
            traced_model=model_path,
            search_algorithm=args.search_algorithm,
            unary_threshold=args.unary_threshold,
            time_limit=args.max_search_time,
            memory_limit=args.max_search_memory
        )
        _log.info(f"{problems_output[problem_pddl]}")

    logging_test_statistics(args, dirname, problems_output)
    _log.info("Test complete!")

if __name__ == "__main__":
    test_main(get_test_args())
