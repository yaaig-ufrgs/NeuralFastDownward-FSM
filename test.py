#!/usr/bin/env python3

import logging
from os import path

from src.pytorch.fast_downward_api import solve_instance_with_fd_nh
from src.pytorch.log import setup_full_logging
from src.pytorch.utils.helpers import (
    logging_test_config,
    create_test_directory,
    save_json,
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
    assert path.exists(model_path)

    for i, problem_pddl in enumerate(args.problem_pddls):
        _log.info(f"Solving instance \"{problem_pddl}\" ({i+1}/{len(args.problem_pddls)})")
        results["results"][problem_pddl] = solve_instance_with_fd_nh(
            domain_pddl=args.domain_pddl,
            problem_pddl=problem_pddl,
            traced_model=model_path,
            search_algorithm=args.search_algorithm,
            unary_threshold=args.unary_threshold,
            time_limit=args.max_search_time,
            memory_limit=args.max_search_memory
        )
        _log.info(f"Result: {results['results'][problem_pddl]}")

    # Compute test results statistics
    decimal_places = 6
    rlist = {}
    for x in results["results"][args.problem_pddls[0]]:
        rlist[x] = [results["results"][p][x] for p in results["results"] if x in results["results"][p]]
        if x == "search_state":
            rlist[x] = [results["results"][p][x] for p in results["results"]]
            results["stats"]["plans_found"] = rlist[x].count("success")
            results["stats"]["total_problems"] = len(rlist[x])
            results["stats"]["coverage"] = \
                round(results["stats"]["plans_found"] / results["stats"]["total_problems"], decimal_places)
        elif x == "plan_length":
            for i in range(len(rlist[x])):
                rlist[x][i] = int(rlist[x][i])
            results["stats"]["max_plan_length"] = max(rlist[x])
            results["stats"]["min_plan_length"] = min(rlist[x])
            results["stats"]["avg_plan_length"] = round(sum(rlist[x]) / len(rlist[x]), decimal_places)
        elif x == "total_time":
            for i in range(len(rlist[x])):
                rlist[x][i] = float(rlist[x][i])
            results["stats"]["total_accumulated_time"] = round(sum(rlist[x]), decimal_places)
        elif x == "search_time":
            for i in range(len(rlist[x])):
                rlist[x][i] = float(rlist[x][i])
            results["stats"]["avg_search_time"] = round(sum(rlist[x]) / len(rlist[x]), decimal_places)
        else:
            for i in range(len(rlist[x])):
                rlist[x][i] = int(rlist[x][i])
            results["stats"][f"avg_{x}"] = round(sum(rlist[x]) / len(rlist[x]), decimal_places)

    _log.info("Test complete!")
    for x in results["stats"]:
        _log.info(f" | {x}: {results['stats'][x]}")

    save_json(f"{dirname}/test_results.json", results)

if __name__ == "__main__":
    test_main(get_test_args())
