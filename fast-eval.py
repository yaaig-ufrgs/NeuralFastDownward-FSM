#!/usr/bin/env python3

import logging
from sys import argv
from os import path, makedirs
from datetime import datetime

from src.pytorch.log import setup_full_logging
import src.pytorch.fast_downward_api as fd_api

"""
Use as:
$ ./eval.py <model.pt> <domain> <tasks>
e.g. $ ./eval.py traced.pt ../../tasks/blocksworld_ipc/probBLOCKS-12-0/domain.pddl ../../tasks/blocksworld_ipc/probBLOCKS-12-0/p*.pddl
"""

_log = logging.getLogger(__name__)

MODEL_NAME = argv[1].split("/")[-2].partition("-pytorch")[0]
OUTPUT_EVAL_FOLDER = f"results/eval/{MODEL_NAME}-eval-pytorch-{datetime.now().isoformat().replace('-', '.').replace(':', '.')}"

if __name__ == "__main__":
    model_fname = argv[1]
    domain_pddl = argv[2]
    instances = argv[3:]

    if not path.exists(OUTPUT_EVAL_FOLDER):
        makedirs(OUTPUT_EVAL_FOLDER)

    setup_full_logging(OUTPUT_EVAL_FOLDER, log_level=logging.INFO)
    _log.info(f"Model: {model_fname}")
    _log.info(f"Instances: {instances}")

    success = 0
    for count, instance in enumerate(instances):
        _log.info(f"Solving instance {instance} [{count+1}/{len(instances)}]")
        cost = fd_api.solve_instance_with_fd_nh(domain_pddl, instance, model_fname)
        if (cost != None):
            success += int(cost)
            _log.info(f"Solved")
        else:
            _log.info(f"Unsolved")
            
    success_rate = 100 * success / len(instances)

    _log.info(f"Success: {success} of {len(instances)} ({success_rate}%)")
