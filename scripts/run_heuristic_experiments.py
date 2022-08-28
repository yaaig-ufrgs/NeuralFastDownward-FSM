#!/usr/bin/env python3

"""
RUN IN SAME FOLDER AS `test.py`
usage: ./scripts/run_heuristic_experiments.py search_algorithm heuristic pddl [pddl ...]
e.g.   ./scripts/run_heuristic_experiments.py eager_greedy goalcount tasks/experiments/blocks/probBLOCKS-7-0.pddl tasks/experiments/depot/depot.pddl
"""

from sys import argv
from os import system, path

_DEST_FOLDER = "results"

algorithm = argv[1]
assert algorithm in ["astar", "eager_greedy"], "invalid search algorithm"
heuristic = argv[2]
assert heuristic in ["add", "blind", "ff", "goalcount", "hmax", "lmcut", "hstar"], "invalid heuristic"

for pddl in argv[3:]:
	domain, problem = pddl.replace(".pddl", "").rsplit("/", 2)[1:]
	folder = path.join(_DEST_FOLDER, f"nfd_train.yaaig_{domain}_{problem}_{algorithm.replace('_', '-')}_{heuristic}_ss0.ns0")
	system(f"./test.py {folder} -a {algorithm} -heu {heuristic}")
