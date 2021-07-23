#!/usr/bin/env python3

from sys import argv

import fast_downward_api as fd_api

"""
Use as:
$ ./test.py <training_domain> <model.pt> <domain> <tasks>
e.g. $ ./test.py blocksworld traced.pt ../../tasks/blocksworld_ipc/probBLOCKS-12-0/domain.pddl ../../tasks/blocksworld_ipc/probBLOCKS-12-0/p*.pddl
"""

domain = argv[1]
model_fname = argv[2]
domain_pddl = argv[3]
problems = argv[4:]

# TODO other domains
domain_max_value = 1
if domain == "blocksworld":
    domain_max_value = 327

success = 0
for problem in problems:
    cost = fd_api.solve_instance_with_fd_nh(domain_pddl, problem, model_fname)
    success += int(cost != None)
success_rate = 100 * success / len(problems)

print(f"Success: {success} of {len(problems)} ({success_rate}%)")
