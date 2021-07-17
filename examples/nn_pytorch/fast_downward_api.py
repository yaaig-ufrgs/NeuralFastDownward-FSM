import subprocess
import sys

import re

FD = "../../fast-downward.py"

def parse_plan(filename):
    PLAN_INFO_REGEX = re.compile(r"; cost = (\d+) \((unit cost|general cost)\)\n")
    last_line = ""
    with open(filename) as sas_plan:
        for last_line in sas_plan:
            pass
    match = PLAN_INFO_REGEX.match(last_line)
    if match:
        return int(match.group(1)), match.group(2)
    else:
        return None, None


def solve_instances_with_fd(domain_pddl, instances_pddl, opts = "astar(lmcut())"):
    """
    Tries to solve a list of PDDL instances.
    Returns a list of costs (same order of instances_pddl).

    """
    instances_costs = []

    for ins in instances_pddl:
        """
        exit_code = 0   -->  success
        exit_code = 1   -->  search plan found and out of memory
        exit_code = 2   -->  seach plan found and out of time
        exit_code = 3   -->  seach plan found and out of time & memoru
        exit_code = 11  -->  search unsolvable
        exit_code = 12  -->  search unsolvable incomplete
        exit_code = 22  -->  search out of memory
        exit_code = 23  -->  search out of time
        exit_code = 24  -->  search out of memory & time

        """
        exit_code = subprocess.call([FD, domain_pddl, ins, "--search", opts])

        cost, problem_type = parse_plan("sas_plan")
        instances_costs.append(cost)
        
    return instances_costs


def solve_instance_with_fd(domain_pddl, instance_pddl, opts = "astar(lmcut())"):
    """
    Tries to solve a PDDL instance. Return the cost (or None if search fails).
    """
    exit_code = subprocess.call([FD, domain_pddl, instance_pddl, "--search", opts])
    cost, problem_type = parse_plan("sas_plan")
    return cost
