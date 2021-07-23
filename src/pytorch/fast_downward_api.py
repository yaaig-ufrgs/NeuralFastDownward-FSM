import subprocess
import json

import re

FD = "./fast-downward.py"
SAS_PLAN_FILE = "sas_plan"
CACHE_PLAN_COST = "plan_cost.json"

def parse_plan():
    PLAN_INFO_REGEX = re.compile(r"; cost = (\d+) \((unit cost|general cost)\)\n")
    last_line = ""
    try:
        with open(SAS_PLAN_FILE) as sas_plan:
            for last_line in sas_plan:
                pass
    except:
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
        cost, _ = parse_plan()
        instances_costs.append(cost)
        
    return instances_costs


def solve_instance_with_fd(domain_pddl, instance_pddl, opts = "astar(lmcut())", time_limit = "999999s", memory_limit = "32G", force = False):
    """
    Tries to solve a PDDL instance. Return the cost (or None if search fails).
    """

    cost = get_cached_plan_cost(instance_pddl)
    if force or cost == None:
        exit_code = subprocess.call([FD, "--search-time-limit", time_limit, "--search-memory-limit", memory_limit, domain_pddl, instance_pddl, "--search", opts])
        # exit_code = subprocess.call([FD, domain_pddl, instance_pddl, "--search", opts])
        cost, _ = parse_plan()
        add_cached_plan_cost(instance_pddl, cost)

    return cost


def solve_instance_with_fd_nh(domain_pddl, instance_pddl, traced_model, blind = True, unary_threshold = 0.01, time_limit = "1800s", memory_limit = "3800M"):
    """
    Tries to solve a PDDL instance with the torch_sampling_network. Return the cost (or None if search fails).
    Default limits from paper (30 min, 3.8GB)
    """

    opts = (f'astar(nh(torch_sampling_network(path={traced_model},'
            f'blind={str(blind).lower()},unary_threshold={unary_threshold})))')
    return solve_instance_with_fd(domain_pddl, instance_pddl, opts, time_limit, memory_limit, force=True)


def get_cached_plan_cost(pddl):
    try:
        with open(CACHE_PLAN_COST, "r") as f:
            data = json.loads(f.read())
    except:
        data = {}
    if pddl in data:
        return data[pddl]
    return None


def add_cached_plan_cost(pddl, cost):
    try:
        with open(CACHE_PLAN_COST, "r") as f:
            data = json.loads(f.read())
        data[pddl] = cost
        with open(CACHE_PLAN_COST, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)
    except:
        # Create it if not exists
        data = {}
        with open(CACHE_PLAN_COST, "w") as f:
            data[pddl] = cost
            json.dump(data, f, indent=4, sort_keys=True)
