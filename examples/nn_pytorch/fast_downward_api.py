import subprocess
import sys

FD = "../../fast-downward.py"

def solve_instances_with_fd(traced_model: str, domain_pddl: str, instances_pddl: list[str],
                            blind: bool = False) -> [int]:
    """
    Tries to solve a list of PDDL instances with the network. 
    Returns a list of tuples representing (instance_index,exit_code).
    """

    instance_exit_codes = []

    opts = (f'astar(nh(torch_sampling_network(path={traced_model},\n'
            f'blind={str(blind).lower()})))')

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
        instance_exit_codes.append(exit_code)
        
    return instance_exit_codes
