#!/usr/bin/env python3

"""
For all sample states in the `sample_file`, get h* with a*+lmcut.
Usage: ./get_hstar_pddl.py [problem_pddls]
  e.g. ./get_hstar_pddl.py tasks/experiments/blocks7/probBLOCKS-7-0/p1.pddl
"""

from sys import argv
from subprocess import check_output, CalledProcessError
from re import findall
from os import remove

def get_hstar(path: str, problem_pddl_files: [str]) -> [int]:
    FD = ("." if "scripts" in path else "..") + "/fast-downward.py"
    domain_pddl_file = f"./{'/'.join(problem_pddl_files[0].split('/')[:-1])}/domain.pddl"

    hstars = []
    for i, problem in enumerate(problem_pddl_files):
        p_split = problem.split('/')
        if p_split[-1] == "domain.pddl":
            continue

        problem_name = p_split[-2] + "_" + p_split[-1].split('.')[0]

        try:
            # Run FD with astar+lmcut to get h*
            output = check_output([
                FD,
                "--sas-file",
                f"output_{problem_name}.sas",
                "--plan-file",
                f"sas_plan_{problem_name}",
                domain_pddl_file,
                problem,
                "--search",
                "astar(lmcut())"
            ])
        except CalledProcessError as e:
            if e.returncode == 12: # no solution
                output = e.output
            else:
                print(f"Error with {sample}: {e} (2)")
                continue

        try:
            output = output.decode("utf-8")
            if "Completely explored state space -- no solution!" in output:
                cost = -1
            else:
                re_cost = findall(r".*Plan length: (\d+) step\(s\)..*", output)
                assert len(re_cost) == 1
                cost = re_cost[0]
            hstars.append(int(cost))
        except Exception as e:
            print(f"Error with {sample}: {e} (3)")
            continue

        for f in [f"output_{problem_name}.sas", f"sas_plan_{problem_name}"]:
            remove(f)

    return hstars

if __name__ == "__main__":
    get_hstar(argv[0], argv[1:])
