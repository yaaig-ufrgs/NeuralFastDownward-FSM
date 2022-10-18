#! /usr/bin/env python3

"""
Get the entire state space of a pddl using the sampling engine.
Usage: ./get_state_space.py ../tasks/experiments/blocks/probBLOCKS-7-0.pddl
"""

from sys import argv
from subprocess import check_output, CalledProcessError
from re import match
from resource import setrlimit, RLIMIT_AS
from math import ceil, prod
import numpy as np
import os

FD = "../fast-downward.py"
OUTPUT_FOLDER = "state_space"

def save_output(instance_name: str, output: bytes):
    with open(f"{OUTPUT_FOLDER}/{instance_name}.output", "w") as f:
        f.write(output.decode("utf-8"))

def save_state_space(instance_name: str, output: bytes):
    output = output.decode("utf-8")
    assert "Completely explored state space -- no solution!" in output
    total_states = int(match(r".*Evaluated (\d+) state\(s\).", output.replace("\n", " ")).group(1))
    lines = output.split("\n")
    lines_wrote = 0
    with open(f"{OUTPUT_FOLDER}/{instance_name}.state_space", "w") as f:
        for line in lines:
            if line[:2] == ";;":
                f.write(line[2:] + "\n")
                lines_wrote += 1
    assert lines_wrote-1 == total_states


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for instance_pddl in argv[1:]:
        domain_name, instance_name = instance_pddl.split("/")[-2:]
        if instance_name == "domain.pddl":
            continue
        instance_name = f"{domain_name}_{instance_name.split('.pddl')[0]}"

        print(f"{instance_pddl}... ", end="", flush=True)
        try:
            output = check_output([
                FD,
                "--sas-file",
                f"{OUTPUT_FOLDER}/{instance_name}_output.sas",
                "--plan-file",
                f"{OUTPUT_FOLDER}/{instance_name}_sas_plan",
                instance_pddl,
                "--search",
                "eager_greedy([blind_print()], endless=true)"
            ])
            assert False, "We expect error 12, if not, something strange is happening"
        except CalledProcessError as e:
            if e.returncode == 12:
                path = f"{OUTPUT_FOLDER}/{instance_name}.state_space"
                print(f"ok (saved to {OUTPUT_FOLDER}/{instance_name}.state_space)")
                save_state_space(instance_name, e.output)
            elif e.returncode == 22:
                print("memory limit")
            else:
                print(f"unknown error (check output)")
                save_output(instance_name, e.output)
        except MemoryError as e:
            print("MemoryError")

        os.remove(f"{OUTPUT_FOLDER}/{instance_name}_output.sas")
