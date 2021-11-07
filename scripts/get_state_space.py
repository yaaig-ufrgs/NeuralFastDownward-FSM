#! /usr/bin/env python3

"""
Comment lines 172 and 173 of ../src/search/search_egnines/eager_search.cc
Use: ./get_state_space.py ../tasks/IPC/blocks/probBLOCKS-*-0.pddl
"""

from sys import argv
from subprocess import check_output, CalledProcessError
from re import match
from resource import setrlimit, RLIMIT_AS
from math import ceil
import os

SUCCESS_CODE = 12
MEMORY_LIMIT_CODE = 22

FD = "../fast-downward.py"
MEMORY_LIMIT = 4*1024*1024*1024 # 4 GB
OUTPUT_FOLDER = "state_space"

def converter(line_state: str, length: int):
    decimals = line_state.split(" ")
    assert ceil(length / 64) == len(decimals)
    binary = ""
    for i in range(len(decimals)):
        b = str(bin(int(decimals[i])))[2:]
        zeros = 64 - len(b)
        if i == 0 and length % 64 > 0:
            zeros = length % 64 - len(b)
            assert zeros >= 0
        binary += ("0" * zeros) + b
    return binary

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
            if line[:2] == "@ ":
                f.write(line[2:] + "\n")
                lines_wrote += 1
    assert lines_wrote-1 == total_states

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    setrlimit(RLIMIT_AS, (MEMORY_LIMIT, MEMORY_LIMIT))

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
                "eager_greedy([blind_print()])"
            ])
            # We expect error 12, if not, something strange is happening
            print(f"success? (comment lines 172 and 173 of ../src/search/search_egnines/eager_search.cc)")
            save_output(instance_name, output)
        except CalledProcessError as e:
            if e.returncode == SUCCESS_CODE:
                path = f"{OUTPUT_FOLDER}/{instance_name}.state_space"
                print(f"ok (saved to {OUTPUT_FOLDER}/{instance_name}.state_space)")
                save_state_space(instance_name, e.output)
            elif e.returncode == MEMORY_LIMIT_CODE:
                print("memory limit")
            else:
                print(f"unknown error (check output)")
                save_output(instance_name, e.output)

        os.remove(f"{OUTPUT_FOLDER}/{instance_name}_output.sas")

