#! /usr/bin/env python3

"""
Get the entire state space of a pddl using the sampling engine.

Comment lines 172 and 173 of ../src/search/search_egnines/eager_search.cc
Use: ./get_state_space.py ../tasks/IPC/blocks/probBLOCKS-*-0.pddl
"""

from sys import argv
from subprocess import check_output, CalledProcessError
from re import match
from resource import setrlimit, RLIMIT_AS
from math import ceil, prod
import numpy as np
import os

SUCCESS_CODE = 12
MEMORY_LIMIT_CODE = 22

FD = "../fast-downward.py"
MEMORY_LIMIT = 4*1024*1024*1024 # 4 GB
OUTPUT_FOLDER = "state_space"

############################

def binary_to_fdr_state(binary: str, ranges: [int]):
    # binary 01000 1 0 0 0 1 00010 00001 00010
    # ranges [5, 1, 1, 1, 1, 1, 5, 5]
    # fdr [1, 0, 1, 1, 1, 0, 3, 4, 3]
    fdr = [-1] * len(ranges)
    ini = 0
    for i, range in enumerate(ranges):
        range -= 1
        subbin = binary[ini:ini+range]
        if subbin.count("1") > 1:
            return None
        fdr[i] = subbin.index("1") if "1" in subbin else range
        ini += range
    return fdr

def add_state_to_valid_states(valid_states, fdr: [int]):
    if len(fdr) == 1:
        valid_states[fdr[0]] = True
    else:
        add_state_to_valid_states(valid_states[fdr[0]], fdr[1:])

def check_state_in_valid_states(valid_states, fdr: [int]):
    if fdr == None:
        return False
    if len(fdr) == 1:
        return valid_states[fdr[0]]
    return check_state_in_valid_states(valid_states[fdr[0]], fdr[1:])

def state_in_state_space(state_space_file: str, states: [str]):
    with open(state_space_file,) as f:
        lines = [l if l[-1] != "\n" else l[:-1] for l in f.readlines()]
    atoms = lines[0].split(";")

    # e.g. atoms = ['Atom A1', 'Atom A2', '', 'Atom B1', '', 'Atom C1', 'Atom C2', 'Atom C3']
    #      ranges = [3, 2, 4]
    ranges = []
    i, decr = 0, 0
    while i < len(atoms):
        i = atoms.index("", i) + 1 if "" in atoms[i:] else len(atoms) + 1
        ranges.append(i - 1 - decr + 1) # +1 because it needs value for <none of the options>
        decr = i
    num_atoms = ["Atom" in x for x in atoms].count(True)

    # create a n-dimensions array (n = total of fdr variables)
    valid_states = np.reshape([False]*prod(ranges), tuple(ranges)).tolist()

    # Set True for all states \in state_space_file
    # state [3, 2, 4] = valid_states[3][2][4] is True
    for line in lines[1:]:
        add_state_to_valid_states(
            valid_states,
            binary_to_fdr_state(
                converter(line, num_atoms),
                ranges
            )
        )
    
    # Dictionary where the key is the state and the value is if it is in state space
    states_in_state_space = {}
    for state in states:
        states_in_state_space[state] = check_state_in_valid_states(
            valid_states,
            binary_to_fdr_state(state, ranges)
        )

    return states_in_state_space

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

############################

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
        except MemoryError as e:
            print("MemoryError")

        os.remove(f"{OUTPUT_FOLDER}/{instance_name}_output.sas")
