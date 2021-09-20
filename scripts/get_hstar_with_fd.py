#!/usr/bin/env python3

# usage: ./get_hstar_with_fd.py sample problem_pddl
#   e.g. ./get_hstar_with_fd.py ../samples/fukunaga_blocks_probBLOCKS-4-0_dfs_fs_500x200_ss1 ../tasks/IPC/blocks/probBLOCKS-4-1.pddl

"""
Expected problem_pddl format:
(define (problem BLOCKS-4-1)
(:domain BLOCKS)
(:objects A C D B )
(:INIT (CLEAR B) (ONTABLE D) (ON B C) (ON C A) (ON A D) (HANDEMPTY))
(:goal (AND (ON D C) (ON C A) (ON A B)))
)

Excepted line 2 of the sample:
#<State>=Atom...
"""

from sys import argv
from subprocess import check_output, CalledProcessError
from shutil import copyfile
from re import findall
from os import remove

FD = ("." if "scripts" in argv[0] else "..") + "/fast-downward.py"

samples_file = argv[1]
problem_pddl_file = argv[2]
domain_pddl_file = f"{'/'.join(problem_pddl_file.split('/')[:-1])}/domain.pddl"
copyfile(domain_pddl_file, "domain.pddl")

with open(problem_pddl_file,) as f:
    pddl = f.readlines()
with open(samples_file,) as f:
    samples = f.readlines()

atoms = samples[1]
if atoms[-1] == "\n":
    atoms = atoms[:-1]

hstar = {}
for i, sample in enumerate(samples):
    if sample[0] == "#":
        continue
    if sample[-1] == "\n":
        sample = sample[:-1]
    _, sample = sample.split(";")
    if sample in hstar:
        continue

    try:
        pddl[-3] = "(:init " + check_output(["./boolean2state.py", atoms, sample]).decode("utf-8")[:-1] + ")\n"
    except CalledProcessError as e:
        print("Error with {sample}: {e}")

    with open("problem.pddl", "w") as f:
        for line in pddl:
            f.write(line)

    # Run FD with astar+lmcut to get h*
    output = check_output([FD, "problem.pddl", "--search", "astar(lmcut())"]).decode("utf-8")
    
    try:
        re_cost = findall(r".*Plan length: (\d+) step\(s\)..*", output)
        assert len(re_cost) == 1
        hstar[sample] = re_cost[0]
    except Exception as e:
        print("Error with {sample}: {e}")

for file in ["domain.pddl", "problem.pddl", "sas_plan"]:
    remove(file)

with open(f"hstar_{samples_file.split('/')[-1]}", "w") as f:
    f.write("sample,hstar\n")
    for h in hstar:
        f.write(f"{h},{hstar[h]}\n")
