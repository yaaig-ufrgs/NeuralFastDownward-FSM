#!/usr/bin/env python3

"""
For all sample states in the `sample_file`, get h* with a*+lmcut.

Usage: ./get_hstar_with_fd.py sample_file problem_pddl
  e.g. ./get_hstar_with_fd.py ../samples/yaaig_blocks_probBLOCKS-4-0_dfs_fs_500x200_ss1 ../tasks/IPC/blocks/probBLOCKS-4-0.pddl

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

samples_name = samples_file.split('/')[-1]
problem = f"problem_{samples_name}.pddl"
domain = f"domain_{samples_name}.pddl"

copyfile(domain_pddl_file, domain)

with open(problem_pddl_file,) as f:
    pddl = f.readlines()
with open(samples_file,) as f:
    samples = f.readlines()
atoms = samples[1][:-1]
samples = [sample[:-1] if sample[-1] == "\n" else sample for sample in samples if sample[0] != "#"]

total = len(samples)
hstar = {}
for i, sample in enumerate(samples):
    _, sample = sample.split(";")
    if sample in hstar:
        continue

    try:
        output = "(:init " + check_output(["./boolean2state.py", atoms, sample]).decode("utf-8")[:-1] + ")\n"
        for j in range(len(pddl)):
            if ":init" in pddl[j] or ":INIT" in pddl[j]:
                pddl[j] = output
                while ":goal" not in pddl[j+1] and ":GOAL" not in pddl[j+1]:
                    pddl.remove(pddl[j+1])
                break
    except CalledProcessError as e:
        print(f"Error with {sample}: {e} (1)")
        continue

    with open(problem, "w") as f:
        for line in pddl:
            f.write(line)

    try:
        # Run FD with astar+lmcut to get h*
        output = check_output([
            FD,
            "--sas-file",
            f"output_{samples_name}.sas",
            "--plan-file",
            f"sas_plan_{samples_name}",
            domain,
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
        hstar[sample] = cost
        print(f"[{i+1}/{total}] {sample}: {hstar[sample]}")
    except Exception as e:
        print(f"Error with {sample}: {e} (3)")
        continue

for file in [domain, problem, f"output_{samples_name}.sas", f"sas_plan_{samples_name}"]:
    remove(file)

with open(f"hstar_{samples_name}", "w") as f:
    f.write("sample,hstar\n")
    for h in hstar:
        f.write(f"{h},{hstar[h]}\n")
