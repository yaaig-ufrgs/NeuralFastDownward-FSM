#! /usr/bin/env python3

# Run from the root
# e.g. RSL/check_groundtruth_states_in_samples.py samples/*

from sys import argv
from os.path import exists
import state_validators
from simulator import Simulator

with_contrasting = True
pddl_base = "tasks/ferber21/test_states"

files = []
for arg in argv[1:]:
    if arg.split(".")[-1] == "txt":
        continue
    sample_file = arg
    domain = arg.split("_")[1]
    instance = "_".join(arg.split("_")[2:-3])
    instance_file = (f"{pddl_base}/{domain}/moderate/{instance}/p1.pddl", f"{pddl_base}/{domain}/hard/{instance}/p1.pddl")
    files.append((instance_file, sample_file))

for instance_files, sample_file in files:
    instance_file = None
    for f in instance_files:
        if exists(f):
            instance_file = f
            break
    if instance_file == None:
        print(f"{sample_file} PDDL file not found")
        continue

    domain_file = f"{'/'.join(instance_file.split('/')[:-1])}/domain.pddl"

    simulator = Simulator(domain_file, instance_file, None, 0)

    if "blocks" in simulator.domainFile:
        validator = state_validators.blocks_state_validator(simulator.lpvariables)
    elif "npuzzle" in simulator.domainFile:
        validator = state_validators.npuzzle_state_validator(simulator.problem.init)
    elif "scanalyzer" in simulator.domainFile:
        validator = state_validators.scanalyzer_state_validator()
    elif "transport" in simulator.domainFile:
        validator = state_validators.transport_state_validator(simulator.problem.init)
    elif "visitall" in simulator.domainFile:
        validator = state_validators.visitall_state_validator()
    else:
        raise NotImplementedError("State validator not implemented for this domain!")

    with open(f"{sample_file}_facts.txt",) as f:
        atoms = f.read().replace("\n", "").split(";")
    with open(sample_file,) as f:
        samples = [l if l[-1] != "\n" else l[:-1] for l in f.readlines() if l[0] != "#"]

    max_h = 0
    if with_contrasting:
        for sample in samples:
            h, _ = sample.split(";")
            max_h = max(int(h), max_h)

    valid, total = 0, 0
    for sample in samples:
        h, s = sample.split(";")
        if not with_contrasting and int(h) == max_h:
            continue
        atomsInFormula = set([atoms[i] for i, b in enumerate(s) if b == "1"])
        if validator.is_valid(atomsInFormula):
            valid += 1
        total += 1

    print(f"{sample_file.split('/')[-1]},{valid},{total},{round(100*valid/total, 4)}%")
