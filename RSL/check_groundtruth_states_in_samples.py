#! /usr/bin/env python3

# Run from the root
# e.g. ./RSL/check_groundtruth_states_in_samples.py tasks/ferber21selected/transport/p01.pddl samples-test/rsl_transport_p01_countBoth_100000_ss80970

from sys import argv
import state_validators
from simulator import Simulator

with_contrasting = True

instance_file = argv[1]
domain_file = f"{'/'.join(instance_file.split('/')[:-1])}/domain.pddl"
sample_file = argv[2]

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
for sample in samples:
    h, _ = sample.split(";")
    max_h = max(int(h), max_h)

valid, total = 0, 0
for sample in samples:
    h, s = sample.split(";")
    if not with_contrasting and int(h) == max_h:
        continue
    atomsInFormula = set()
    for i, b in enumerate(s):
        if b == "1":
            atomsInFormula.add(atoms[i])
    if validator.is_valid(atomsInFormula):
        valid += 1
    total += 1

print(f"is valid with groundtruth: {valid}/{total} ({round(valid/total, 2)}%)")
