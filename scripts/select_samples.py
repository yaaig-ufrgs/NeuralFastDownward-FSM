#!/usr/bin/env python3
"""
From the sampling output file, select samples according to a strategy.

Use: $ ./select_samples.py sas_plan  <init|random|entire>
e.g. $ ./select_samples.py ../sas_plan random
"""

from sys import argv
from random import randrange

if len(argv) != 3 or argv[2] not in ["init", "random", "entire"]:
    print("Use: ./select_samples.py sas_plan <init|random|entire>")
    exit()

sas_plan = argv[1]
strategy = argv[2]

with open(sas_plan,) as f:
    lines = f.readlines()

state_len = len(lines[2].split(";"))
plan, plans = [], []
for i in range(len(lines)):
    if lines[i][:5] == "# ---":
        plans.append(plan)
        plan = []
    elif lines[i][0] != "#":
        last_idx = 2*state_len + len(lines[i].split(";")[0])
        plan.append(lines[i][:last_idx])

with open(f"{sas_plan}_{strategy}", "w") as output:
    for plan in plans:
        if strategy == "init":
            output.write(plan[-1] + "\n")

        elif strategy == "random":
            output.write(plan[randrange(len(plan))] + "\n")

        elif strategy == "entire":
            for sample in plan:
                output.write(sample + "\n")
