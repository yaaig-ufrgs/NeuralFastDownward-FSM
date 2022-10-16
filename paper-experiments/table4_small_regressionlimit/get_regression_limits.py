#!/usr/bin/env python3

from sys import argv
from glob import glob
import os
from subprocess import check_output
from re import findall

fd_root = os.path.abspath(__file__).split("NeuralFastDownward")[0] + "NeuralFastDownward"

limits = {
    "blocks": [None, None, None],
    "grid": [None, None, None],
    "npuzzle": [None, None, None],
    "rovers": [None, None, None],
    "scanalyzerunit": [None, None, None],
    "transportunit": [None, None, None],
    "visitall": [None, None, None]
}

pddl = {
    "blocks": "tasks/experiments/blocks/probBLOCKS-7-0.pddl",
    "grid": "tasks/experiments/grid/grid.pddl",
    "npuzzle": "tasks/experiments/npuzzle/prob-n3-1.pddl",
    "rovers": "tasks/experiments/rovers/rovers.pddl",
    "scanalyzerunit": "tasks/experiments/scanalyzer/scanalyzer.pddl",
    "transportunit": "tasks/experiments/transport/transport.pddl",
    "visitall": "tasks/experiments/visitall/p-1-4.pddl"
}

# d*
statespace_files = glob(f"{fd_root}/tasks/experiments/statespaces/*_hstar")
for file in statespace_files:
    with open(file) as f:
        fm = -1
        for h in [int(x.strip().split(";")[0]) for x in f.readlines() if not x.startswith("#")]:
            fm = max(fm, h)
        domain = file.split("/")[-1].split("_")[1]
        assert domain in limits
        limits[domain][0] = str(fm)

# F and \bar F
sas_file, plan_file = "output.sas", "sas_plan"
commandline = (
    'python3 {fd_root}/fast-downward.py --sas-file {sas_file} --plan-file {plan_file} --build release {problem_pddl} --search '
    'sampling_search_yaaig(eager_greedy([ff(transform=sampling_transform())], transform=sampling_transform()), techniques=['
    'gbackward_yaaig(searches=1, samples_per_search=1, regression_depth={regression_limit})])'
)
for regression_limit in ["facts", "facts_per_avg_effects"]:
    for domain in limits:
        assert domain in pddl
        cl = commandline.format(
            fd_root=fd_root,
            sas_file=sas_file,
            plan_file=plan_file,
            problem_pddl=f"{fd_root}/{pddl[domain]}",
            regression_limit=regression_limit
        )
        output = check_output(cl.split(" ", 10)).decode("utf-8")
        assert "Regression depth value: " in output
        id = 1 if regression_limit == "facts" else 2
        limits[domain][id] = findall(".*Regression depth value: (\d+).*", output)[0]
os.remove(sas_file)
os.remove(plan_file)


print("domain,dstar,facts,factseff")
for domain in limits:
    print(",".join([domain]+limits[domain]))
