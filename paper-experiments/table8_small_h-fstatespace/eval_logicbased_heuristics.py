#!/usr/bin/env python3

"""
e.g. python3 python3 eval_logicbased_heuristics.py "goalcount ff" "blocks grid npuzzle rovers scanalyzerunit transportunit visitall"
"""

from sys import argv
import os
from subprocess import check_output

fd_root = os.path.abspath(__file__).split("NeuralFastDownward")[0] + "NeuralFastDownward"
statespace_files = {
    "blocks": f"{fd_root}/tasks/experiments/statespaces/statespace_blocks_probBLOCKS-7-0_hstar",
    "grid": f"{fd_root}/tasks/experiments/statespaces/statespace_grid_grid_hstar",
    "npuzzle": f"{fd_root}/tasks/experiments/statespaces/statespace_npuzzle_prob-n3-1_hstar",
    "rovers": f"{fd_root}/tasks/experiments/statespaces/statespace_rovers_rovers_hstar",
    "scanalyzerunit": f"{fd_root}/tasks/experiments/statespaces/statespace_scanalyzerunit_scanalyzer_hstar",
    "transportunit": f"{fd_root}/tasks/experiments/statespaces/statespace_transportunit_transport_hstar",
    "visitall": f"{fd_root}/tasks/experiments/statespaces/statespace_visitall_p-1-4_hstar"
}

pddls = {
    "blocks": f"{fd_root}/tasks/experiments/blocks/probBLOCKS-7-0.pddl",
    "grid": f"{fd_root}/tasks/experiments/grid/grid.pddl",
    "npuzzle": f"{fd_root}/tasks/experiments/npuzzle/prob-n3-1.pddl",
    "rovers": f"{fd_root}/tasks/experiments/rovers/rovers.pddl",
    "scanalyzerunit": f"{fd_root}/tasks/experiments/scanalyzer/scanalyzer.pddl",
    "transportunit": f"{fd_root}/tasks/experiments/transport/transport.pddl",
    "visitall": f"{fd_root}/tasks/experiments/visitall/p-1-4.pddl"
}

heuristics = argv[1].split()
for h in heuristics:
    assert h in ["ff", "goalcount"]
domains = argv[2].split() if len(argv) > 2 else "all"

if domains == ["all"]:
    domains = list(statespace_files.keys())
for d in domains:
    assert d in statespace_files

def read_samples(sample_file):
    pairs = []
    with open(sample_file, "r") as f:
        for h, s in [l.strip().split(";") for l in f.readlines() if not l.startswith("#")]:
            pairs.append((int(h), s))
    return pairs

sas_file, plan_file = "output.sas", "sas_plan"
commandline = (
    'python3 {fd_root}/fast-downward.py --sas-file {sas_file} --plan-file {plan_file} --build release {problem_pddl} '
    '--translate-options --unit-cost --search-options --search sampling_search_yaaig(eager_greedy([ff(transform=sampling_transform())], '
    'transform=sampling_transform()), techniques=[gbackward_yaaig(searches=1, max_samples=1)], evaluator={evaluator}(), evaluate_file={evaluate_file})'
)
print("heuristic,domain,value")
for heuristic in heuristics:
    for domain in domains:
        cl = commandline.format(
            fd_root=fd_root,
            sas_file=sas_file,
            plan_file=plan_file,
            problem_pddl=pddls[domain],
            evaluator=heuristic,
            evaluate_file=statespace_files[domain]
        )
        output = check_output(cl.split(" ", 13)).decode("utf-8")
        assert "Solution found." in output

        heuristic_pairs = read_samples(plan_file)
        hstar_pairs = read_samples(statespace_files[domain])
        n = len(heuristic_pairs)
        assert n == len(hstar_pairs)
        err = 0
        for i in range(n):
            assert heuristic_pairs[i][1] == hstar_pairs[i][1]
            err += abs(heuristic_pairs[i][0] - hstar_pairs[i][0])
        value = round(err / n, 2)

        print(heuristic, domain, value, sep=",")

os.remove(sas_file)
os.remove(plan_file)
