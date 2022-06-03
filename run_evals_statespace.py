#!/usr/bin/env python3

"""
Evals models over their respective statespaces.
Usage:
$ ./run_evals_statespace.py results/bounds/results
"""

import os
import csv
import time
from sys import argv
from glob import glob

THREADS = 5

count = -1
up = 0
wait_else = 0
up_else = 0
os.system(f"tsp -K")
os.system(f"tsp -S {THREADS}")

statespaces_dict = {'blocks': "tasks/experiments/statespaces/statespace_blocks_probBLOCKS-7-0_hstar",
                    'npuzzle': "tasks/experiments/statespaces/statespace_npuzzle_prob-n3-1_hstar",
                    'visitall': "tasks/experiments/statespaces/statespace_visitall-opt14-strips_p-1-4_hstar",
                    'grid': "tasks/experiments/statespaces/statespace_grid_grid_hstar",
                    'rovers': "tasks/experiments/statespaces/statespace_rovers_rovers_hstar",
                    'transport': "tasks/experiments/statespaces/statespace_transport_transport_hstar",
                    'transportunit': "tasks/experiments/statespaces/statespace_transportunit_transport_hstar",
                    'scanalyzer': "tasks/experiments/statespaces/statespace_scanalyzer_scanalyzer_hstar",
                    'scanalyzerunit': "tasks/experiments/statespaces/statespace_scanalyzerunit_scanalyzer_hstar"
                    }

models = []
for i in range(1, len(argv)):
    models += glob(f"{argv[i]}/*/*_ss0.ns0/models/traced_0.pt")

for model in models:
    model_name = model.split('/')[-3]
    domain_name = model_name.split('_')[2]
    if "unit" in model:
        domain_name += "unit"
    if "visitall-opt14-strips" in domain_name:
        domain_name = "visitall"
    statespace = statespaces_dict[domain_name]
    if up < THREADS:
        tsp_eval = f"tsp taskset -c {up}"
    else:
        if up % THREADS == 0 or up_else % THREADS == 0:
            up_else = 0
        tsp_eval = f"tsp -D {wait_else} taskset -c {up_else} "
        wait_else += 1
        up_else += 1

    eval_cmd = f"{tsp_eval} ./eval.py {model} {statespace} -sp true -plt true"
    print(eval_cmd)
    os.system(eval_cmd)
    print("----------------------------------------")
    count += 1
    up += 1
