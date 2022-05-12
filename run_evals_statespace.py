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

THREADS = 10

count = -1
up = 0
wait_else = 0
up_else = 0
os.system(f"tsp -K")
os.system(f"tsp -S {THREADS}")

statespaces_dict = {'blocks': "../NeuralFastDownward-results/resnets/experiments/blocks/samples/statespace/statespace_blocks_probBLOCKS-7-0_hstar",
                    'npuzzle': "../NeuralFastDownward-results/resnets/experiments/npuzzle/samples/statespace/statespace_npuzzle_prob-n3-1_hstar",
                    'visitall': "../NeuralFastDownward-results/resnets/experiments/visitall-opt14-strips/samples/statespace/statespace_visitall-opt14-strips_p-1-4_hstar",
                    'grid': "../NeuralFastDownward-results/resnets/experiments/grid/samples/statespace/statespace_grid_grid_hstar",
                    'rovers': "../NeuralFastDownward-results/resnets/experiments/rovers/samples/statespace/statespace_rovers_rovers_hstar",
                    'transport': "../NeuralFastDownward-results/resnets/experiments/transport/samples/statespace/statespace_transport_transport_hstar",
                    'scanalyzer': "../NeuralFastDownward-results/resnets/experiments/scanalyzer/samples/statespace/statespace_scanalyzer_scanalyzer_hstar"
                    }

models = []
for i in range(1, len(argv)):
    models += glob(f"{argv[i]}/*/*_ss0.ns0/models/traced_0.pt")

for model in models:
    model_name = model.split('/')[-3]
    domain_name = model_name.split('_')[2]
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
