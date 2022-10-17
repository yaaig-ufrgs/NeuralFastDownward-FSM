#!/usr/bin/env python3

"""
To generate the table data, run the json
    e.g. at the root of NeuralFastDownward: python3 run.py paper-experiments/table8_small_h-fstatespace/blocks_1pct_bfsrw_factseff_random20.json

Run this script
    e.g. python3 calculate_eval.py results/transportunit_1pct_bfsrw_factseff_random20/nfd_train.*/statespace_*
"""

from sys import argv
import os

def read_samples(file):
    with open(file, "r") as f:
        return [s for s in f.readlines() if not s.startswith("#")]

table = {}
for file in argv[1:]:
    domain, hnn = file.split("/")[-3].split("_", 1)

    with open(file, "r") as f:
        sum_err = 0
        errs = [float(x.strip().split(",")[-1]) for x in f.readlines()[1:]]
        assert min(errs) >= 0.0
        if hnn not in table:
            table[hnn] = {}
        if domain not in table[hnn]:
            table[hnn][domain] = []
        table[hnn][domain].append(sum(errs)/len(errs))

print("hnn,domain,avg_err")
for hnn in table:
    for domain in table[hnn]:
        values = table[hnn][domain]
        print(hnn, domain, round(sum(values)/len(values), 2), sep=",")
