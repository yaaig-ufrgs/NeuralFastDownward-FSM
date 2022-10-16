#!/usr/bin/env python3

"""
To generate the table data, run the json in hvalue/ and its corresponding in hstar/
e.g. at the root of NeuralFastDownward: python3 run.py paper-experiments/table7_small_h-improv-diff-hstar/*/with_sai_sui/blocks_1pct_bfsrw_factseff*.json)

Use as an argument only the samples of hvalue/
e.g. python3 calculate_diff.py hvalue/with_sai_sui/samples/blocks_1pct_bfsrw_factseff/*
"""

from sys import argv
import os

def read_samples(file):
    with open(file, "r") as f:
        return [s for s in f.readlines() if not s.startswith("#")]

table = {}
for sample_path in argv[1:]:
    assert "hvalue/" in sample_path
    hstar_sample_path = sample_path.replace("hvalue/", "hstar/").replace("/yaaig_", "_hstar/yaaig_")
    if not os.path.exists(hstar_sample_path):
        print(f"Not found hstar/ file for {sample_path}")
        continue

    experiment, samples_folder, folder, sample = sample_path.split("/")[-4:]
    assert samples_folder == "samples"
    domain = folder.split("_")[0]
    regression_limit = folder.split("_")[3]

    samples_hvalue = read_samples(sample_path)
    samples_hstar = read_samples(hstar_sample_path)
    n = len(samples_hvalue)
    assert n == len(samples_hstar)

    s = 0
    for i in range(n):
        h_hvalue, s_hvalue = samples_hvalue[i].split(";")
        h_hstar, s_hstar = samples_hstar[i].split(";")
        assert s_hvalue == s_hstar
        h_hvalue, h_hstar = int(h_hvalue), int(h_hstar)
        assert h_hvalue >= h_hstar
        s += h_hvalue - h_hstar

    if experiment not in table:
        table[experiment] = {}
    if regression_limit not in table[experiment]:
        table[experiment][regression_limit] = {}
    if domain not in table[experiment][regression_limit]:
        table[experiment][regression_limit][domain] = []
    table[experiment][regression_limit][domain].append(s/n)

print("experiment,regression_limit,domain,value")
for experiment in table:
    for regression_limit in table[experiment]:
        for domain in table[experiment][regression_limit]:
            values = table[experiment][regression_limit][domain]
            print(experiment, regression_limit, domain, round(sum(values)/len(values), 2), sep=",")
