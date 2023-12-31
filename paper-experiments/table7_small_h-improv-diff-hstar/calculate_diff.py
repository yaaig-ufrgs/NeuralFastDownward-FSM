#!/usr/bin/env python3

"""
To generate the table data, run the json in hvalue/ and its corresponding in hstar/
e.g. at the root of NeuralFastDownward: python3 run.py paper-experiments/table7_small_h-improv-diff-hstar/*/with_sai_sui/blocks_1pct_bfsrw_factseff*.json

Use as an argument only the samples of hvalue/
e.g. python3 calculate_diff.py hvalue/with_sai_sui/samples/blocks_1pct_bfsrw_factseff/*
"""

from sys import argv
import os

def read_samples(sample_file):
    pairs = []
    with open(sample_file, "r") as f:
        for h, s in [l.strip().split(";") for l in f.readlines() if not l.startswith("#")]:
            pairs.append((int(h), s))
    return pairs

table = {}
for sample_path in argv[1:]:
    assert "hvalue/" in sample_path
    hstar_sample_path = sample_path.replace("hvalue/", "hstar/").replace("/yaaig_", "-hstar/yaaig_")
    if not os.path.exists(hstar_sample_path):
        print(f"Not found hstar/ file for {sample_path}: {hstar_sample_path}")
        continue

    experiment, samples_folder, folder, sample = sample_path.split("/")[-4:]
    assert samples_folder == "samples"
    domain = folder.split("_")[0]
    regression_limit = folder.split("_")[3]

    hvalue_pairs = read_samples(sample_path)
    hstar_pairs = read_samples(hstar_sample_path)
    n = len(hvalue_pairs)
    assert n == len(hstar_pairs)
    err = 0.0
    for i in range(n):
        assert hvalue_pairs[i][1] == hstar_pairs[i][1]
        err += abs(hvalue_pairs[i][0] - hstar_pairs[i][0])
    err /= n

    if experiment not in table:
        table[experiment] = {}
    if regression_limit not in table[experiment]:
        table[experiment][regression_limit] = {}
    if domain not in table[experiment][regression_limit]:
        table[experiment][regression_limit][domain] = []
    table[experiment][regression_limit][domain].append(err)

print("experiment,regression_limit,domain,value")
for experiment in table:
    for regression_limit in table[experiment]:
        for domain in table[experiment][regression_limit]:
            values = table[experiment][regression_limit][domain]
            print(experiment, regression_limit, domain, round(sum(values)/len(values), 2), sep=",")
