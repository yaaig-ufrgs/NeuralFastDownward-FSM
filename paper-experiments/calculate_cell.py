#!/usr/bin/env python3

"""
Calculates the average of several seeds (i.e. cell from a paper table)

$ python3 calculate_cell.py expanded table2_small_algorithms-hstar/results/blocks_1pct_bfs_200_hstar/*
$ python3 calculate_cell.py hvalue table2_small_algorithms-hstar/samples/blocks_1pct_bfs_200_hstar/*
"""

from sys import argv
import os
import json

def get_avg_expanded(results):
    values = []
    for folder in folders:
        with open(os.path.join(folder, "tests/nfd_test/test_results.json"), "r") as f:
            test_results = json.load(f)
        model = list(test_results["statistics"].keys())[0]
        values.append(test_results["statistics"][model]["avg_expanded"])
    return sum(values) / len(values)

def get_avg_hvalue(samples):
    values = []
    for sample in samples:
        with open(sample, "r") as f:
            hvalues = [int(l.strip().split(";")[0]) for l in f.readlines() if not l.startswith("#")]
        values.append(sum(hvalues) / len(hvalues))
    return sum(values) / len(values)

if __name__ == "__main__":
    type = argv[1]
    folders = argv[2:]
    experiment = None
    for folder in folders:
        e = folder.rsplit("/", 1)[0]
        if not experiment:
            experiment = e
        assert experiment == e, "All results must be from the same experiment to compute their mean"

    if type == "expanded":
        avg = get_avg_expanded(folders)
    elif type == "hvalue":
        avg = get_avg_hvalue(folders)
    else:
        raise Exception("invalid type")
    print("experiment:", experiment)
    print("n:", len(folders))
    print("avg:", avg)
