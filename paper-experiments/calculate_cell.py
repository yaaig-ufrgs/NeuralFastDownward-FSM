#!/usr/bin/env python3

"""
Calculates the average of several seeds (i.e. cell from a paper table)

$ python3 calculate_cell.py expanded table2_small_algorithms-hstar/results/*/*
$ python3 calculate_cell.py hvalue table2_small_algorithms-hstar/samples/*/*
"""

from sys import argv
import os
import json

def get_avg_expanded(results):
    values = []
    for result in results:
        with open(os.path.join(result, "tests/nfd_test/test_results.json"), "r") as f:
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
    exp = {}
    for folder in argv[2:]:
        e = folder.rsplit("/", 1)[0]
        if e not in exp:
            exp[e] = []
        exp[e].append(folder)
    print("experiment,n,avg")
    for e in exp:
        for folder in exp[e]:
            if type == "expanded":
                avg = get_avg_expanded(exp[e])
            elif type == "hvalue":
                avg = get_avg_hvalue(exp[e])
            else:
                raise Exception("invalid type")
        print(e, len(exp[e]), avg, sep=",")
