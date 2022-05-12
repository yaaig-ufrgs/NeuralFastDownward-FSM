#!/usr/bin/env python3

"""
Evals models over their respective statespaces.
Usage:
$ ./compile_eval_csv.py ../results/*/results/*/*_ss0.ns0
"""

import os
import csv
import time
import json
from sys import argv
from glob import glob

num_samples_dict = {
    "transport": 637632,
    "visitall": 125483,
    "blocks": 65990,
    "grid": 452353,
    "npuzzle": 181440,
    "rovers": 565824,
    "scanalyzer": 46080,
}

l = []
l_avg = []

for result in argv[1:]:
    # COMMENT THIS FOR FUTURE RUNS
    if "unit" in result:
        continue
    train_args = {}
    with open(f"{result}/train_args.json") as json_file:
        train_args = json.load(json_file)
    sample_used = train_args["samples"]
    used_avi = False
    used_min = False
    percentage = 0.01
    num_samples = -1
    num_samples_statespace = -1
    experiment = "NA"
    sampling_algorithm = "NA"
    bound = "NA"
    if "bfsrw" in sample_used:
        sampling_algorithm = "bfs_rw"
    if "_rw" in sample_used or "baseline" in result or "tech-rw" in sample_used or "-rw-" in result:
        sampling_algorithm = "rw"
    if "_avi" in sample_used:
        used_avi = True
    if "_min-both" in sample_used:
        used_min = True
    if "_min-none" in sample_used:
        used_min = False
    if "bnd-def" in sample_used:
        bound = "default"
    if "bnd-props" in sample_used or "propositions" in result:
        bound = "propositions"
    if "bnd-propseff" in sample_used or "propositions-eff":
        bound = "propositions-eff"
    if "baseline" in result:
        bound = "default"
    if not used_avi:
        experiment = "no_avi"
    if not used_min:
        experiment = "no_min"
    if "mutex" in result:
        experiment = "no_mutex"
    if "valid-states" in result:
        experiment = "valid_states"
    if "hstar-value" in result:
        experiment = "hstar_value"
    if "bounds" in result:
        experiment = "bounds"
    if "baseline" in result:
        experiment = "baseline"
    if used_avi and used_min and bound == "propositions-eff":
        experiment = "best"
    if "random-sample-pct" in result:
        experiment = "random_sample"
    if "best-pct" in sample_used:
        experiment = "best_pct"
    if "1pct" in sample_used:
        percentage = 0.01
    if "5pct" in sample_used:
        percentage = 0.05
    if "25pct" in sample_used:
        percentage = 0.25
    if "50pct" in sample_used:
        percentage = 0.50
    if "100pct" in sample_used:
        percentage = 1.0

    statespace_file = glob(f"{result}/statespace*")
    if len(statespace_file) == 0:
        continue
    with open(statespace_file[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr = []
            curr.append(row["domain"])
            num_samples_statespace = num_samples_dict[row["domain"]]
            num_samples = round(int(num_samples_dict[row["domain"]]) * percentage)
            #curr.append(row["instance"])
            curr.append(sampling_algorithm)
            curr.append(experiment)
            curr.append(bound)
            curr.append(row["sample_seed"])
            curr.append(row["network_seed"])
            curr.append(str(num_samples))
            curr.append(str(num_samples_statespace))
            curr.append(row["state"])
            curr.append(row["y"])
            curr.append(row["pred"])
            curr.append(row["rmse"])
            l.append(curr)

    eval_results_file = glob(f"{result}/eval_results.csv")
    if len(eval_results_file) == 0:
        continue
    with open(eval_results_file[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr = []
            curr.append(row["domain"])
            #curr.append(row["instance"])
            curr.append(sampling_algorithm)
            curr.append(experiment)
            curr.append(bound)
            curr.append(row["sample_seed"])
            curr.append(row["network_seed"])
            curr.append(str(round(int(row["num_samples"]) * percentage)))
            curr.append(row["num_samples"])
            curr.append(row["misses"])
            curr.append(row["mean_rmse_loss"])
            curr.append(row["max_rmse_loss"])
            l_avg.append(curr)

print(
    f"domain,sampling_algorithm,preprocessing_method,bound,sample_seed,network_seed,num_samples,num_samples_statespace,state,hstar,hnn,rmse"
)
for e in l:
    print(f"{','.join(e)}")

print("###")
print(
    f"domain,sampling_algorithm,preprocessing_method,bound,sample_seed,network_seed,num_samples,num_samples_statespace,misses,mean_rmse_loss,max_rmse_loss"
)
for e in l_avg:
    print(f"{','.join(e)}")
