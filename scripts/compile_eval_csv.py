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

l = []
l_avg = []

for result in argv[1:]:
    train_args = {}
    with open(f'{result}/train_args.json') as json_file:
        train_args = json.load(json_file)
    sample_used = train_args['samples']
    used_avi = False
    used_min = False
    percentage = "1%" 
    experiment = "NA"
    sampling_algorithm = "NA"
    bound = "NA"
    if "bfsrw" in sample_used:
        sampling_algorithm = "bfs_rw"
    if "_avi" in sample_used:
        used_avi = True
    if "min-both" in sample_used:
        used_min = True
    if "bnd-def" in sample_used:
        bound = "default"
    if "bnd-props" in sample_used:
        bound = "propositions"
    if "bnd-propseff" in sample_used:
        bound = "propositions-eff"
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
    if used_avi and used_avi and bound == "propositions-eff":
        experiment = "best"
    if "random-sample-pct" in result:
        experiment = "random_sample"
    if "best-pct" in sample_used:
        experiment = "best_pct"
    if "1pct" in sample_used:
        percentage = "1%"
    if "5pct" in sample_used:
        percentage = "5%"
    if "25pct" in sample_used:
        percentage = "25%"
    if "50pct" in sample_used:
        percentage = "50%"
    if "100pct" in sample_used:
        percentage = "100%"

    statespace_file = glob(f"{result}/statespace*")[0]
    with open(statespace_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr = []
            curr.append(row['domain'])
            curr.append(row['instance'])
            curr.append(sampling_algorithm)
            curr.append(experiment)
            curr.append(bound)
            curr.append(percentage)
            curr.append(row['sample_seed'])
            curr.append(row['network_seed'])
            curr.append(row['state'])
            curr.append(row['y'])
            curr.append(row['pred'])
            curr.append(row['rmse'])
            l.append(curr)

    eval_results_file = glob(f"{result}/eval_results.csv")[0]
    with open(eval_results_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr = []
            curr.append(row['domain'])
            curr.append(row['instance'])
            curr.append(sampling_algorithm)
            curr.append(experiment)
            curr.append(bound)
            curr.append(percentage)
            curr.append(row['num_samples'])
            curr.append(row['sample_seed'])
            curr.append(row['network_seed'])
            curr.append(row['misses'])
            curr.append(row['mean_rmse_loss'])
            curr.append(row['max_rmse_loss'])
            l_avg.append(curr)

print(f"domain,instance,sampling_algorithm,preprocessing_method,bound,sample_percentage,sample_seed,network_seed,state,hstar,hnn,rmse")
for e in l:
    print(f"{','.join(e)}")

print("###")
print(f"domain,instance,sampling_algorithm,preprocessing_method,bound,sample_percentage,num_samples_statespace,sample_seed,network_seed,misses,mean_rmse_loss,max_rmse_loss")
for e in l_avg:
    print(f"{','.join(e)}")
