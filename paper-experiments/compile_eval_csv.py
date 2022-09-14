#!/usr/bin/env python3

"""
Compiles evaluations results.
Usage:
$ ./compile_eval_csv.py results/*/*
"""

import os
import csv
import time
import json
from sys import argv
from glob import glob
from statistics import mean, pstdev

num_samples_dict = {
    "transport": 637632,
    "visitall": 79931,
    "blocks": 65990,
    "grid": 452353,
    "npuzzle": 181440,
    "rovers": 565824,
    "scanalyzer": 46080,
    "scanalyzer_unitcost": 46080,
    "transport_unitcost": 637632,
}

all_csv = glob(f"{argv[1]}/*/*/eval_results.csv")
#f_all = open('statespace_hnn_all.csv', 'w')
#writer_all = csv.writer(f_all)
#writer_all.writerow("domain,sampling_algorithm,preprocessing_method,bound,sample_seed,network_seed,pecentage,num_samples,num_samples_statespace,state,hstar,hnn,error".split(','))

f_avg = open('statespace_hnn_avg.csv', 'w')
writer_avg = csv.writer(f_avg)
writer_avg.writerow("domain,sampling_algorithm,preprocessing_method,bound,percentage,num_samples,random_samples,num_samples_statespace,mean_misses,mean_loss,pstdev_loss,max_loss,mean_rmse,pstdev_rmse".split(','))

d = {}

for result in argv[1:]:
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
    random_sample_pct = 0
    if "bfs" in sample_used:
        sampling_algorithm = "bfs"
    if "dfs" in sample_used:
        sampling_algorithm = "dfs"
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
    if "bnd-def" in sample_used or "bnd-200" in sample_used:
        bound = "default"
    if "bnd-props" in sample_used or "propositions" in result:
        bound = "prop"
    if "bnd-propseff" in sample_used or "propositions-eff" in result:
        bound = "propeff"
    if "baseline" in result:
        bound = "default"
    if "diameter" in result:
        bound = "diameter"
    if not used_avi:
        experiment = "no_avi"
    if not used_min:
        experiment = "no_min"
    if "bounds" in result:
        experiment = "bounds"
    if "baseline" in result and not used_avi and not used_min:
        experiment = "no_min_no_avi"
    if used_avi and used_min and bound == "propositions-eff" and sampling_algorithm == "bfs_rw":
        experiment = "best"
    if "-nomutex" in result:
        experiment = "no_mutex"
    if "-vs" in result:
        experiment = "valid_states"
    if "random-sample-pct" in result:
        experiment = "random_sample"
    if "best-pct" in result:
        experiment = "best_pct"
    if "hstar-pct" in result:
        experiment = "hstar_value"
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
    if "normal" in sample_used:
        sample_folder = sample_used.split('/')[-2]
        random_sample_pct = (100 - int(sample_folder[-2:])) / 100

    statespace_file = glob(f"{result}/statespace*")
    if len(statespace_file) == 0:
        print("ERROR")
        exit(1)
    """
    with open(statespace_file[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr = []
            if "unit" in result:
                row["domain"] += "_unitcost"
            curr.append(row["domain"])
            num_samples_statespace = num_samples_dict[row["domain"]]
            num_samples = round(int(num_samples_dict[row["domain"]]) * percentage)
            #curr.append(row["instance"])
            curr.append(sampling_algorithm)
            curr.append(experiment)
            curr.append(bound)
            curr.append(row["sample_seed"])
            curr.append(row["network_seed"])
            curr.append(str(percentage))
            curr.append(str(num_samples))
            curr.append(str(num_samples_statespace))
            curr.append(row["state"])
            curr.append(row["y"])
            curr.append(row["pred"])
            curr.append(row["error"])
            #print(f"{','.join(curr)}")
            writer_all.writerow(curr)
    """

    eval_results_file = glob(f"{result}/eval_results.csv")
    if len(eval_results_file) == 0:
        print("ERROR")
        exit(1)
    with open(eval_results_file[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            curr = []
            if "unit" in result:
                row["domain"] += "_unitcost"
            curr.append(row["domain"])
            #curr.append(row["instance"])
            curr.append(sampling_algorithm)
            curr.append(experiment)
            curr.append(bound)
            curr.append(row["sample_seed"])
            curr.append(row["network_seed"])
            curr.append(str(percentage))
            curr.append(str(round(int(row["num_samples"]) * percentage)))
            curr.append(str(round((int(row["num_samples"]) * percentage) * random_sample_pct)))
            curr.append(row["num_samples"])
            curr.append(row["misses"])
            curr.append(row["mean_loss"])
            curr.append(row["max_loss"])
            key = f"{row['domain']}_{sampling_algorithm}_{experiment}_{bound}_{str(round(int(row['num_samples']) * percentage))}_rs{random_sample_pct}"
            if key not in d:
                d[key] = {'domain': None, 'sampling_algorithm': None, 'experiment': None, 'bound': None, 'percentage': None, 'num_samples': None, 'num_samples_pct': None, 'random_samples': None, 'misses': [], 'mean_loss': [], 'max_loss': [], 'mean_rmse': []}
            d[key]['domain'] = row["domain"]
            d[key]['sampling_algorithm'] = sampling_algorithm
            d[key]['experiment'] = experiment
            d[key]['bound'] = bound
            d[key]['percentage'] = str(percentage)
            d[key]['num_samples'] = row["num_samples"]
            d[key]['num_samples_pct'] = str(round(int(row["num_samples"]) * percentage))
            d[key]['random_samples'] = str(round((int(row["num_samples"]) * percentage) * random_sample_pct))
            d[key]['misses'].append(float(row["misses"]))
            d[key]['mean_loss'].append(float(row["mean_loss"]))
            d[key]['max_loss'].append(float(row["max_loss"]))
            d[key]['mean_rmse'].append(float(row["rmse"]))
            #print(f"{','.join(curr)}")
            #writer_avg.writerow(curr)

# "domain,sampling_algorithm,preprocessing_method,bound,percentage,num_samples,random_samples,num_samples_statespace,mean_misses,mean_loss,pstdev_loss,max_loss,mean_rmse,pstdev_rmse
for key in d:
    curr = []
    curr.append(d[key]['domain'])
    curr.append(d[key]['sampling_algorithm'])
    curr.append(d[key]['experiment'])
    curr.append(d[key]['bound'] )
    curr.append(d[key]['percentage'])
    curr.append(d[key]['num_samples_pct'])
    curr.append(d[key]['random_samples'])
    curr.append(d[key]['num_samples'])
    curr.append(round(mean(d[key]['misses']),2))
    curr.append(round(mean(d[key]['mean_loss']),2))
    curr.append(round(pstdev(d[key]['mean_loss']),2))
    curr.append(round(mean(d[key]['max_loss']),2))
    curr.append(round(mean(d[key]['mean_rmse']),2))
    curr.append(round(pstdev(d[key]['mean_rmse']),2))
    writer_avg.writerow(curr)

f_avg.close()
