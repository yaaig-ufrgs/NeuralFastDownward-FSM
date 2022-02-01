#!/usr/bin/env python3

from glob import glob
from json import load

"""
MOVE THIS SCRIPT TO THE RESULTS FOLDER!
"""

d = {}

print("domain,problem,ss,ns,shs,s_perc,samples,count_not_full_samples,count_not_used_samples,avg_diff_pred_in_full,avg_diff_pred_not_full,avg_diff_pred_in_used,avg_diff_pred_not_used")
#      domain,problem,ss,ns,shs,s_perc,samples,count_not_full_samples,count_not_used_samples,avg_diff_pred_in_full,avg_diff_pred_not_full,avg_diff_pred_in_used,avg_diff_pred_not_used")
for train_dir in glob("nfd_train.*"):
    #print(train_dir)
    log_files = glob(f"{train_dir}/tests/nfd_test/downward_logs/*.log")
    train_args_file = f"{train_dir}/train_args.json"
    train_args = {}

    num_samples_total = 0
    with open(train_args_file) as json_file:
        train_args = load(json_file)
    sample_file = f"../{train_args['samples']}"
    with open(sample_file, 'r') as f:
        num_samples_total = sum(1 for line in f)

    num_evals = 0

    count_not_full_samples = 0
    count_not_used_samples = 0
    count_not_in_both = 0

    count_avg_diff_pred_full_found = 0
    sum_diff_pred_full_found = 0
    count_avg_diff_pred_full_not_found = 0
    sum_diff_pred_full_not_found = 0

    count_avg_diff_pred_used_found = 0
    sum_diff_pred_used_found = 0
    count_avg_diff_pred_used_not_found = 0
    sum_diff_pred_used_not_found = 0

    with open(log_files[0], 'r') as f:
        for line in f:
            found_full = True
            line = line.strip()
            if "In full samples: true" in line:
                found_full = True
                count_avg_diff_pred_full_found += 1
                h_pred = int(next(f).strip().split(" ")[-1])
                h_full_sample = int(next(f).strip().split(" ")[-1])
                sum_diff_pred_full_found += abs(h_pred - h_full_sample)

            if "In full samples: false" in line:
                found_full = False
                count_avg_diff_pred_full_not_found += 1
                h_pred = int(next(f).strip().split(" ")[-1])
                h_full_sample = int(next(f).strip().split(" ")[-1])
                sum_diff_pred_full_not_found += abs(h_pred - h_full_sample)

            if "In used samples: true" in line:
                count_avg_diff_pred_used_found += 1
                h_pred = int(next(f).strip().split(" ")[-1])
                h_used_sample = int(next(f).strip().split(" ")[-1])
                sum_diff_pred_used_found += abs(h_pred - h_used_sample)

            if "In used samples: false" in line:
                h_pred = int(next(f).strip().split(" ")[-1])
                h_used_sample = int(next(f).strip().split(" ")[-1])
                if found_full:
                    count_avg_diff_pred_used_not_found += 1 # só conta se tiver achado no found, caso contrário não faz sentido
                    sum_diff_pred_used_not_found += abs(h_pred - h_full_sample)
                elif not found_full:
                    count_not_in_both += 1;

            if "Count not in full samples" in line:
                count_not_full_samples = int(line.split(" ")[-1])
            if "Count not in used samples" in line:
                count_not_used_samples = int(line.split(" ")[-1])

            if "Evaluated" in line:
                num_evals = line.split(" ")[-2]

    avg_diff_pred_full_found = "-" if count_avg_diff_pred_full_found == 0 else sum_diff_pred_full_found / count_avg_diff_pred_full_found
    avg_diff_pred_full_not_found = "-" if count_avg_diff_pred_full_not_found == 0 else sum_diff_pred_full_not_found / count_avg_diff_pred_full_not_found
    avg_diff_pred_used_found = "-" if count_avg_diff_pred_used_found == 0 else sum_diff_pred_used_found / count_avg_diff_pred_used_found
    avg_diff_pred_used_not_found = "-" if count_avg_diff_pred_used_not_found == 0 else sum_diff_pred_used_not_found / count_avg_diff_pred_used_not_found
    #print(count_not_full_samples)
    #print(count_not_used_samples)
    train_split = train_dir.split("_")
    #print(train_split)
    domain = train_split[2]
    #print(domain)
    problem = train_split[3]
    #print(problem)
    ss = train_split[-3].split(".")[0][2:]
    #print(ss)
    ns = train_split[-3].split(".")[1].split("-")[0][2:]
    #print(ns)
    shs = train_split[-2].split("-")[0]
    #print(shs)
    s_perc = train_split[-1]
    #print(sample_percentage)
    print(f"{domain},{problem},{ss},{ns},{shs},{s_perc},{num_samples_total},{count_not_full_samples},{count_not_used_samples},{avg_diff_pred_full_found},{avg_diff_pred_full_not_found},{avg_diff_pred_used_found},{avg_diff_pred_used_not_found}")
