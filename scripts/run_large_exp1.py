#!/usr/bin/env python3

"""
Usage: $ ./run_large_exp1.py large_tasks.csv tasks_info.csv
Attention: The CSV file must be in the following format:
...
"""

import os
import csv
import time
from sys import argv
from subprocess import check_output
import re
from random import random

NET_SEEDS = 1
SAMPLE_SEEDS = 1
THREADS = 10

d = {}

with open(argv[1], 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        domain = row['domain']
        instance = row['problem_pddl']
        difficulty = row['difficult']
        if domain not in d:
            d[domain] = {}
        d[domain][instance] = {'difficulty': difficulty}


tasks_info = {}
with open(argv[2]) as f:
    for line in [x.strip().split(",") for x in f.readlines()[1:]]:
        domain, problem = line[0], line[1]
        if domain not in tasks_info:
            tasks_info[domain] = {}
        tasks_info[domain][problem] = {
            "variables": int(line[2]),
            "mutex_groups": int(line[3]),
            "mutex_groups_size": int(line[4]),
            "operators": int(line[5]),
            "axiom_rules": int(line[6]),
            "goal_facts": int(line[7]),
            "fact_pairs": int(line[8]),
            "bytes_per_state": int(line[9]),
            "task_size": int(line[10])
        }

count = -1
up = 0
wait_else = 1
up_else = 0
os.system(f"tsp -K")
os.system(f"tsp -S {THREADS}")
for domain in d:
    for instance in d[domain]:
        curr = d[domain][instance]
        instance_name = instance.split('/')[-1].split('.')[0]
        ns, ss = 0, 0
        max_epochs = 5
        maxs = []
        mem_limit = 2048
        max_vars = (int(60000000 / tasks_info[domain][instance_name]["variables"]), "variables")
        maxs.append(max_vars)
        max_atoms = (int(800000000 / tasks_info[domain][instance_name]["fact_pairs"]), "atoms")
        maxs.append(max_atoms)
        max_bytes_per_state = (int(12000000 / tasks_info[domain][instance_name]["bytes_per_state"]), "bytes_per_state")
        maxs.append(max_bytes_per_state)
        max_normal = (400, "normal") # max_samples = -1 e mem_limit_mb = 400
        maxs.append(max_normal)
        for max_samples in maxs:
            if up < THREADS:
                tsp_sample = f"tsp taskset -c {up}"
                tsp_train = f"tsp -D {0 if count == -1 else count+1} taskset -c {up}"
            else:
                if up % THREADS == 0 or up_else % THREADS == 0:
                    up_else = 0
                tsp_sample = f"tsp -D {wait_else} taskset -c {up_else} "
                tsp_train = f"tsp -D {count+1} taskset -c {up_else}"
                wait_else += 2
                up_else += 1

            max_s = max_samples[0]
            extra_name = max_samples[1]
            if extra_name == "normal":
                max_s = -1
                mem_limit = 400

            sample_out_dir = f"samples/{domain}/{difficulty}/samples_{domain}_{instance_name}_bfsrw"
            if not os.path.exists(sample_out_dir):
                os.makedirs(sample_out_dir)

            sample_cmd = f"{tsp_sample} ./fast-downward.py --sas-file {sample_out_dir}/yaaig_{domain}_{instance_name}_{extra_name}_{max_s}_tech-bfsrw_subtech-percentage_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss}-output.sas --plan-file {sample_out_dir}/yaaig_{domain}_{instance_name}_{extra_name}_{max_s}_tech-bfsrw_subtech-percentage_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss} --build release {instance} --search 'sampling_search_yaaig(eager_greedy([ff(transform=sampling_transform())], transform=sampling_transform()), techniques=[gbackward_yaaig(searches=1, samples_per_search=-1, max_samples={max_s}, bound_multiplier=1.0, technique=bfs_rw, subtechnique=percentage, bound=propositions_per_mean_effects, depth_k=99999, random_seed={ss}, restart_h_when_goal_state=true, allow_duplicates=interrollout, unit_cost=false, max_time=600.0, mem_limit_mb={mem_limit})], state_representation=complete, random_seed={ss}, minimization=both, avi_k=0, avi_its=9999, avi_epsilon=-1, avi_unit_cost=false, avi_rule=vu_u, sort_h=false, mse_hstar_file=, mse_result_file={sample_out_dir}/yaaig_{domain}_{instance_name}_{extra_name}_{max_s}_tech-bfsrw_subtech-percentage_avi-1_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss}_rmse, assignments_by_undefined_state=10, contrasting_samples=0, evaluator=blind())'"
            train_args = f"{sample_out_dir}/yaaig_{domain}_{instance_name}_{extra_name}_{max_s}_tech-bfsrw_subtech-percentage_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss} -mdl resnet -diff False -pte False -pat 100 -hl 2 -b 512 -e {max_epochs} -a relu -o regression -sb True -lo False -f 1 -clp 0 -lr 0.0001 -w 0 -no False -sibd 100 -hpred False -trd -1 -dnw 0 -d 0 -bi True -biout True -of results/{domain}/{difficulty}/results_{domain}_{instance_name}_bfsrw -rst True -s {ns} -sp False -spn -1 -rmg False -cfst False -sfst False -itc 0 -cut False -gpu False -tsize 0.9 -spt 1.0 -us False -ust False -cdead True -lf mse -wm kaiming_uniform -hu 250 -t 1800"
            train_cmd = f"{tsp_train} ./train.py {train_args}"
            print(sample_cmd)
            os.system(sample_cmd)
            print()
            print(train_cmd)
            os.system(train_cmd)
            print("----------------------------------------")
            count += 2
            up += 1
