#!/usr/bin/env python3

"""
Usage: $ ./run_large_exp.py exp.csv
Attention: The CSV file must be in the following format:

domain,instance,max_samples,max_epochs,max_expansions
blocks,path_to_instance/p07.pddl,999,99,9
blocks,path_to_instance/p09.pddl,777,77,7
grid,path_to_instance/p08.pddl,555,55,5
...
"""

import os
import csv
import time
from sys import argv

NET_SEEDS = 2
SAMPLE_SEEDS = 2
THREADS = 10

d = {}

with open(argv[1], 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        domain = row['domain']
        instance = row['instance']
        max_samples = row['max_samples']
        max_epochs = row['max_epochs']
        max_expansions = row['max_expansions']
        difficulty = row['difficult']
        if domain not in d:
            d[domain] = {}
        d[domain][instance] = {'max_samples': max_samples, 'max_epochs': max_epochs, 'max_expansions': max_expansions, 'difficulty': difficulty}
        

count = -1
up = 0
wait_else = 1
up_else = 0
os.system(f"tsp -K")
os.system(f"tsp -S {THREADS}")
for domain in d:
    for instance in d[domain]:
        curr = d[domain][instance]
        max_samples = curr['max_samples']
        max_epochs = curr['max_epochs']
        max_exp = curr['max_expansions']
        difficulty = curr['difficulty']
        instance_name = instance.split('/')[-1].split('.')[0]
        for ns in range(NET_SEEDS):
            for ss in range(SAMPLE_SEEDS):
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
                    
                sample_out_dir = f"samples/{domain}/{difficulty}/samples_{domain}_{instance_name}_bfsrw"
                if not os.path.exists(sample_out_dir):
                    os.makedirs(sample_out_dir)

                sample_cmd = f"{tsp_sample} ./fast-downward.py --sas-file {sample_out_dir}/yaaig_{domain}_{instance_name}_tech-bfsrw_subtech-percentage_avi-1_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss}-output.sas --plan-file {sample_out_dir}/yaaig_{domain}_{instance_name}_tech-bfsrw_subtech-percentage_avi-1_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss} --build release {instance} --search 'sampling_search_yaaig(eager_greedy([ff(transform=sampling_transform())], transform=sampling_transform()), techniques=[gbackward_yaaig(searches=1, samples_per_search=-1, max_samples={max_samples}, contrasting_percentage=0, bound_multiplier=1.0, technique=bfs_rw, subtechnique=percentage, bound=propositions_per_mean_effects, depth_k=99999, random_seed={ss}, restart_h_when_goal_state=true, state_filtering=true, bfs_percentage=10, allow_duplicates=interrollout, unit_cost=false, max_time=600.0, mem_limit_mb=2048)], state_representation=complete, random_seed={ss}, minimization=both, avi_k=1, avi_epsilon=-1, avi_unit_cost=false, avi_rule=vu_u, sort_h=false, mse_hstar_file=, mse_result_file={sample_out_dir}/yaaig_{domain}_{instance_name}_tech-bfsrw_subtech-percentage_avi-1_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss}_rmse, assignments_by_undefined_state=10, evaluator=blind())'"
                train_args = f"{sample_out_dir}/yaaig_{domain}_{instance_name}_tech-bfsrw_subtech-percentage_avi-1_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss} -mdl resnet -diff False -pte False -pat 100 -hl 2 -b 512 -e {max_epochs} -a relu -o regression -sb True -lo False -f 1 -clp 0 -lr 0.0001 -w 0 -no False -sibd 100 -hpred False -trd -1 -dnw 0 -d 0 -bi True -biout True -of results/{domain}/{difficulty}/results_{domain}_{instance_name}_bfsrw -rst True -s {ns} -sp False -spn -1 -rmg False -cfst False -sfst False -itc 0 -cut False -gpu False -tsize 0.9 -spt 1.0 -us False -ust False -cdead True -lf mse -wm kaiming_uniform -hu 250 -t 1800"
                test_args = f"-diff False -a eager_greedy -heu nn -e {max_exp} -t 300 -m 2048 -sdir None -atn 25 -ats 0 -pt all -dlog False -unit-cost False results/{domain}/{difficulty}/results_{domain}_{instance_name}_bfsrw/nfd_train.yaaig_{domain}_{instance_name}_tech-bfsrw_subtech-percentage_avi-1_dups-ir_min-both_repr-fs_bnd-propseff_ss{ss}.ns{ns}"
                train_test_cmd = f'{tsp_train} ./train-and-test.sh "{train_args}" "{test_args}"'
                print(sample_cmd)
                os.system(sample_cmd)
                print()
                print(train_test_cmd)
                os.system(train_test_cmd)
                print("----------------------------------------")
                count += 2
                up += 1
