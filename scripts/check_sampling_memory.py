#!/usr/bin/env python3

"""
usage: python3 scripts/check_sampling_memory.py reference/large_tasks.csv reference/tasks_info.csv
"""

from sys import argv
from subprocess import check_output
import re
from random import random

search_command = "sampling_search_yaaig(eager_greedy([ff(transform=sampling_transform())], transform=sampling_transform()), "\
    "techniques=[gbackward_yaaig(searches=1, samples_per_search=-1, max_samples={max_samples}, bound_multiplier=1.0, technique=bfs_rw, "\
    "subtechnique=percentage, bound=propositions_per_mean_effects, depth_k=99999, random_seed=0, restart_h_when_goal_state=true, "\
    "allow_duplicates=interrollout, unit_cost=false, max_time=600, mem_limit_mb=2048)], state_representation=complete, random_seed=0, "\
    "minimization=both, avi_k=1, avi_its=9999, avi_epsilon=-1, avi_unit_cost=false, avi_rule=vu_u, sort_h=false, mse_hstar_file=, "\
    "mse_result_file=samples/yaaig_npuzzle_prob-n9-4_tech-bfsrw_subtech-percentage_avi-1_dups-ir_min-both_repr-fs_bnd-propseff_ss0_rmse, "\
    "assignments_by_undefined_state=10, contrasting_samples=0, evaluator=blind())"

def get_mem(line, output):
    try:
        peak, curr = re.findall(f".*[t=.*, peak=(\d*) KB, curr=(\d*) KB] {line}.*".replace("[", "\[").replace("]", "\]"), output)[0]
        return str(int(int(peak)/1000)), str(int(int(curr)/1000))
    except Exception as e:
        return "NA", "NA"

pddls = {}
with open(argv[1]) as f:
    for line in [x.strip().split(",") for x in f.readlines()[1:]]:
        domain, problem = line[0], line[1]
        if domain not in pddls:
            pddls[domain] = {}
        pddls[domain][problem] = line[3]

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

unique = str(random())[2:]
print("domain,problem,max_samples,curr_bfs,curr_bfs_rw,curr_trie_avi,curr_map_avi,curr_avi,curr_end,peak_bfs,peak_bfs_rw,peak_trie_avi,peak_map_avi,peak_avi,peak_end")
for domain in pddls:
    for problem in pddls[domain]:
        # SET MAX SAMPLES
        max_samples = 2000000 / tasks_info[domain][problem]["variables"]
        max_samples = int(max_samples)

        output = check_output([
            "./fast-downward.py",
            "--sas-file", f"{unique}-output.sas",
            "--plan-file", f"{unique}_sas_plan",
            "--build", "release",
            pddls[domain][problem],
            "--search", search_command.format(max_samples=max_samples)
        ]).decode("utf-8")
        assert "Solution found." in output

        mem_bfs_sampling = get_mem("Starting random walk search", output)
        mem_all_sampling = get_mem(".Extracting samples", output)
        mem_trie_avi = get_mem("[AVI] Time creating trie", output)
        mem_mapping_avi = get_mem("[AVI] Time creating AVI mapping", output)
        mem_avi = get_mem("[AVI] Total time", output)
        mem_end = get_mem("Search time:", output)

        print(
            domain, problem, max_samples,
            mem_bfs_sampling[1], mem_all_sampling[1], mem_trie_avi[1], mem_mapping_avi[1], mem_avi[1], mem_end[1],
            mem_bfs_sampling[0], mem_all_sampling[0], mem_trie_avi[0], mem_mapping_avi[0], mem_avi[0], mem_end[0],
            sep=",", flush=True
        )

        break
    break
