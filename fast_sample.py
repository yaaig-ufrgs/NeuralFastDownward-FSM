#!/usr/bin/env python3

"""
./run_experiment.py --help

Observation: do not enable multi-threading for multiple runs of this same script.
             If using multi-threading, do it for only one instance directory.

NEEDS TESTING.

"""

import sys
import os
from glob import glob
from src.pytorch.utils.parse_args import get_sample_args
from src.pytorch.utils.default_args import SAMPLE_TECHNIQUE

COUNT = 0
ID_COUNT = 0
FIRST = True

def run_multi_thread(cmd, threads):
    global COUNT
    global ID_COUNT
    global FIRST
    thread_id = COUNT
    if COUNT < threads and FIRST:
        #os.system(f'tsp taskset -c {thread_id} ' + cmd)
        print(f'tsp taskset -c {thread_id} ' + cmd)
        COUNT += 1
    else:
        if FIRST or COUNT == threads:
            COUNT = 0
        FIRST = False
        #os.system(f'tsp -D {ID_COUNT} taskset -c {COUNT} ' + cmd)
        print(f'tsp -D {ID_COUNT} taskset -c {COUNT} ' + cmd)
        ID_COUNT += 1
        COUNT += 1


def get_full_state_repr_name(state_repr):
    if state_repr == "fs":
        return "complete"
    elif state_repr == "ps":
        return "partial"
    elif state_repr == "us":
        return "undefined"
    return state_repr


def yaaig_ferber(args, meth):
    search_algo = ""
    if args.search_algorithm == "greedy":
        search_algo = f'eager_greedy([{args.search_heuristic}(transform=sampling_transform())],transform=sampling_transform())'
    elif args.search_algorith == "astar":
        search_algo = f'astar({args.search_heuristic}(transform=sampling_transform()),transform=sampling_transform())'

    state_repr = get_full_state_repr_name(args.state_representation)
    instances = glob(f"{args.instances_dir}/*.pddl")
    start = args.seed
    end = args.seed+1 if args.mult_seed <= 1 else args.mult_seed+1
    domain = ""
    for instance in instances:
        instance_split = instance.split('/')
        instance_name = instance_split[-1][:-5]
        domain = instance_split[-2]
        if instance_name != "domain" and instance_name != "source":
            for i in range(start, end):
                cmd, out = "", ""
                if meth == "yaaig":
                    out = f'{args.output_dir}/{meth}_{domain}_{instance_name}_{args.technique}_{args.state_representation}_{args.searches}x{args.samples_per_search}_ss{i}'
                    cmd = (f'./fast-downward.py '
                           f'--sas-file {out}-output.sas --plan-file {out} '
                           f'--build release {instance} '
                           f'--search \'sampling_search_yaaig({search_algo}, '
                           f'techniques=[gbackward_yaaig(searches={args.searches}, samples_per_search={args.samples_per_search}, '
                           f'technique={args.technique}, random_seed={i}, restart_h_when_goal_state={args.restart_h_when_goal_state})], '
                           f'state_representation={state_repr}, random_seed={i}, minimization={args.minimization}, avi_k={args.avi_k}, '
                           f'assignments_by_undefined_state={args.us_assignments}, contrasting_samples={args.contrasting})\'')
                    print(cmd)
                elif meth == "ferber":
                    out = f'{args.output_dir}/{meth}_{domain}_{instance_name}_{args.ferber_technique}_{args.ferber_select_state.replace("_", "-")}_{args.ferber_num_tasks}_{args.ferber_min_walk_len}_{args.ferber_max_walk_len}_ss{i}'
                    cmd = (f'./fast-downward.py '
                           f'--sas-file {out}-output.sas --plan-file {out} '
                           f'--build release {instance} '
                           f'--search \'sampling_search_ferber({search_algo}, '
                           f'techniques=[{args.ferber_technique}_none({args.ferber_num_tasks}, '
                           f'distribution=uniform_int_dist({args.ferber_min_walk_len}, {args.ferber_max_walk_len}), random_seed={i})], '
                           f'select_state_method={args.ferber_select_state}, random_seed={i})\'')
                    pass
                if args.threads > 1:
                    run_multi_thread(cmd, args.threads)
                else:
                    os.system(cmd)
                    #print(cmd)


    sas_files = glob(f'{args.output_dir}/*_{domain}_*-output.sas')
    for sf in sas_files:
        if os.path.isfile(sf):
            os.remove(sf)


def rsl(args):
    global COUNT
    global ID_COUNT
    instances = glob(f"{args.instances_dir}/*.pddl")
    start = args.seed
    end = args.seed+1 if args.mult_seed <= 1 else args.mult_seed+1
    print(start, end)
    for instance in instances:
        instance_split = instance.split('/')
        instance_name = instance_split[-1][:-5]
        if instance_name != "domain" and instance_name != "source":
            for i in range(start, end):
                cmd = (f'./RSL/sampling.py --out_dir {args.output_dir} '
                       f'--instance {instance} --num_train_states {args.rsl_num_states} '
                       f'--num_demos {args.rsl_num_demos} --max_len_demo {args.rsl_max_len_demo} --seed {i} '
                       f'--random_sample_percentage {args.contrasting} --regression_method {args.technique} '
                       f'--check_state_invars {args.rsl_check_invars}')
                if args.threads > 1:
                    run_multi_thread(cmd, args.threads)
                else:
                    os.system(cmd)
                    print(cmd)


def sample(args):
    os.system(f"tsp -K")
    os.system(f"tsp -S {args.threads}")
    args.minimization = bool2str(args.minimization)
    args.ferber_technique = "iforward" if args.ferber_technique == "forward" else "gbackward"

    if args.method == "yaaig" or args.method == "ferber":
        yaaig_ferber(args, meth=args.method)
    elif args.method == "rsl":
        args.technique = "countBoth" if args.technique == SAMPLE_TECHNIQUE else args.technique
        rsl(args)
    else:
        print("Invalid configuration.")
        exit(1)


def bool2str(b):
    return str(b).lower() if type(b) is bool else b


if __name__ == "__main__":
    sample(get_sample_args())
