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


def fukunaga_ferber(args, meth):
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
                out = f'{args.output_dir}/{meth}_{domain}_{instance_name}_{args.technique}_{args.state_representation}_{args.searches}x{args.samples_per_search}_ss{i}'
                if meth == "fukunaga":
                    cmd = (f'./fast-downward.py '
                           f'--sas-file {out}-output.sas --plan-file {out} '
                           f'--build release {instance} '
                           f'--search \'sampling_search_fukunaga(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()) '
                           f'techniques=[gbackward_fukunaga(searches={args.searches}, samples_per_search={args.samples_per_search}, '
                           f'technique={args.technique}, random_seed={i})], state_representation={state_repr}, '
                           f'random_seed={i}, match_heuristics={args.match_heuristics}, assignments_by_undefined_state={args.us_assignments}, '
                           f'contrasting_samples={args.contrasting})\'')
                    print(cmd)
                    if args.threads > 1:
                        run_multi_thread(cmd, args.threads)
                    else:
                        os.system(cmd)
                        #print(cmd)
                else:
                    # TODO Ferber
                    pass
    sas_files = glob(f'{args.output_dir}/*_{domain}_*-output.sas')
    for sf in sas_files:
        if os.path.isfile(sf):
            os.remove(sf)


def rsl(args):
    global COUNT
    global ID_COUNT
    first = True
    instances = glob(f"{args.instances_dir}/*.pddl")
    start = args.seed
    end = args.seed+1 if args.mult_seed <= 1 else args.mult_seed+1
    print(start, end)
    domain = ""
    for instance in instances:
        instance_split = instance.split('/')
        instance_name = instance_split[-1][:-5]
        domain = instance_split[-2]
        if instance_name != "domain" and instance_name != "source":
            for i in range(start, end):
                cmd = (f'./RSL/sampling.py --out_dir {args.output_dir} '
                       f'--instance {instance} --num_train_states {args.rsl_num_states} '
                       f'--num_demos {args.rsl_num_demos} --max_len_demo {args.rsl_max_len_demo} '
                       f'--random_sample_percentage {args.contrasting} --regression_method {args.technique} '
                       f'--check_state_invars {args.rsl_check_invars}')
                if args.threads > 1:
                    run_multi_thread(cmd, args.threads)
                else:
                    os.system(cmd)


def sample(args):
    os.system(f"tsp -K")
    os.system(f"tsp -S {args.threads}")
    args.match_heuristics = bool2str(args.match_heuristics)

    if args.method == "fukunaga" or args.method == "ferber":
        fukunaga_ferber(args, meth=args.method)
    elif args.method == "rsl":
        rsl(args)
    else:
        print("Invalid configuration.")
        exit(1)


def bool2str(b):
    return str(b).lower() if type(b) is bool else b


if __name__ == "__main__":
    sample(get_sample_args())
