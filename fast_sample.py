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
from scripts.get_hstar_pddl import get_hstar_tasks

COUNT = 0
ID_COUNT = 0
FIRST = True

def run_multi_thread(cmd, threads):
    global COUNT
    global ID_COUNT
    global FIRST
    thread_id = COUNT
    if COUNT < threads and FIRST:
        print(f'tsp taskset -c {thread_id} ' + cmd)
        os.system(f'tsp taskset -c {thread_id} ' + cmd)
        COUNT += 1
    else:
        if FIRST or COUNT == threads:
            COUNT = 0
        FIRST = False
        print(f'tsp -D {ID_COUNT} taskset -c {COUNT} ' + cmd)
        os.system(f'tsp -D {ID_COUNT} taskset -c {COUNT} ' + cmd)
        ID_COUNT += 1
        COUNT += 1


def get_full_state_repr_name(state_repr):
    if state_repr == "fs":
        return "complete"
    elif state_repr == "fs-nomutex":
        return "complete_no_mutex"
    elif state_repr == "ps":
        return "partial"
    elif state_repr == "us":
        return "undefined"
    elif state_repr == "au":
        return "assign_undefined"
    elif state_repr == "uc":
        return "undefined_char"
    elif state_repr == "vps":
        return "values_partial"
    elif state_repr == "vfs":
        return "values_complete"
    elif state_repr == "vs":
        return "valid"
    return state_repr


def get_bound_type(bound):
    if str(bound).isdigit():
        return str(bound)
    if bound == "default":
        return "def"
    elif bound == "propositions":
        return "props"
    elif bound == "propositions_per_mean_effects":
        return "propseff"


def yaaig_ferber(args, meth):
    search_algo = ""
    if args.search_algorithm == "greedy":
        search_algo = f'eager_greedy([{args.search_heuristic}(transform=sampling_transform())], transform=sampling_transform())'
    elif args.search_algorithm == "astar":
        search_algo = f'astar({args.search_heuristic}(transform=sampling_transform()), transform=sampling_transform())'

    if args.technique == "dfs" or args.technique == "dfs_rw": # recheck this
        args.samples_per_search = int(1.0/args.searches*args.max_samples+0.999)

    if args.bound == "max_task_hstar":
        assert(args.test_tasks_dir != "")
        test_tasks = glob(f"{args.test_tasks_dir}/*")
        test_tasks += glob(f"{args.test_tasks_dir}/../*.pddl")
        args.bound = max(get_hstar_tasks("scripts", test_tasks))
        assert(args.bound > 0)

    state_repr = get_full_state_repr_name(args.state_representation)
    instances = [args.instance] if args.instance.endswith(".pddl") else glob(f"{args.instance}/*.pddl")

    if ".." in args.seed:
        start, end = [int(n) for n in args.seed.split('..')]
        end += 1
    else:
        start = int(args.seed)
        end = int(args.seed)+1 if int(args.mult_seed) <= 0 else int(args.mult_seed)+1

    domain = ""
    for instance in instances:
        instance_split = instance.split('/')
        instance_name = instance_split[-1][:-5]
        domain = instance_split[-2]
        if instance_name != "domain" and instance_name != "source":
            for i in range(start, end):
                cmd, out, subtech, depthk, avik, avits, dups, bound = "", "", "", "", "", "", "", ""
                tech = args.technique.replace('_', '')
                if args.technique == "dfs_rw" or args.technique == "bfs_rw":
                    subtech = f"_subtech-{args.subtechnique.replace('_', '')}"
                    if args.k_depth < 99999:
                        depthk = f"_depthk-{args.k_depth}"
                if args.avi_k > 0:
                    avik = f"_avi-{args.avi_k}"
                    # avik = f"_avi-{args.avi_k}"
                    # avi_iterations = "max" if args.avi_its >= 9999 else args.avi_its
                    # avits = f"_it-{avi_iterations}"
                if args.allow_dups != "none":
                    dups = "_dups-" + ("ir" if args.allow_dups == "interrollout" else args.allow_dups)
                # sps = f"srch-{args.searches}_sps-{args.samples_per_search}_maxs-{args.max_samples}" if args.samples_per_search != -1 else f"maxs-{args.max_samples}"
                sps = f"_maxs-{args.max_samples}" if args.max_samples != -1 else ""
                boundtype = f"bnd-{get_bound_type(args.bound)}"
                boundmult = "" if args.bound_multiplier == 1.0 else f"_bmul-{str(args.bound_multiplier).replace('.', '-')}"
                if meth == "yaaig":
                    out = f'{args.output_dir}/{meth}_{domain}_{instance_name}_tech-{tech}{subtech}{depthk}{avik}{avits}{dups}_min-{args.minimization}_repr-{args.state_representation}_{boundtype}{boundmult}{sps}_ss{i}'
                    rmse_out = out + "_rmse"
                    cmd = (f'./fast-downward.py '
                           f'--sas-file {out}-output.sas --plan-file {out} '
                           f'--build release {instance} '
                           f'{"--translate-options --unit-cost --search-options " if args.sample_unit_cost == "true" and args.evaluator == "pdb(hstar_pattern([]))" else ""}'
                           f'--search \"sampling_search_yaaig({search_algo}, '
                           f'techniques=[gbackward_yaaig(searches={args.searches}, samples_per_search={args.samples_per_search}, max_samples={args.max_samples}, '
                           f'bound_multiplier={args.bound_multiplier}, 'f'technique={args.technique}, subtechnique={args.subtechnique}, '
                           f'bound={args.bound}, depth_k={args.k_depth}, random_seed={i}, restart_h_when_goal_state={args.restart_h_when_goal_state}, '
                           f'allow_duplicates={args.allow_dups}, unit_cost={args.sample_unit_cost}, '
                           f'max_time={args.max_time}, mem_limit_mb={args.mem_limit})], '
                           f'state_representation={state_repr}, random_seed={i}, minimization={args.minimization}, '
                           f'avi_k={args.avi_k}, avi_its={args.avi_its}, avi_epsilon={args.avi_eps}, avi_unit_cost={args.sample_unit_cost}, '
                           f'avi_rule={args.avi_rule}, sort_h={args.sort_h}, mse_hstar_file={args.statespace}, mse_result_file={rmse_out}, '
                           f'assignments_by_undefined_state={args.us_assignments}, contrasting_samples={args.contrasting}, evaluator={args.evaluator})\"')
                elif meth == "ferber":
                    out = f'{args.output_dir}/{meth}_{domain}_{instance_name}_{args.ferber_technique}_{args.ferber_select_state.replace("_", "-")}_{args.ferber_num_tasks}_{args.ferber_min_walk_len}_{args.ferber_max_walk_len}_ss{i}'
                    cmd = (f'./fast-downward.py '
                           f'--sas-file {out}-output.sas --plan-file {out} '
                           f'--build release {instance} '
                           f'--search \"sampling_search_ferber({search_algo}, '
                           f'techniques=[{args.ferber_technique}_none({args.ferber_num_tasks}, '
                           f'distribution=uniform_int_dist({args.ferber_min_walk_len}, {args.ferber_max_walk_len}), random_seed={i})], '
                           f'select_state_method={args.ferber_select_state}, random_seed={i})\"')
                if args.threads > 1:
                    run_multi_thread(cmd, args.threads)
                else:
                    print(cmd)
                    os.system(cmd)


    if args.threads <= 1:
        sas_files = glob(f'{args.output_dir}/*-output.sas')
        for sf in sas_files:
            if os.path.isfile(sf):
                os.remove(sf)


def sample(args):
    os.system(f"tsp -K")
    os.system(f"tsp -S {args.threads}")
    args.restart_h_when_goal_state = bool2str(args.restart_h_when_goal_state)
    args.sample_unit_cost = bool2str(args.sample_unit_cost)
    args.sort_h = bool2str(args.sort_h)
    args.ferber_technique = "iforward" if args.ferber_technique == "forward" else "gbackward"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.method == "yaaig" or args.method == "ferber":
        yaaig_ferber(args, meth=args.method)
    else:
        print("Invalid configuration.")
        exit(1)


def bool2str(b):
    return str(b).lower() if type(b) is bool else b


if __name__ == "__main__":
    sample(get_sample_args())
