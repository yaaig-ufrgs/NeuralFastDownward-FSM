#!/usr/bin/env python3

import os
import re
from glob import glob
from src.pytorch.utils.parse_args import get_sample_args
from scripts.get_hstar_pddl import get_hstar_tasks

COUNT = 0
ID_COUNT = 0
FIRST = True


PID = 0

def run_multi_core(cmd, cores):
    global PID
    pcore = PID % cores
    pdep = PID - cores
    cmd = f"tsp taskset -c {pcore} {cmd}"
    if pdep >= 0:
        cmd = cmd.replace("tsp", f"tsp -D {pdep}")
    print("fast-sample.py:", cmd, end="\n\n")
    os.system(cmd)
    PID += 1


def get_bound_type(regression_depth):
    if str(regression_depth).isdigit():
        return str(regression_depth)
    if regression_depth == "default":
        return "def"
    elif regression_depth == "facts":
        return "facts"
    elif regression_depth == "facts_per_avg_effects":
        return "factseff"


def yaaig_sample(args, meth):
    search_algo = ""
    if args.search_algorithm == "greedy":
        search_algo = f'eager_greedy([{args.search_heuristic}(transform=sampling_transform())], transform=sampling_transform())'
    elif args.search_algorithm == "astar":
        search_algo = f'astar({args.search_heuristic}(transform=sampling_transform()), transform=sampling_transform())'

    if args.technique == "dfs":
        args.samples_per_search = int(1.0/args.searches*args.max_samples+0.999)

    if args.regression_depth == "max_task_hstar":
        assert(args.test_tasks_dir != "")
        test_tasks = glob(f"{args.test_tasks_dir}/*")
        test_tasks += glob(f"{args.test_tasks_dir}/../*.pddl")
        args.regression_depth = max(get_hstar_tasks("scripts", test_tasks))
        assert(args.regression_depth > 0)
    elif args.regression_depth == "state_space_diameter":
        # expected state-space filename: statespace_transportunit_transport_hstar
        assert args.statespace
        statespace_unit = args.statespace.split("_")
        statespace_unit[-3] += "unit"
        statespace_unit = "_".join(statespace_unit)
        statespace_regression_depth_file = args.statespace if not os.path.exists(statespace_unit) else statespace_unit
        assert statespace_regression_depth_file
        max_h = 0
        with open(statespace_regression_depth_file, "r") as ss_file:
            for h, _ in [l.split(";") for l in ss_file.readlines() if not l.startswith("#")]:
                max_h = max(max_h, int(h))
        args.regression_depth = max_h

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
                cmd, out, depthk, sui, suits, dups = "", "", "", "", "", ""
                tech = args.technique.replace('_', '')
                if args.technique == "bfs_rw":
                    if args.k_depth < 99999:
                        depthk = f"_depthk-{args.k_depth}"
                if args.successor_improvement:
                    sui = f"_sui"
                if args.allow_dups != "none":
                    dups = "_dups-" + ("ir" if args.allow_dups == "interrollout" else args.allow_dups)
                sps = f"_maxs-{args.max_samples}" if args.max_samples != -1 else ""
                boundtype = f"bnd-{get_bound_type(args.regression_depth)}"
                boundmult = "" if args.regression_depth_multiplier == 1.0 else f"_bmul-{str(args.regression_depth_multiplier).replace('.', '-')}"
                rsquant = "" if args.random_percentage == 0 else f"_rs-{round(args.max_samples*(args.random_percentage))}"
                fd_build = "debug" if args.debug else "release"
                assert meth == "yaaig"
                out = f'{args.output_dir}/{meth}_{domain}_{instance_name}_tech-{tech}{depthk}{sui}{suits}{dups}_sai-{args.sample_improvement}_repr-{args.state_representation}_{boundtype}{boundmult}{sps}{rsquant}_ss{i}'
                cmd = (f'./fast-downward.py '
                        f'--sas-file {out}-output.sas --plan-file {out} '
                        f'--build {fd_build} {instance} '
                        f'{"--translate-options --unit-cost --search-options " if args.unit_cost == "true" and args.evaluator == "pdb(hstar_pattern([]))" else ""}'
                        f'--search \"sampling_search_yaaig({search_algo}, '
                        f'techniques=['
                        f'gbackward_yaaig('
                        f'searches={args.searches}, '
                        f'samples_per_search={args.samples_per_search}, '
                        f'max_samples={args.max_samples}, '
                        f'random_percentage={args.random_percentage}, '
                        f'regression_depth_multiplier={args.regression_depth_multiplier}, '
                        f'technique={args.technique}, '
                        f'regression_depth={args.regression_depth}, '
                        f'depth_k={args.k_depth}, '
                        f'random_seed={i}, '
                        f'restart_h_when_goal_state={args.restart_h_when_goal_state}, '
                        f'state_filtering={args.state_filtering}, '
                        f'bfs_percentage={args.bfs_percentage}, '
                        f'allow_duplicates={args.allow_dups}, '
                        f'unit_cost={args.unit_cost}, '
                        f'max_time={args.max_time}, '
                        f'mem_limit_mb={args.mem_limit})], '
                        f'state_representation={args.state_representation}, '
                        f'random_seed={i}, '
                        f'sai={args.sample_improvement}, '
                        f'sui={args.successor_improvement}, '
                        f'sui_rule={args.sui_rule}, '
                        f'statespace_file={args.statespace}, '
                        f'random_value={str(args.random_value)}, '
                        f'random_multiplier={args.random_multiplier}, '
                        f'evaluator={args.evaluator})\"')
                if args.cores > 1:
                    run_multi_core(cmd, args.cores)
                else:
                    print(cmd)
                    os.system(cmd)


    if args.cores <= 1:
        sas_files = glob(f'{args.output_dir}/*-output.sas')
        for sf in sas_files:
            if os.path.isfile(sf):
                os.remove(sf)


def sample(args):
    global PID
    args.pid = int(args.pid)
    if args.pid == 0:
        os.system(f"tsp -K")
        os.system(f"tsp -S {args.cores}")
    else:
        PID = args.pid

    args.restart_h_when_goal_state = bool2str(args.restart_h_when_goal_state)
    args.state_filtering = bool2str(args.state_filtering)
    args.unit_cost = bool2str(args.unit_cost)
    args.successor_improvement = bool2str(args.successor_improvement)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.method == "yaaig":
        yaaig_sample(args, meth=args.method)
    else:
        print("Invalid configuration.")
        exit(1)

    with open("PID", 'w') as f:
        f.write(str(PID))


def bool2str(b):
    return str(b).lower() if type(b) is bool else b


if __name__ == "__main__":
    sample(get_sample_args())
