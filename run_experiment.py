#!/usr/bin/env python3

"""
./run_experiment.py --help

Example:
$ ./run_experiment.py samples -exp-type combined -trn-e 10 -exp-ns 5 -exp-ss 5

TODO: Make test work when we have have model folders with  '.1', '.2', etc.
      Alternative, for now: train and test separately.
"""

import sys
import os
from math import ceil
from glob import glob
from src.pytorch.utils.parse_args import get_exp_args
from src.pytorch.utils.default_args import (
    DEFAULT_HIDDEN_UNITS,
    DEFAULT_MAX_TRAINING_TIME,
    DEFAULT_MAX_SEARCH_TIME,
    DEFAULT_MAX_EXPANSIONS,
)


COUNTER = 0
THREAD_ID = -1


def filter_samples(samples: [str], seed: int) -> [str]:
    filtered_samples = []
    for sample in samples:
        ss = sample.split('/')[1].split('_')[-1][2:]
        if ss.isnumeric() and (int(ss) == seed or seed == -1):
            filtered_samples.append(sample)
    return filtered_samples


def run_train_test(args, sample_seed: int, net_seed: int, runs: int):
    global COUNTER
    global THREAD_ID
    threads = args.exp_threads
    sample_files = glob(f"{args.samples}/*")
    sample_files = filter_samples(sample_files, sample_seed)
    files_len = len(sample_files)*runs
    max_per_thread = ceil(files_len/threads)

    for sample in sample_files:
        sample_name = sample.split('/')[1]
        sample_name_split = sample_name.split('_')
        sample_type = sample_name_split[0]
        domain = sample_name_split[1]
        instance = sample_name_split[2]
        technique = sample_name_split[3]
        sample_size = sample_name_split[4]
        trained_model_dir = f"{args.train_output_folder}/nfd_train.{sample_name}.ns{net_seed}"

        train_args = (f'{sample} -mdl {args.train_model} -pat {args.train_patience} '
                      f'-hl {args.train_hidden_layers} -hu {args.train_hidden_units} '
                      f'-b {args.train_batch_size} -e {args.train_max_epochs} '
                      f'-lr {args.train_learning_rate} -w {args.train_weight_decay} '
                      f'-d {args.train_dropout_rate} -bi {args.train_bias} '
                      f'-of {args.train_output_folder} -rst {args.train_restart_no_conv} '
                      f'-s {net_seed} -shs {args.train_shuffle_seed} '
                      f'-rmg {args.train_remove_goals}')
        if args.train_max_training_time != DEFAULT_MAX_TRAINING_TIME:
            train_args += f' -t {args.train_max_training_time}'

        test_args = (f'-a {args.test_search_algorithm} -m {args.test_max_search_memory} '
                     f'-sdir {args.test_samples_dir} -atn {args.test_auto_tasks_n} '
                     f'-dlog {args.test_downward_logs} {trained_model_dir}')

        if args.problem_pddls != []:
            test_args += f' {args.problem_pddls}'
        if args.test_max_expansions != DEFAULT_MAX_EXPANSIONS:
            test_args += f' -e {args.test_max_expansions}'

        if COUNTER % max_per_thread == 0:
            THREAD_ID += 1
            os.system(f"tsp taskset -c {THREAD_ID} ./train-and-test.sh '{train_args}' '{test_args}'")
            #print(f"tsp taskset -c {THREAD_ID} ./train-and-test.sn '{train_args}' '{test_args}'")
        else:
            os.system(f"tsp -D {COUNTER-1} taskset -c {THREAD_ID} ./train-and-test.sh '{train_args}' '{test_args}'")
            #print(f"tsp -D {COUNTER-1} taskset -c {THREAD_ID} ./train-and-test.sh '{train_args}' '{test_args}'")

        COUNTER += 1
        

def only_test(args):
    """
    Batch-testing on trained models.
    """
    count = 0
    id_count = 0
    first = True

    test_args = (f'-a {args.test_search_algorithm} -m {args.test_max_search_memory} '
                 f'-sdir {args.test_samples_dir} -atn {args.test_auto_tasks_n} '
                 f'-dlog {args.test_downward_logs}')

    if args.problem_pddls != []:
        test_args += f' {args.problem_pddls}'
    if args.test_max_expansions != DEFAULT_MAX_EXPANSIONS:
        test_args += f' -e {args.test_max_expansions}'

    for net in glob(f'{args.tst_model_dir}/*'):
        if "nfd_train" in net:
            thread_id = count
            if count < args.exp_threads and first:
                os.system(f"tsp taskset -c {thread_id} ./test.py {net} {test_args}")
                count += 1
            else:
                if first or count == args.exp_threads:
                    count = 0
                first = False
                os.system(f"tsp -D {id_count} taskset -c {count} ./test.py {net} {test_args}")
                id_count += 1
                count += 1
                

def only_train(args):
    """
    Batch-testing on trained models.
    """
    count = 0
    id_count = 0
    first = True

    train_args = (f'-mdl {args.train_model} -pat {args.train_patience} '
                  f'-hl {args.train_hidden_layers} -hu {args.train_hidden_units} '
                  f'-b {args.train_batch_size} -e {args.train_max_epochs} '
                  f'-lr {args.train_learning_rate} -w {args.train_weight_decay} '
                  f'-d {args.train_dropout_rate} -bi {args.train_bias} '
                  f'-of {args.train_output_folder} -rst {args.train_restart_no_conv} '
                  f'-s {args.exp_net_seed} -shs {args.train_shuffle_seed} '
                  f'-rmg {args.train_remove_goals}')

    if args.train_max_training_time != DEFAULT_MAX_TRAINING_TIME:
        train_args += f' -t {args.train_max_training_time}'

    sample_files = filter_samples(glob(f"{args.samples}/*"), args.exp_sample_seed)

    for sample in sample_files:
        thread_id = count
        if count < args.exp_threads and first:
            os.system(f"tsp taskset -c {thread_id} ./train.py {sample} {train_args}")
            #print(f"tsp taskset -c {thread_id} ./train.py {sample} {train_args}")
            count += 1
        else:
            if first or count == args.exp_threads:
                count = 0
            first = False
            os.system(f"tsp -D {id_count} taskset -c {count} ./train.py {sample} {train_args}")
            #print(f"tsp -D {id_count} taskset -c {count} ./train.py {sample} {train_args}")
            id_count += 1
            count += 1


def experiment(args):
    args.train_hidden_units = DEFAULT_HIDDEN_UNITS[0] if args.train_hidden_units == DEFAULT_HIDDEN_UNITS else args.train_hidden_units
    args.test_max_search_time = 99999999 if args.test_max_search_time == DEFAULT_MAX_SEARCH_TIME else args.test_max_search_time

    os.system(f"tsp -K")
    os.system(f"tsp -S {args.exp_threads}")

    if args.exp_only_train:
        only_train(args)
    elif args.exp_only_test:
        only_test(args)
    else:
        if args.exp_type == "single":
            run_train_test(args, args.exp_sample_seed, args.exp_net_seed, runs=1)

        elif args.exp_type == "fixed_net_seed":
            for i in range(1, args.exp_sample_seed+1):
                run_train_test(args, i, args.exp_net_seed, runs=args.exp_sample_seed)

        elif args.exp_type == "fixed_sample_seed":
            for i in range(1, args.exp_net_seed+1):
                run_train_test(args, args.exp_sample_seed, i, runs=args.exp_net_seed)

        elif args.exp_type == "change_all":
            for i in range(1, args.exp_net_seed+1):
                run_train_test(args, i, i, runs=args.exp_net_seed)

        elif args.exp_type == "all":
            total_runs = args.exp_sample_seed * args.exp_net_seed
            for i in range(1, args.exp_sample_seed+1):
                for j in range(1, args.exp_net_seed+1):
                    run_train_test(args, i, j, runs=total_runs)

        elif args.exp_type == "combined":
            max_seed = max(args.exp_net_seed, args.exp_sample_seed)
            total_runs = (max_seed * 3) - 2
            run_train_test(args, args.exp_fixed_seed, args.exp_fixed_seed, runs=total_runs)
            for i in range(2, max_seed+1):
                run_train_test(args, i, args.exp_fixed_seed, runs=total_runs)
            for i in range(2, max_seed+1):
                run_train_test(args, args.exp_fixed_seed, i, runs=total_runs)
            for i in range(2, max_seed+1):
                run_train_test(args, i, i, runs=total_runs)

        else:
            print("Invalid experiment configuration.")
            exit(1)


if __name__ == "__main__":
    experiment(get_exp_args())
