#!/usr/bin/env python3

"""
./run_experiment.py --help

Examples:
$ ./run_experiment.py samples -exp-type combined -trn-e 10 -exp-ns 5 -exp-ss 5
$ ./run_experiment.py samples -exp-eval true -evl-mdl results/*/models/traced_0.pt

"""

import sys
import os
from math import ceil
from glob import glob
from src.pytorch.utils.parse_args import get_exp_args
import src.pytorch.utils.default_args as default_args


COUNTER = 0
THREAD_ID = -1


def filter_samples(samples: [str], seed: int) -> [str]:
    filtered_samples = []
    for sample in samples:
        ss = sample.split("/")[1].split("_")[-1][2:]
        if ss.isnumeric() and (int(ss) == seed or seed == -1):
            filtered_samples.append(sample)
    return filtered_samples


def run_train_test(args, sample_seed: int, net_seed: int, runs: int):
    global COUNTER
    global THREAD_ID
    threads = args.exp_threads
    sample_files = glob(f"{args.samples}/*")
    sample_files = filter_samples(sample_files, sample_seed)
    files_len = len(sample_files) * runs
    max_per_thread = ceil(files_len / threads)

    for sample in sample_files:
        sample_name = sample.split("/")[1]
        sample_name_split = sample_name.split("_")
        sample_type = sample_name_split[0]
        domain = sample_name_split[1]
        instance = sample_name_split[2]
        technique = sample_name_split[3]
        sample_size = sample_name_split[4]
        trained_model_dir = (
            f"{args.train_output_folder}/nfd_train.{sample_name}.ns{net_seed}"
        )

        train_args = (
            f"{sample} -mdl {args.train_model} -pat {args.train_patience} "
            f"-hl {args.train_hidden_layers} -hu {args.train_hidden_units} "
            f"-b {args.train_batch_size} -e {args.train_max_epochs} -a {args.train_activation} "
            f"-o {args.train_output_layer} -sb {args.train_save_best_epoch_model} "
            f"-lo {args.train_linear_output} -f {args.train_num_folds} -clp {args.train_clamping} "
            f"-lr {args.train_learning_rate} -w {args.train_weight_decay} -no {args.train_normalize_output} "
            f"-sibd {args.train_seed_increment_when_born_dead} -hpred {args.train_save_heuristic_pred} "
            f"-trd {args.train_num_threads} -dnw {args.train_data_num_workers} "
            f"-d {args.train_dropout_rate} -bi {args.train_bias} -biout {args.train_bias_output} "
            f"-of {args.train_output_folder} -rst {args.train_restart_no_conv} "
            f"-s {net_seed} -shs {args.train_shuffle_seed} -sp {args.train_scatter_plot} -spn {args.train_plot_n_epochs} "
            f"-rmg {args.train_remove_goals} -cfst {args.train_contrast_first} "
            f"-sfst {args.train_standard_first} -itc {args.train_intercalate_samples} "
            f"-cut {args.train_cut_non_intercalated_samples} -gpu {args.train_use_gpu} "
            f"-tsize {args.train_training_size} -spt {args.train_sample_percentage} "
            f"-us {args.train_unique_samples} -ust {args.train_unique_states} "
            f"-lf {args.train_loss_function} -wm {args.train_weights_method}"
        )

        if args.train_max_training_time != default_args.MAX_TRAINING_TIME:
            train_args += f" -t {args.train_max_training_time}"

        if args.train_additional_folder_name != "":
            train_args += f" -addfn {args.train_additional_folder_name}"

        test_args = (
            f"-a {args.test_search_algorithm} -m {args.test_max_search_memory} "
            f"-sdir {args.test_samples_dir} -atn {args.test_auto_tasks_n} "
            f"-ats {args.test_auto_tasks_seed} -pt {args.test_test_model} "
            f"-dlog {args.test_downward_logs} {trained_model_dir}"
        )

        if args.problem_pddls != []:
            test_args += f" {args.problem_pddls}"
        if args.test_max_expansions != default_args.MAX_EXPANSIONS:
            test_args += f" -e {args.test_max_expansions}"

        if COUNTER % max_per_thread == 0:
            THREAD_ID += 1
            cmd = f"tsp taskset -c {THREAD_ID} ./train-and-test.sh '{train_args}' '{test_args}'"
            if args.exp_only_train:
                cmd = f"tsp taskset -c {THREAD_ID} ./train.py {train_args}"
            elif args.exp_only_test:
                cmd = f"tsp taskset -c {THREAD_ID} ./test.py {test_args}"
            os.system(cmd)
            #print(cmd + "\n")
        else:
            cmd = f"tsp -D {COUNTER-1} taskset -c {THREAD_ID} ./train-and-test.sh '{train_args}' '{test_args}'"
            if args.exp_only_train:
                cmd = f"tsp -D {COUNTER-1} taskset -c {THREAD_ID} ./train.py {train_args}"
            if args.exp_only_test:
                cmd = f"tsp -D {COUNTER-1} taskset -c {THREAD_ID} ./test.py {test_args}"
            os.system(cmd)
            #print(cmd + "\n")

        COUNTER += 1


def only_eval(args):
    """
    Batch-eval on trained models.
    """
    count = 0
    id_count = 0
    first = True

    eval_args = (f"-ls {args.eval_log_states} -sp {args.eval_save_preds} "
                 f"-s {args.eval_seed} -shs {args.eval_shuffle_seed} "
                 f"-sh {args.eval_shuffle} -tsize {args.eval_training_size} "
                 f"-us {args.eval_unique_samples} -ls {args.eval_log_states} "
                 f"-plt {args.eval_save_plots} -ft {args.eval_follow_training}"
                )

    if args.eval_trained_models[-1] == '*':
        args.eval_trained_models = glob(args.eval_trained_models)

    if len(args.eval_trained_models) == 0:
        print("ERROR: Trained models not found.")
        exit(1)

    sample_files = glob(f"{args.samples}/*")
    if len(sample_files) == 0:
        print("ERROR: Sample files not found.")
        exit(1)
    sample_files = " ".join(sample_files)

    for model in args.eval_trained_models:
        thread_id = count
        if count < args.exp_threads and first:
            os.system(
                f"tsp taskset -c {thread_id} ./eval.py {model} {sample_files} {eval_args}"
            )
            # print(f"tsp taskset -c {thread_id} ./eval.py {model} {sample_files} {eval_args}")
            count += 1
        else:
            if first or count == args.exp_threads:
                count = 0
            first = False
            os.system(
                f"tsp -D {id_count} taskset -c {count} ./eval.py {model} {sample_files} {eval_args}"
            )
            # print(f"tsp -D {id_count} taskset -c {count} ./eval.py {model} {sample_files} {eval_args}")
            id_count += 1
            count += 1


def experiment(args):
    args.train_hidden_units = (
        default_args.HIDDEN_UNITS[0]
        if args.train_hidden_units == default_args.HIDDEN_UNITS
        else args.train_hidden_units
    )
    args.test_max_search_time = (
        99999999
        if args.test_max_search_time == default_args.MAX_SEARCH_TIME
        else args.test_max_search_time
    )
    args.train_additional_folder_name = (
        ""
        if args.train_additional_folder_name == default_args.ADDITIONAL_FOLDER_NAME
        else " ".join(args.train_additional_folder_name)
    )

    net_seeds, sample_seeds = args.exp_net_seed.split('..'), args.exp_sample_seed.split('..')
    min_net_seed = int(net_seeds[0])
    max_net_seed = int(net_seeds[1]) if len(net_seeds) > 1 else int(net_seeds[0])
    min_sample_seed = int(sample_seeds[0])
    max_sample_seed = int(sample_seeds[1]) if len(sample_seeds) > 1 else int(sample_seeds[0])

    os.system(f"tsp -K")
    os.system(f"tsp -S {args.exp_threads}")

    if args.exp_only_eval:
        # Evaluate on all networks passed through argument.
        only_eval(args)
    else:
        if args.exp_type == "single":
            run_train_test(args, max_sample_seed, max_net_seed, runs=1)

        elif args.exp_type == "all":
            #total_runs = args.exp_sample_seed * args.exp_net_seed
            total_runs = ((max_sample_seed-min_sample_seed) + 1) * ((max_net_seed-min_net_seed) + 1)
            for i in range(min_sample_seed, max_sample_seed + 1):
                for j in range(min_net_seed, max_net_seed + 1):
                    run_train_test(args, i, j, runs=total_runs)

        elif args.exp_type == "combined":
            max_seed = max(max_net_seed, max_sample_seed)
            total_runs = (max_seed * 3) - 2
            run_train_test(
                args, 1, 1, runs=total_runs
            )
            for i in range(2, max_seed + 1):
                run_train_test(args, i, 1, runs=total_runs)
            for i in range(2, max_seed + 1):
                run_train_test(args, 1, i, runs=total_runs)
            for i in range(2, max_seed + 1):
                run_train_test(args, i, i, runs=total_runs)

        else:
            print("ERROR: Invalid experiment configuration.")
            exit(1)


if __name__ == "__main__":
    experiment(get_exp_args())
