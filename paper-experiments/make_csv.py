#!/usr/bin/env python3

"""
Compiles multiple result folders into a single table.
Choose table columns in variable `HEADER`.

Usage: ./compile_results.py train_folder [train_folder ...]
  e.g. ./compile_results.py ../../NeuralFastDownward/results/nfd_train.*

This script is currently a mess; will change before paper submission.
"""

from sys import argv
from json import load
import csv

# training columns:
#     hostname, date, commit, domain, problem, samples, model, patience, output_layer
#     linear_output, num_folds, hidden_layers, hidden_units, batch_size, learning_rate
#     max_epochs, max_training_time, activation, weight_decay, dropout_rate, training_size
#     sample_percentage, shuffle, shuffle_seed, remove_goals, standard_first, contrast_first
#     intercalate_samples, cut_non_intercalated_samples, dataloader_num_workers, gpu, bias
#     bias_output, normalize_output, weights_method, network_seed, num_threads, train_loss,
#     val_loss, training_time, epochs, fold

# test columns:
#     search_algorithm, heuristic, max_search_time, max_search_memory, max_expansions, plans_found,
#     total_problems, coverage, max_plan_length, min_plan_length, avg_plan_length, mdn_plan_length,
#     avg_plan_cost, mdn_plan_cost, avg_initial_h, mdn_initial_h, avg_expanded, mdn_expanded,
#     avg_reopened, mdn_reopened, avg_evaluated, mdn_evaluated, avg_generated, mdn_generated,
#     avg_dead_ends, mdn_dead_ends, avg_search_time, mdn_search_time, avg_expansion_rate,
#     mdn_expansion_rate, total_accumulated_time

# sample columns:
#     sampling_technique, sampling_algorithm, sampling_amount, sample_seed, state_representation

#####################################################################
# ADD HERE ALL ARGS (AND ORDER) THAT SHOULD BE ADDED TO THE TABLE
HEADER = [
    "domain", "problem", "test_algorithm", "test_heuristic", "statespace_size",
    "sampling_algorithm", "preprocessing_method", "bfs_samples", "rw_samples", "total_samples",
    "rw_bound_method", "rw_bound", "bound_multiplier", "sample_seed", "network_seed", "epochs", "max_epochs", "train_loss", "val_loss", "training_time",
    "plans_found", "total_problems", "coverage",
    "avg_plan_length", "avg_plan_cost", "avg_initial_h", "avg_expanded", "avg_evaluated", "avg_generated", "avg_search_time", "total_accumulated_time", 
]
#####################################################################

# abbrevs
state_representation = {
    "fs": "full state",
    "ps": "partial assignment"
}
sampling_algorithma = {
    # "rw": "random walk",
    "dfs": "dfs",
    "bfs": "bfs",
    "countBoth": "count_both",
    "rw-hstar": "h*",
    "rw-gbfs-ff": "gbfs(ff)",
    "bfsrw": "bfs_rw"
}

# ---

ss_samples = {
    "blocks": 65991,
    "grid": 452353,
    "npuzzle": 181441,
    "rovers": 565824,
    "scanalyzer": 46080,
    "scanalyzer_unitcost": 46080,
    "storage": 63547,
    "transport": 637632,
    "transport_unitcost": 637632,
    "visitall": 79931,
}

bounds = {
    "blocks": [200, 64, 17, 24],
    "grid": [200, 76, 44, 32],
    "npuzzle": [200, 81, 41, 31],
    "rovers": [200, 32, 27, 19],
    "scanalyzer_unitcost": [200, 42, 20, 15],
    "scanalyzer": [200, 42, 20, 15],
    "transport_unitcost": [200, 66, 35, 17],
    "transport": [200, 66, 35, 17],
    "visitall": [200, 31, 17, 15],
}

# ---

HEADER_LINE = ",".join(HEADER) #replace("sample_seed","ss").replace("network_seed","ns")

mapp = {}
for train_folder in argv[1:]:
    if train_folder[-1] == "/":
        train_folder = train_folder[:-1]
    test_folder = f"{train_folder}/tests/nfd_test"
    # expected sample_file format = yaaig_blocks_probBLOCKS-7-1_dfs_fs_500x200_ss1
    sample_file = train_folder.split("/")[-1].split(".")[1]

    try:
        with open(f"{train_folder}/train_args.json") as train_args_file:
            train_args = load(train_args_file)
    except Exception as _:
        train_args = {}
    try:
        with open(f"{train_folder}/eval_results.csv") as eval_results_file:
            reader = csv.DictReader(eval_results_file)
            for row in reader:
                eval_rmse = row['rmse']
                eval_diff = row['mean_loss']
                break
    except Exception as _:
        eval_rmse = "NA"
    try:
        with open(f"{train_folder}/nfd.log") as train_log_file:
            train_log = train_log_file.read().replace("\n", " ")
    except Exception as _:
        train_log = None
    try:
        with open(f"{test_folder}/test_results.json") as test_results_file:
            test_results = load(test_results_file)
    except:
        continue
        # test_results = {"statistics" : "None"}

    for fold in test_results["statistics"]:
        line = []

        domain = train_args["domain"] if "domain" in train_args else train_folder.split("/")[-1].split(".")[1].split("_")[1]
        domain = domain.replace("-opt14-strips", "")
        samples_file = train_args["samples"].split("/")[2] if "samples" in train_args else "NA"

        sampling_algorithm = samples_file.split("-")[2] if samples_file != "NA" else "NA"
        if sampling_algorithm == "baseline":
            sampling_algorithm = "rw"
        if sampling_algorithm == "bfs_rw":
            sampling_algorithm = "bfs_rw"

        #if samples_file.endswith("-unitcost"):
        if "unit" in samples_file:
            samples_file = samples_file.split("-unitcost")[0]
            domain = domain + "_unitcost"
        domain = domain.replace("-unitcost", "_unitcost")

        if domain == "scanalyzerunit" or domain == "scanalyzerunit_unitcost" :
            domain = "scanalyzer_unitcost"
        elif domain == "transportunit":
            domain = "transport_unitcost"

        if "-random-sample-" in samples_file or "randomsample" in samples_file:
            samples_file = samples_file.replace("-random-sample", "") + "-random_sample"

        # fix blocks1pct
        for dd in ["blocks", "grid", "npuzzle", "rovers", "scanalyzer-unitcost", "scanalyzer", "transport-unitcost", "transport", "visitall"]:
            samples_file = samples_file.replace(f"{dd}1pct", f"{dd}-1pct")
        domain = domain.replace("-opt14-strips", "")

        for h in HEADER:
            try:
                # training columns
                if h == "network_seed":
                    h = "seed"
                elif h == "test_heuristic":
                    h = "heuristic"
                elif h == "test_algorithm":
                    h = "search_algorithm"

                if h == "preprocessing_method":
                    if samples_file == "NA":
                        value = "NA"
                    else:
                        value = samples_file.split("-")[-1]
                        if value not in ["nomin", "vs", "noavi", "nomutex", "hstar"]:
                            value = "best"
                        elif value == "vs":
                            value = "perfect_mutex"
                        if value.startswith("no"):
                            value = "no_" + value[2:]
                        if "baseline" in samples_file or "noavinomin" in train_folder:
                            value = "no_min_no_avi"
                        if "-vs-hstar" in samples_file:
                            value = "perfect_mutex_hstar"
                        if "-nomutex-hstar" in samples_file:
                            value = "nomutex_hstar"

                elif h == "sampling_algorithm":
                    if samples_file == "NA":
                        value = "NA"
                    else:
                        value = sampling_algorithm
                        if value in sampling_algorithma:
                            value = sampling_algorithma[value]

                elif h == "rw_bound_method" or h == "rw_bound":
                    if samples_file == "NA":
                        value = "NA"
                    else:
                        value = samples_file.split("-", 3)[-1]
                        value = value\
                            .replace("-nomin", "")\
                            .replace("-nomutex", "")\
                            .replace("-noavi", "")\
                            .replace("-hstar", "")\
                            .replace("-vs", "")\
                            .replace("bfs-", "")\
                            .replace("rw-", "")
                        if value == "baseline-1": value = "default"
                        if value == "200nomin": value = "default"
                        if value == "propeff-1-25" or value == "propeffnomin" or value == "propeff-1-50" or value == "propeff-2-00" or value == "propeff-2-50" or value == "propeff-3-00":
                            value = "propeff"
                        if value == "propeff-1-25nomin" or value == "propeffnomin" or value == "propeff-1-50nomin" or value == "propeff-2-00nomin" or value == "propeff-2-50nomin" or value == "propeff-3-00nomin":
                            value = "propeff"
                        if value == "propnomin":
                            value = "prop"
                        if value.endswith("-f1"): value = value.rsplit("-f1", 1)[0]
                        if value.endswith("-avi"): value = value.rsplit("-avi", 1)[0]
                        if h == "rw_bound":
                            if value.endswith("-unitcost"):
                                domain += "_unitcost" # assert "unitcost" in domain
                                value = value.split("-unitcost")[0]
                            if value == "default" or value == "200" or value == "200nomin": value = bounds[domain][0]
                            elif value == "propositions" or value == "prop": value = bounds[domain][1]
                            elif value == "propositions-eff" or value == "propseff" or value == "propeff" or value == "propeff-1-25" or value == "propeff-1-50" or value == "propeff-2-00" or value == "propeff-2-50" or value == "propeff-3-00":
                                value = bounds[domain][2]
                            elif value == "random_sample" or value == "randomsample": value = "NA"
                            #elif value == "diameter": value = int(train_folder.split('/')[-1].split('_')[-3].split('-')[-1])
                            elif value == "diameter": value = bounds[domain][3]
                            else:
                                print(value)
                                assert False, value + " ... " + train_folder

                elif h == "bfs_samples":
                    if samples_file == "NA":
                        value = "NA"
                    else:
                        pct = int(samples_file.split("-")[1].split("pct")[0]) * 0.01
                        if "bfs" in sampling_algorithm:
                            value = int(round(ss_samples[domain] * pct * 0.1, 0))
                        else:
                            value = 0
                elif h == "rw_samples":
                    if samples_file == "NA":
                        value = "NA"
                    else:
                        pct = int(samples_file.split("-")[1].split("pct")[0]) * 0.01
                        if "bfs" in sampling_algorithm:
                            value = int(round(ss_samples[domain] * pct, 0)) - int(round(ss_samples[domain] * pct * 0.1, 0))
                        else:
                            value = int(round(ss_samples[domain] * pct, 0))
                elif h == "total_samples":
                    if samples_file == "NA":
                        value = "NA"
                    else:
                        pct = int(samples_file.split("-")[1].split("pct")[0]) * 0.01
                        value = int(round(ss_samples[domain] * pct, 0))
                elif h == "statespace_size":
                    value = ss_samples[domain]

                elif h == "domain":
                    value = domain
                elif h in train_args:
                    value = train_args[h]
                    if h == "problem" and value == domain.split("_unitcost")[0]:
                        value += "_adapted"
                elif h == "fold":
                    value = fold
                elif h == "val_loss" or h == "train_loss":
                    if not train_log:
                        value = "NA"
                    else:
                        key = f"avg_{h}="
                        if key in train_log:
                            value = train_log.split(key)[-1].split(" ")[0]
                elif h == "training_time":
                    if not train_log:
                        value = "NA"
                    else:
                        value = round(float(train_log.split("Elapsed time: ")[-1].split(" ")[0].replace("s", "")), 2)
                elif h == "epochs":
                    if not train_log:
                        value = "NA"
                    else:
                        value = f"{1+int(train_log.split('Epoch ')[-1].split(' ')[0])}"
                elif h == "max_epochs":
                    value = f"{train_args['max_epochs']}" if "max_epochs" in train_args else "NA"
                elif h == "training_time":
                    value = round(float(train_log.split("Elapsed time: ")[1].split("s")[0]), 4)

                # test columns
                elif h == "coverage":
                    # plans_found = test_results["statistics"][fold]["plans_found"]
                    # total_problems = test_results["statistics"][fold]["total_problems"]
                    value = round(test_results["statistics"][fold]["coverage"], 2)
                    # value = f"{plans_found}/{total_problems} ({coverage}%)"
                elif h in test_results["configuration"]:
                    value = test_results["configuration"][h]
                elif h in test_results["statistics"][fold]:
                    value = test_results["statistics"][fold][h]

                # sampling columns
                #elif h == "sampling_technique":
                #    value = sample_file.split("_")[0]
                #elif h == "sampling_algorithm":
                #    value = sample_file.split("_")[3]
                #    if value in sampling_algorithm:
                #        value = sampling_algorithm[value]
                #    elif value in ["astar", "eager-greedy"]:
                #        value += f"({sample_file.split('_')[4]})"
                elif h == "state_representation":
                    value = sample_file.split("_")[4]
                elif h == "sampling_amount":
                    value = sample_file.split("_")[5]
                elif h == "sample_seed":
                    value = sample_file.split("ss")[1] if samples_file != "NA" else "NA"
                elif h == "statespace_rmse":
                    value = eval_rmse
                elif h == "statespace_diff":
                    value = eval_diff
              
                else:
                    if h == "domain":
                        value = domain
                    elif h == "problem":
                        value = train_folder.split("/")[-1].split(".")[1].split("_")[2]
                        if value == domain.split("_unitcost")[0]:
                            value += "_adapted"
                    elif h == "seed":
                        value = "NA"
                    elif h == "bound_multiplier":
                        if "1-25" in train_folder:
                            value = 1.25
                        elif "1-50" in train_folder:
                            value = 1.50
                        elif "2-00" in train_folder:
                            value = 2.00
                        elif "2-50" in train_folder:
                            value = 2.50
                        elif "3-00" in train_folder:
                            value = 3.00
                        else:
                            value = 1.0
                    else:
                        raise Exception(f"Column \"{h}\" unexpected.")

                if h == "domain":
                     value = value.replace("-opt14-strips", "")


            except Exception as e:
                print(train_folder)
                raise e
                value = "NA"
            line.append(str(value))

        # sp_id = HEADER.index("sample_percentage")
        line_sp_id = "1"
        if line_sp_id not in mapp:
            mapp[line_sp_id] = []
        mapp[line_sp_id].append(",".join(line))

mapp_keys = [x for x in mapp]
mapp_keys.sort()

header_when_update_column = HEADER.index("sample_seed")
col_value = None
print(HEADER_LINE)
for n in mapp_keys:
    for item in mapp[n]:
        v = item.split(",")[header_when_update_column]
        if v != col_value:
            # print(HEADER_LINE)
            col_value = v
        print(item)
# print(HEADER_LINE)
