#!/usr/bin/env python3

"""
Compiles multiple result folders into a single table.
Choose table columns in variable `HEADER`.

Usage: ./compile_results.py train_folder [train_folder ...]
  e.g. ./compile_results.py ../../NeuralFastDownward/results/nfd_train.*
"""

from sys import argv
from json import load
import csv


#####################################################################
# ADD HERE ALL ARGS (AND ORDER) THAT SHOULD BE ADDED TO THE TABLE
HEADER = [
    "domain", "problem", "test_algorithm", "test_heuristic", "statespace_size",
    "sampling_algorithm", "sai", "sui", "hstar", "mutex", "perfect_mutex", "total_samples", "bfs_samples", "rw_samples", "random_samples",
    "rw_bound_method", "rw_bound", "sample_seed", "network_seed", "epochs", "max_epochs",
    "train_loss", "val_loss", "training_time", "plans_found", "total_problems", "coverage", "avg_plan_length",
    "avg_plan_cost", "avg_initial_h", "avg_expanded", "avg_evaluated", "avg_generated", "avg_search_time",
    "total_accumulated_time", "statespace_diff",
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
    # default, facts, factseff, farthest state from the goal
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
                eval_diff = row['mean_loss']
                break
    except Exception as _:
        eval_diff = "NA"
    try:
        with open(f"{train_folder}/nfd.log") as train_log_file:
            train_log = train_log_file.read().replace("\n", " ")
    except Exception as _:
        train_log = None
    try:
        is_untested = False
        with open(f"{test_folder}/test_results.json") as test_results_file:
            test_results = load(test_results_file)
    except:
        is_untested = True
        test_results = {"statistics" : ["None"]}

    for fold in test_results["statistics"]:
        line = []

        domain = train_args["domain"] if "domain" in train_args else train_folder.split("/")[-1].split(".")[1].split("_")[1]
        domain = domain.replace("-opt14-strips", "")
        samples_file = train_args["samples"].split("/")[-1] if "samples" in train_args else "NA"

        sampling_algorithm = "NA"
        if "bfsrw" in samples_file:
            sampling_algorithm = "bfs_rw"
        elif "bfs" in samples_file:
            sampling_algorithm = "bfs"
        elif "dfs" in samples_file:
            sampling_algorithm = "dfs"
        elif "rw" in samples_file:
            sampling_algorithm = "rw"

        rw_bound_method = "NA"
        rw_bound = "NA"
        if "_bnd-" in samples_file:
            rw_bound_method = samples_file[samples_file.find("_bnd-"):].split('-')[1].split('_')[0]
            if rw_bound_method == "200":
                rw_bound_method = "default"
            if rw_bound_method == "default":
                rw_bound = bounds[domain][0]
            elif rw_bound_method == "facts":
                rw_bound = bounds[domain][1]
            elif rw_bound_method == "factseff":
                rw_bound = bounds[domain][2]

        if "unit-" in train_folder:
            domain = domain + "_unitcost"

        if domain == "scanalyzerunit" or domain == "scanalyzerunit_unitcost" :
            domain = "scanalyzer_unitcost"
        elif domain == "transportunit":
            domain = "transport_unitcost"

        random_samples = 0
        if "_rs-" in samples_file:
            random_samples = samples_file.split('-')[-1].split('_')[0]

        total_samples = 0
        if "_maxs-" in samples_file:
            total_samples = samples_file[samples_file.find("_maxs-"):].split('-')[1].split('_')[0]

        sui = "false"
        if "_sui-1" in samples_file:
            sui = "true"

        sai = "false"
        if "_sai-both" in samples_file:
            sai = "true"

        hstar = "false"
        if "-hstar" in train_folder:
            hstar = "true"

        mutex = "false"
        if "nomutex" not in train_folder:
            mutex = "true"
            
        perfect_mutex = "false"
        if "forward_statespace" in train_folder:
            perfect_mutex = "true"

        domain = domain.replace("-opt14-strips", "")

        for h in HEADER:
            value = "NA"
            try:
                if h == "network_seed":
                    h = "seed"
                if h == "test_heuristic":
                    h = "heuristic"
                if h == "test_algorithm":
                    h = "search_algorithm"

                # training columns
                if h in train_args:
                    value = train_args[h]
                    if h == "problem" and value == domain.split("_unitcost")[0]:
                        value += "_adapted"
                elif h == "domain":
                    value = domain
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
                if not is_untested: 
                    if h == "coverage":
                        # plans_found = test_results["statistics"][fold]["plans_found"]
                        # total_problems = test_results["statistics"][fold]["total_problems"]
                        value = round(test_results["statistics"][fold]["coverage"], 2)
                        # value = f"{plans_found}/{total_problems} ({coverage}%)"
                    elif h in test_results["configuration"]:
                        value = test_results["configuration"][h]
                    elif h in test_results["statistics"][fold]:
                        value = test_results["statistics"][fold][h]

                # sampling columns
                if h == "state_representation":
                    value = sample_file.split("_")[4]
                elif h == "sample_seed":
                    value = sample_file.split("ss")[1] if samples_file != "NA" else "NA"
                elif h == "statespace_diff":
                    value = eval_diff
                else:
                    if h == "domain":
                        value = domain
                    elif h == "problem":
                        value = train_folder.split("/")[-1].split(".")[1].split("_")[2]
                        if value == domain.split("_unitcost")[0]:
                            value += "_adapted"
                    if h == "sui":
                        value = sui
                    if h == "sai":
                        value = sai
                    if h == "hstar":
                        value = hstar
                    if h == "mutex":
                        value = mutex
                    if h == "perfect_mutex":
                        value = perfect_mutex

                    elif h == "sampling_algorithm":
                        if samples_file != "NA":
                            value = sampling_algorithm
                            if value in sampling_algorithma:
                                value = sampling_algorithma[value]

                    elif h == "rw_bound_method":
                        value = rw_bound_method
                    elif h == "rw_bound":
                        value = rw_bound

                    elif h == "bfs_samples":
                        if samples_file != "NA":
                            if "bfs" in sampling_algorithm:
                                value = round(int(total_samples) * 0.1)
                            else:
                                value = 0
                    elif h == "rw_samples":
                        if samples_file != "NA":
                            if "bfs" in sampling_algorithm:
                                value = int(total_samples) - (round(int(total_samples) * 0.1)) - int(random_samples)
                            else:
                                value = int(total_samples) - int(random_samples)
                    elif h == "total_samples":
                        if samples_file != "NA":
                            value = total_samples
                    elif h == "statespace_size":
                        value = ss_samples[domain]
                    elif h == "random_samples":
                        value = random_samples

                if h == "domain":
                     value = value.replace("-opt14-strips", "")

            except Exception as e:
                print(train_folder)
                raise e
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
