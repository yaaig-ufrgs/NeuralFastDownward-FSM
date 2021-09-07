#!/usr/bin/env python3

"""
Merge sets of three csv files (hnn, h*, goalcount) and make box plots.

Requirements:   matplotlib numpy pandas seaborn

Usage:          $ ./box_plot.py [hastar_csv_dir] [goalcount_csv_dir] [hnn_csv_dir]

Example:        $ ./box_plot.py ../../NeuralFastDownward-results/heuristics/state-value-pairs/hstar/ \
                  ../../NeuralFastDownward-results/heuristics/state-value-pairs/goalcount/ \
                  ../../NeuralFastDownward-results/heuristics/state-value-pairs/nn/*

Save location: experiment (not csv) directory. Some gambiarra was made but it works.

Beware: it may take some time.
"""

from sys import argv
import csv
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def merge_csv(hnn: str, hstar: str, goalcount) -> dict:
    merged_data = {}
    with open(hnn, "r") as f:
        merged_data = {}
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h_pred = row[2]
            merged_data[state] = [int(h_pred), None, None]

    with open(hstar, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h = row[1]
            if state in merged_data:
                merged_data[state][1] = int(h)

    with open(goalcount, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h = row[1]
            if state in merged_data:
                merged_data[state][2] = int(h)


    filtered_merged_data = {k: v for k, v in merged_data.items() if None not in v}

    return filtered_merged_data


def plot_scatter(data: dict, instance: str, filename: str):
    # https://wellsr.com/python/how-to-make-seaborn-boxplots-in-python/
    hnn = [data[key][0] for key in data]
    hstar = [data[key][1] for key in data]
    goalcount = [data[key][2] for key in data]

    hnn_sub_exact = [a - b for a, b in zip(hnn, hstar)]
    goalcount_sub_exact = [a - b for a, b in zip(goalcount, hstar)]

    # |   exact  |     sub     |    heuristic   |
    # |-----------------------------------------|
    # | 4        |     3-4     |    goalcount   |
    # | 2        |     5-2     |    goalcount   |
    # | ...      |     ...     |       ...      |
    # | 9        |     16-9    |       hnn      |
    # | 4        |     7-4     |       hnn      |
    # | ...      |     ...     |       ...      |

    df_sub_hnn = pd.DataFrame(list(zip(hstar, hnn_sub_exact)), columns =['h*', 'heuristic - h*']).assign(heuristic="hnn")
    df_sub_gc = pd.DataFrame(list(zip(hstar, goalcount_sub_exact)), columns =['h*', 'heuristic - h*']).assign(heuristic="goalcount")
    cdf = pd.concat([df_sub_hnn, df_sub_gc])
    #print(cdf)

    # Gambiarra below.
    filename_split = filename.split('/')
    domain = filename_split[-1].split('_')[4]
    exp_dir = filename_split[-2]
    results_dir = '/'.join(filename_split[0:3])
    out_dir = glob.glob(results_dir+"/fukunaga/regression-hl1-hu16-relu/"+domain+"/"+exp_dir)[0]
    filename = filename_split[-1]

    ax = sns.boxplot(x="h*", y="heuristic - h*", hue="heuristic", data=cdf, fliersize=2).set_title(instance)
    ax.figure.savefig(out_dir+"/"+filename, dpi=300)
    plt.clf()
    print("Saved to: "+out_dir+"/"+filename)


hstar_dir = argv[1]
if hstar_dir[-1] != "/":
    hstar_dir += "/"

goalcount_dir = argv[2]
if goalcount_dir[-1] != "/":
    goalcount_dir += "/"

for exp_dir in argv[3:]:
    if exp_dir[-1] != "/":
        exp_dir += "/"

    problem_name = '_'.join(exp_dir.split('/')[-2].split('_')[2:4])
    csv_hnn = glob.glob(exp_dir + "heuristic_pred.csv")
    csv_hstar = glob.glob(hstar_dir + problem_name + ".csv")
    csv_goalcount = glob.glob(goalcount_dir + problem_name + ".csv")

    # Ignore instances that don't have all the statistics.
    if len(csv_hstar) == 0 or len(csv_hnn) == 0 or len(csv_goalcount) == 0:
        continue

    merged = merge_csv(csv_hnn[0], csv_hstar[0], csv_goalcount[0])

    if len(merged) == 0:
        continue

    plot_filename = "box_hstar_hnn_gc_" + problem_name
    plot_scatter(merged, problem_name, exp_dir+plot_filename)

print("Done!")
