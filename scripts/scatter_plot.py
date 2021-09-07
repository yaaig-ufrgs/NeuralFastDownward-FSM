#!/usr/bin/env python3

"""
Merge two csv files (hnn and another heuristic), merge them,
and make a scatter plot.

Usage:    $ ./scatter_plot.py [csv_dir] [hnn_csv_dir]
Example:  $ ./scatter_plot.py ../../NeuralFastDownward-results/heuristics/state-value-pairs/goalcount/ \
            ../../NeuralFastDownward-results/heuristics/state-value-pairs/nn/*

Save location: experiment (not csv) directory. Some gambiarra was made but it works.
"""

from sys import argv
import csv
import glob
import matplotlib.pyplot as plt
import numpy as np

def merge_csv(a: str, b: str) -> dict:
    merged_data = {}
    with open(a, "r") as f:
        merged_data = {}
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h_pred = row[2]
            merged_data[state] = [int(h_pred), None]

    with open(b, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h = row[1]
            if state in merged_data:
                merged_data[state][1] = int(h)

    filtered_merged_data = {k: v for k, v in merged_data.items() if None not in v}

    return filtered_merged_data


def plot_scatter(data: dict, compared_heuristic, instance: str, filename: str):
    hnn = [data[key][0] for key in data]
    h = [data[key][1] for key in data]

    fig, ax = plt.subplots()
    ax.scatter(h, hnn, s=2, alpha=0.35, c="red", zorder=10)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.80, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    #ax.set_title("h^nn vs. " + compared_heuristic)
    ax.set_title(instance)

    if compared_heuristic == "hstar":
        compared_heuristic = "h*"

    ax.set_xlabel(compared_heuristic)
    ax.set_ylabel("h^NN")

    filename_split = filename.split('/')
    domain = filename_split[-1].split('_')[3]
    exp_dir = filename_split[-2]
    results_dir = '/'.join(filename_split[0:3])
    out_dir = glob.glob(results_dir+"/fukunaga/regression-hl1-hu16-relu/"+domain+"/"+exp_dir)[0]
    filename = filename_split[-1]

    fig.savefig(out_dir+"/"+filename, dpi=300)
    plt.clf()
    plt.close(fig)
    print("Saved to: "+out_dir+"/"+filename)


csv_dir = argv[1]
if csv_dir[-1] != "/":
    csv_dir += "/"
compared_heuristic = csv_dir.split('/')[-2]

for exp_dir in argv[2:]:
    if exp_dir[-1] != "/":
        exp_dir += "/"

    problem_name = '_'.join(exp_dir.split('/')[-2].split('_')[2:4])
    csv_hnn = glob.glob(exp_dir + "heuristic_pred.csv")
    csv_h = glob.glob(csv_dir + problem_name + ".csv")

    if len(csv_h) == 0 or len(csv_hnn) == 0:
        continue

    merged = merge_csv(csv_hnn[0], csv_h[0])

    if len(merged) == 0:
        continue

    plot_filename = "hnn_vs_" + compared_heuristic + "_" + problem_name
    plot_scatter(merged, compared_heuristic, problem_name, exp_dir+plot_filename)

print("Done!")
