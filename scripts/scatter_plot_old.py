#!/usr/bin/env python3

"""
Merge two csv files (hnn and another heuristic), merge them,
and make a scatter plot.

Usage: $ ./scatter_plot.py plot_filename compared_heuristic [experiment_dirs]

Example: $ ./scatter_plot.py someplotname h\* results/nfd_train.fukunaga_probBLOCKS-12-0_dfs_fs_500x200_ss2.ns2

Beware: make sure you have only one nfd_test.N with a problem_instance.csv inside it.

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


def plot_scatter(data: dict, compared_heuristic, filename: str):
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
    ax.set_title("h^nn vs. " + compared_heuristic)

    ax.set_xlabel(compared_heuristic)
    ax.set_ylabel("h^NN")

    fig.savefig(filename, dpi=300)


plot_filename = argv[1]
compared_heuristic = argv[2]
for exp_dir in argv[3:]:
    if exp_dir[-1] != "/":
        exp_dir += "/"
    csv_hnn = glob.glob(exp_dir + "heuristic_pred.csv")
    data = {} # {state: (h1, h2)}
    csv_h = glob.glob(exp_dir+"tests/*/*.csv")
    if len(csv_h) == 0 or len(csv_hnn) == 0:
        continue
    merged = merge_csv(csv_hnn[0], csv_h[0])
    plot_scatter(merged, compared_heuristic, exp_dir+plot_filename)
    print("OK")
