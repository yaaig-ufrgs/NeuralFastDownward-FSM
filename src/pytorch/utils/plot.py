import matplotlib.pyplot as plt
import imageio
import glob
import logging
import csv
import pandas as pd
import seaborn as sns
import numpy as np
from os import path, makedirs, remove

_log = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL').setLevel(logging.WARNING)

def save_y_pred_scatter(data: dict, t: int, fold_idx: int, directory: str):
    if t == -1:
        t = "final"

    if not path.exists(directory):
        makedirs(directory)

    dir_split = directory.split('/')[-2].split('_')
    seeds = dir_split[-1].split('.')
    seeds = seeds[0] + "_" + seeds[1]
    plot_title = '_'.join(dir_split[2:-1]) + "_" + seeds
    plot_filename = f"{plot_title}_epoch_{str(t)}_{fold_idx}" 

    real = [data[key][0] for key in data]
    pred = [data[key][1] for key in data]

    fig, ax = plt.subplots()
    ax.scatter(real, pred, s=2, alpha=0.35, c="red", zorder=10)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, 'k-', alpha=0.80, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("h^sample")
    ax.set_ylabel("h^NN")
    epoch = "\nepoch " + str(t)
    ax.set_title(plot_title+epoch, fontsize=10)

    fig.savefig(directory+"/"+plot_filename)

    plt.clf()
    plt.close(fig)


def save_h_pred_scatter(directory: str, csv_hnn: str, csv_h: str) -> dict:
    """
    Creates a scatter plot with hnn and some other heuristic (if data for it is available).
    """
    merged_data = {}
    with open(csv_hnn, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h_pred = row[2]
            merged_data[state] = [int(h_pred), None]

    with open(csv_h, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h = row[1]
            if state in merged_data:
                merged_data[state][1] = int(h)

    data = {k: v for k, v in merged_data.items() if None not in v}

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

    dir_split = directory.split('/')[-2].split('_')
    seeds = dir_split[-1].split('.')
    seeds = seeds[0] + "_" + seeds[1]
    plot_name = '_'.join(dir_split[2:-1]) + "_" + seeds

    ax.set_title(plot_name, fontsize=10)

    compared_heuristic = csv_h.split('/')[-2]
    if compared_heuristic == "hstar":
        compared_heuristic = "h*"

    ax.set_xlabel(compared_heuristic)
    ax.set_ylabel("h^NN")

    plot_filename = "hnn_" + compared_heuristic + "_" + plot_name
    fig.savefig(directory+"/"+plot_filename)
    plt.clf()
    plt.close(fig)

    return data


def save_box_plot(directory: str, data: dict, csv_h: str):
    """
    Creates a box plot with hnn, h* and goalcount (when possible, i.e. all the data is available).
    """
    with open(csv_h, "r") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            state = row[0]
            h = row[1]
            if state in data:
                #merged_data[state][2] = int(h)
                data[state].append(int(h))

    hnn = [data[key][0] for key in data]
    goalcount = [data[key][1] for key in data]
    hstar = [data[key][2] for key in data]

    hnn_sub_exact = [a - b for a, b in zip(hnn, hstar)]
    goalcount_sub_exact = [a - b for a, b in zip(goalcount, hstar)]

    df_sub_hnn = pd.DataFrame(list(zip(hstar, hnn_sub_exact)), columns =['h*', 'heuristic - h*']).assign(heuristic="hnn")
    df_sub_gc = pd.DataFrame(list(zip(hstar, goalcount_sub_exact)), columns =['h*', 'heuristic - h*']).assign(heuristic="goalcount")
    cdf = pd.concat([df_sub_hnn, df_sub_gc])

    dir_split = directory.split('/')[-2].split('_')
    seeds = dir_split[-1].split('.')
    seeds = seeds[0] + "_" + seeds[1]
    plot_name = '_'.join(dir_split[2:-1]) + "_" + seeds
    plot_filename = "box_" + plot_name

    ax = sns.boxplot(x="h*", y="heuristic - h*", hue="heuristic", data=cdf, fliersize=2).set_title(plot_name)
    ax.figure.savefig(directory+"/"+plot_filename)
    plt.clf()
    plt.close(ax.figure)


def save_gif_from_plots(directory: str, fold_idx: int):
    """
    Creates a scatter plot gif showing the evolution of the hnn in comparison
    to the sample heuristic.
    Only works if you have the arg -spn set up during training.
    """

    dir_split = directory.split('/')[-2].split('_')
    seeds = dir_split[-1].split('.')
    seeds = seeds[0] + "_" + seeds[1]
    gif_filename = '_'.join(dir_split[2:-1]) + "_" + seeds
    plot_files = sorted(glob.glob(f"{directory}/*{fold_idx}.png"), key=path.getmtime)

    with imageio.get_writer(f'{directory}/{gif_filename}.gif', mode='I') as writer:
        for f in plot_files:
            image = imageio.imread(f)
            writer.append_data(image)


def remove_intermediate_plots(plots_dir: str, fold_idx: int):
   """
   Removes the intermediate plots used to create the plot gif.
   """
   if path.exists(plots_dir):
       plots = glob.glob(plots_dir+"/*.png")
       for plot in plots:
           plot_split = plot.split('_')
           idx = int(plot_split[-1].split('.')[0])
           if idx == fold_idx and plot_split[-2] == "final":
               continue
           remove(plot)
