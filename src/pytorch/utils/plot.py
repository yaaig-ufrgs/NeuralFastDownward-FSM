import matplotlib.pyplot as plt
import imageio
import glob
import logging
import numpy as np
from argparse import Namespace
from os import path, makedirs, remove

_log = logging.getLogger(__name__)
logging.getLogger("matplotlib.font_manager").disabled = True
logging.getLogger("PIL").setLevel(logging.WARNING)


def get_plot_title(directory: str) -> str:
    """
    Return plot title from directory name.
    """
    dir_split = directory.split("/")[-2].split("_")
    plot_title = "_".join(dir_split[2:])
    return plot_title


def save_y_pred_scatter(data: list, t: int, fold_idx: int, directory: str, prefix: str):
    """
    Create and save real y and predicted y scatter plot.
    """
    if data is None:
        return

    if t == -1:
        t = "final"

    if not path.exists(directory):
        makedirs(directory)

    plot_title = get_plot_title(directory)
    plot_filename = f"{prefix}{plot_title}_epoch_{str(t)}_{fold_idx}"

    real = [d[1] for d in data]
    pred = [d[2] for d in data]

    fig, ax = plt.subplots()
    ax.scatter(real, pred, s=2, alpha=0.35, c="red", zorder=10)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    ax.plot(lims, lims, "k-", alpha=0.80, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("h^sample")
    ax.set_ylabel("h^NN")
    epoch = "\nepoch " + str(t)
    hl = round(len(plot_title) / 2)
    ax.set_title(prefix + plot_title[:hl] + "\n" + plot_title[hl:] + epoch, fontsize=9)

    fig.savefig(directory + "/" + plot_filename + ".png")

    plt.clf()
    plt.close(fig)


def save_pred_error_bar_eval(data: list, directory: str, data_type: str):
    """
    Create and save error count histogram plot for eval.
    """
    if len(data) == 0:
        return

    if not path.exists(directory):
        makedirs(directory)

    plot_title = get_plot_title(directory)
    plot_filename = f"eval_error_{data_type}_{plot_title}"

    rounded_errors = [round(d[4]) for d in data]

    d_error_count = {}

    for e in rounded_errors:
        d_error_count[e] = d_error_count.get(e, 0) + 1

    fig, ax = plt.subplots()
    x_vals = list(d_error_count.keys())
    y_vals = list(d_error_count.values())
    low_y, high_y = min(y_vals), max(y_vals)

    ax.bar(x_vals, y_vals, width=0.7, align='center', edgecolor='black', linewidth=0.5)
    #ax.set_xlim(low_x-5, high_x+5)
    ax.set_ylim(low_y, high_y+5)

    ax.set_xlabel("y-pred")
    ax.set_ylabel("count")
    hl = round(len(plot_title) / 2)
    ax.set_title(data_type + "_" + plot_title[:hl] + "\n" + plot_title[hl:], fontsize=9)
    ax.text(0.70, 0.90, f'max_error = {max(rounded_errors)}\navg_error = {round(sum(rounded_errors) / len(rounded_errors), 2)}\nmax_count = {max(d_error_count, key=d_error_count.get)}',
            horizontalalignment='left',
            verticalalignment='center',
            transform = ax.transAxes)
    fig.savefig(directory + "/" + plot_filename + ".png")

    plt.clf()
    plt.close(fig)


def save_y_pred_scatter_eval(data: list, directory: str, data_type: str):
    """
    Create and save real y and predicted y scatter plot for eval.
    """
    if len(data) == 0:
        return

    if not path.exists(directory):
        makedirs(directory)

    plot_title = get_plot_title(directory)
    plot_filename = f"eval_best_{data_type}_{plot_title}"

    real = [round(d[2]) for d in data]
    pred = [round(d[3]) for d in data]

    fig, ax = plt.subplots()
    ax.scatter(real, pred, s=2, alpha=0.35, c="red", zorder=10)

    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]

    ax.plot(lims, lims, "k-", alpha=0.80, zorder=0)
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("h^sample")
    ax.set_ylabel("h^NN")
    hl = round(len(plot_title) / 2)
    ax.set_title(data_type + "_" + plot_title[:hl] + "\n" + plot_title[hl:], fontsize=9)

    fig.savefig(directory + "/" + plot_filename + ".png")

    plt.clf()
    plt.close(fig)


def save_gif_from_plots(directory: str, fold_idx: int):
    """
    Creates a scatter plot gif showing the evolution of the hnn in comparison to the sample heuristic.
    Only works if you have the arg -spn set up during training.
    """
    gif_filename = get_plot_title(directory)
    train_plot_files = sorted(
        glob.glob(f"{directory}/train_*{fold_idx}.png"), key=path.getmtime
    )
    val_plot_files = sorted(
        glob.glob(f"{directory}/val_*{fold_idx}.png"), key=path.getmtime
    )

    with imageio.get_writer(
        f"{directory}/train_{gif_filename}.gif", mode="I"
    ) as writer:
        for f in train_plot_files:
            image = imageio.imread(f)
            writer.append_data(image)

    with imageio.get_writer(f"{directory}/val_{gif_filename}.gif", mode="I") as writer:
        for f in val_plot_files:
            image = imageio.imread(f)
            writer.append_data(image)


def remove_intermediate_plots(plots_dir: str, fold_idx: int):
    """
    Removes the intermediate plots used to create the plot gif.
    """
    if path.exists(plots_dir):
        plots = glob.glob(plots_dir + "/*.png")
        for plot in plots:
            plot_split = plot.split("_")
            idx = int(plot_split[-1].split(".")[0])
            if idx == fold_idx and plot_split[-2] == "final":
                continue
            remove(plot)


def make_extra_plots(args: Namespace, dirname: str, best_fold: dict):
    """
    Manages extra plots, suchs as:
    - h vs predicted h scatter plot animation.
    """
    plots_dir = f"{dirname}/plots"

    if args.scatter_plot and args.plot_n_epochs != -1:
        try:
            _log.info(f"Saving scatter plot GIF.")
            save_gif_from_plots(plots_dir, best_fold["fold"])
            remove_intermediate_plots(plots_dir, best_fold["fold"])
        except:
            _log.error(f"Failed making plot GIF.")

    heuristic_pred_file = f"{dirname}/heuristic_pred.csv"

    if not args.save_heuristic_pred and path.exists(heuristic_pred_file):
        remove(heuristic_pred_file)
