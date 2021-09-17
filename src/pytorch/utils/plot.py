import matplotlib.pyplot as plt
import imageio
import glob
import logging
from os import path, makedirs, remove
import numpy as np

_log = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL').setLevel(logging.WARNING)

def save_y_pred_scatter(data: dict, t: int, directory: str):
    if t == -1:
        t = "final"

    plots_folder = directory+"/plots"
    if not path.exists(plots_folder):
        makedirs(plots_folder)

    dir_split = directory.split('/')[-1].split('_')
    seeds = dir_split[-1].replace('.', '_')[0:7]
    plot_filename = '_'.join(dir_split[2:-1]) + "_" + seeds + "_epoch_" + str(t)

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
    ax.set_title(plot_filename, fontsize=10)

    fig.savefig(plots_folder+"/"+plot_filename)

    plt.clf()
    plt.close(fig)

def save_gif_from_plots(directory: str):
    dir_split = directory.split('/')[-1].split('_')
    seeds = dir_split[-1].replace('.', '_')[0:7]
    gif_filename = '_'.join(dir_split[2:-1]) + "_" + seeds
    plot_files = sorted(glob.glob(directory+"/plots/*"), key=path.getmtime)

    with imageio.get_writer(f'{directory}/plots/{gif_filename}.gif', mode='I') as writer:
        for f in plot_files:
            image = imageio.imread(f)
            writer.append_data(image)
