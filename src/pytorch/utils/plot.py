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

    if not path.exists(directory):
        makedirs(directory)

    #dir_split = directory.split('/')[-1].split('_')
    dir_split = directory.split('/')[-2].split('_')
    seeds = dir_split[-1].replace('.', '_')[0:7]
    plot_title = '_'.join(dir_split[2:-1]) + "_" + seeds
    plot_filename = plot_title + "_epoch_" + str(t)

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

def save_box_plot(directory: str):
    """
    Creates a box plot with hnn, h* and goalcount (when possible, i.e. all the data is available).
    """
    pass


def save_h_pred_scatter(data: dict, directory: str):
    """
    Creates a scatter plot with hnn and some other heuristic (if data for it is available).
    """
    pass

def save_gif_from_plots(directory: str):
    """
    Creates a scatter plot gif showing the evolution of the hnn in comparison
    to the sample heuristic.
    Only works if you have the arg -spn set up during training.
    """

    dir_split = directory.split('/')[-2].split('_')
    seeds = dir_split[-1].replace('.', '_')[0:7]
    gif_filename = '_'.join(dir_split[2:-1]) + "_" + seeds
    plot_files = sorted(glob.glob(directory+"/*"), key=path.getmtime)

    with imageio.get_writer(f'{directory}/{gif_filename}.gif', mode='I') as writer:
        for f in plot_files:
            image = imageio.imread(f)
            writer.append_data(image)

def remove_intermediate_plots(plots_dir: str):
   """
   Removes the intermediate plots used to create the plot gif.
   """
   if path.exists(plots_dir):
       intermediate_plots = glob.glob(plots_dir+"/*_epoch_*[0-9]*.png")
       for plot in intermediate_plots:
           remove(plot)
