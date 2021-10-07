#!/usr/bin/env python3

"""
Creates and moves the trained models to an appropriate folder.

Usage: ./organize_rsl_experiments <directory>
"""

import sys
import os
from glob import glob


def create_train_dir(data):
    for d in data:
        d_split = d.split('/')
        directory = '/'.join(d_split[:-1])
        file = d_split[-1]
        problem = file.split('-')
        #print(problem[0])
        if problem[0] == 'blocks':
            problem_name = "blocks_probBLOCKS-" + '-'.join(problem[2:])[:-3]

        else:
            problem_name = file[:-3]

        technique = d_split[-2].split('_')[-1][2:]
        num_states = d_split[-2].split('_')[1][2:]

        dir_name = f"nfd_train.rsl_{problem_name}_{technique}_{num_states}_ss1337.ns0"
        print(dir_name)
        os.mkdir(directory+'/'+dir_name)
        os.mkdir(directory+'/'+dir_name+'/models')
        os.rename(d, directory+'/'+dir_name+'/models/traced_best_val_loss.pt')

rsl_dir = sys.argv[1]
if rsl_dir[-1] != '/':
        rsl_dir += '/'

blocks = glob(rsl_dir+"*blocks*.pt")
visitall = glob(rsl_dir+"*visitall*.pt")
transport = glob(rsl_dir+"*transport*.pt")
storage = glob(rsl_dir+"*storage*.pt")
scanalyzer = glob(rsl_dir+"*scanalyzer*.pt")
rovers = glob(rsl_dir+"*rovers*.pt")
pipesworld = glob(rsl_dir+"*pipesworld*.pt")
npuzzle = glob(rsl_dir+"*npuzzle*.pt")
grid = glob(rsl_dir+"*grid*.pt")
depot = glob(rsl_dir+"*depot*.pt")

create_train_dir(blocks)
create_train_dir(visitall)
create_train_dir(transport)
create_train_dir(storage)
create_train_dir(scanalyzer)
create_train_dir(rovers)
create_train_dir(pipesworld)
create_train_dir(npuzzle)
create_train_dir(grid)
create_train_dir(depot)
