#!/usr/bin/env python3

"""
Usage: ./big_data.py <diverse_net_folder> <hstar.csv> > file.csv

E.g.:  ./big_data.py blocks_diverse_networks hstar_fukunaga_blocks_probBLOCKS-7-0_dfs_fs_500x200_ss1.csv > big_data.csv

It's a lot of data, so it may take a couple of minutes to process them all.
"""

import sys
import pandas as pd
import json
from glob import glob


net_dir = sys.argv[1]
if net_dir[-1] != '/':
    net_dir += '/'

hstar_dict = pd.read_csv(sys.argv[2], header=None, index_col=0, squeeze=True, skiprows=[0]).to_dict()

exp_dirs = glob(net_dir+"*")

print("instance,hnn,hsample,hstar,type,sample_size,state_type,output_type,activation,hidden_layers,hidden_units,batch_size,learning_rate,sample_seed,net_seed,state")

for exp_dir in exp_dirs:
    dir_split = exp_dir.split('/')[1].split('_')
    #print(dir_split)
    # Could also open the train_args.json file to take all the info, but in this case
    # the filename has all we need.
    leftover = dir_split[-1].split('.')
    instance = dir_split[2] + "_" + dir_split[3]
    sample_type = dir_split[4]
    state_type = dir_split[5]
    sample_size = dir_split[6]
    sample_seed = int(dir_split[7][2:])
    output_type = dir_split[8]
    activation = dir_split[9]
    hidden_layers = int(dir_split[10][2:])
    hidden_units = int(dir_split[11][2:])
    batch_size = int(dir_split[12][1:])
    learning_rate = float(leftover[0][2:].replace('-', '.'))
    net_seed = leftover[1][2:]

    h_pred = pd.read_csv(exp_dir+"/heuristic_pred.csv", header=None, skiprows=[0], index_col=0)
    #print(h_pred)
    for row in h_pred.itertuples():
        state = row[0]
        hsample = int(row[1])
        hnn = int(row[2])
        # hstar is "" (null) if there's no h* data for a given state (e.g. blocks 12)
        hstar = int(hstar_dict[state]) if state in hstar_dict else ""
        print(f"{instance},{hnn},{hsample},{hstar},{sample_type},{sample_size},{state_type},{output_type},{activation},{hidden_layers},{hidden_units},{batch_size},{learning_rate},{sample_seed},{net_seed},{state}")
