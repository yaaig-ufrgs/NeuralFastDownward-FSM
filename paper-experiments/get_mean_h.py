#!/usr/bin/env python3

"""
E.g.: ./get_mean_h.py samples-directory

"""

import pandas as pd
from sys import argv
from glob import glob
import numpy as np
from statistics import geometric_mean

domains_d = {'blocks': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_blocks_probBLOCKS-7-0_hstar",
             'grid': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_grid_grid_hstar",
             'npuzzle': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_npuzzle_prob-n3-1_hstar",
             'rovers': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_rovers_rovers_hstar",
             'scanalyzer': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_scanalyzer_scanalyzer_hstar",
             'scanalyzerunit': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_scanalyzerunit_scanalyzer_hstar",
             'transport': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_transport_transport_hstar",
             'transportunit': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_transportunit_transport_hstar",
             'visitall': "../../NeuralFastDownward/tasks/experiments/statespaces/statespace_visitall_p-1-4_hstar",
             }

#domains = ['blocks', 'grid', 'npuzzle', 'rovers', 'scanalyzer', 'scanalyzerunit', 'transport', 'transportunit', 'visitall']
domains = ['blocks', 'grid', 'npuzzle', 'rovers', 'scanalyzerunit', 'transportunit', 'visitall']

#h_ss = []
#h_bfs = []
#h_dfs = []
#h_rw = []
#h_bfsrw = []

print("domain,sample,n_seeds,mean_h")
for domain in domains:
    samples_domain = glob(f"{argv[1]}/*{domain}-*")
    samples = []
    for s in samples_domain:
        samples.append(s)

    df_statespace = pd.read_csv(domains_d[domain], header=None, sep=';', comment='#', usecols=[0,1], names=['h', 'state'])
    max_df_statespace = max(df_statespace['h'])
    mean_h_fssp = df_statespace['h'].mean()
    #h_ss.append(mean_h_fssp)
    print(f"{domain},statespace,NA,{round(mean_h_fssp,2)}")

    h_algo = []
    for sample in samples:
        #print(sample)
        if sample == None:
            continue
        csv_files = glob(f"{sample}/*_ss[0-9]")
        df_curr = None
        appended_data = []
        for csv_file in csv_files:
            if "rmse" in csv_file:
                continue
            df = pd.read_csv(csv_file, header=None, sep=';', comment='#', usecols=[0,1], names=['h', 'state'])
            appended_data.append(df)
        df_curr = pd.concat(appended_data, ignore_index=True)

        assert df_curr is not None

        mean_h = df_curr['h'].mean()

        #if "-bfs-" in sample:
        #    h_bfs.append(mean_h)
        #if "-dfs-" in sample:
        #    h_dfs.append(mean_h)
        #if "-rw-" in sample:
        #    h_rw.append(mean_h)
        #if "-bfsrw-" in sample:
        #    h_bfsrw.append(mean_h)

        h_algo.append(mean_h)
        print(f"{domain},{sample.split('/')[-1]},{len(csv_files)},{round(mean_h,2)}")
