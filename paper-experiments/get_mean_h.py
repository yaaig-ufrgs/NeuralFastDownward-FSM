#!/usr/bin/env python3

"""
E.g.: ./get_mean_h.py samples-directory

"""

import os
import pandas as pd
from sys import argv
from glob import glob

fd_root = os.path.abspath(__file__).split("NeuralFastDownward")[0] + "NeuralFastDownward"
domains_d = {
    "blocks": f"{fd_root}/tasks/experiments/statespaces/statespace_blocks_probBLOCKS-7-0_hstar",
    "grid": f"{fd_root}/tasks/experiments/statespaces/statespace_grid_grid_hstar",
    "npuzzle": f"{fd_root}/tasks/experiments/statespaces/statespace_npuzzle_prob-n3-1_hstar",
    "rovers": f"{fd_root}/tasks/experiments/statespaces/statespace_rovers_rovers_hstar",
    "scanalyzerunit": f"{fd_root}/tasks/experiments/statespaces/statespace_scanalyzerunit_scanalyzer_hstar",
    "transportunit": f"{fd_root}/tasks/experiments/statespaces/statespace_transportunit_transport_hstar",
    "visitall": f"{fd_root}/tasks/experiments/statespaces/statespace_visitall_p-1-4_hstar"
}


domains = ['blocks', 'grid', 'npuzzle', 'rovers', 'scanalyzerunit', 'transportunit', 'visitall']

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

        h_algo.append(mean_h)
        print(f"{domain},{sample.split('/')[-1]},{len(csv_files)},{round(mean_h,2)}")
