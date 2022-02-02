#!/usr/bin/env python3

"""
Get number of unique samples in each arg file.

Usage: ./unique_samples.py samples_file [samples_file ...]
  e.g. ./unique_samples.py ../../samples/yaaig/blocks/yaaig_probBLOCKS-12-0_*_fullstate_30K_seed*
"""

from sys import argv

for file in argv[1:]:
    with open(file,"r") as f:
        lines = f.readlines()[1:]
    uniques = []
    total = 0
    for line in lines:
        if line[0] != "#":
            l = line[:-1].split(";", 1)[1]
            if l not in uniques:
                uniques.append(l)
            total += 1
    uniques = len(uniques)
    print(file.split("/")[-1], end=",")
    print(f"{uniques}/{total} ({(100*uniques/total):.2f}%)")
