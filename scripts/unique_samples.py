#!/usr/bin/env python3

"""
Get number of unique samples in each arg file.

Usage: ./unique_samples.py samples_file [samples_file ...]
  e.g. ./unique_samples.py ../../samples/yaaig/blocks/yaaig_probBLOCKS-12-0_*
"""

from sys import argv
from numpy import array, unique

for file in argv[1:]:
    with open(file, "r") as f:
        samples = [x.strip().split(";")[1] for x in f.readlines() if x and x[0] != "#"]
        u, t = len(unique(array(samples))), len(samples)
        print(f"{file},{u},{t} ({(100*u/t):.2f}%)")
