#!/usr/bin/env python3

# Return the RMSE of the sample_files
#
# Usage: ./eval_sample.py h*_sample sample_file [sample_file sample_file ...]

from sys import argv
from sklearn.metrics import mean_squared_error as mse

hstar = {}
with open(argv[1], "r") as f:
    for h, s in [x.strip().split(";") for x in f.readlines() if x and x[0] != "#"]:
        assert s not in hstar
        hstar[s] = int(h)

for sample_file in argv[2:]:
    hstar_values, sampling_values = [], []
    with open(argv[2], "r") as f:
        for h, s in [x.strip().split(";") for x in f.readlines() if x and x[0] != "#"]:
            assert s in hstar
            hstar_values.append(hstar[s])
            sampling_values.append(int(h))
    e = mse(hstar_values, sampling_values, squared=False)
    print(sample_file, e, sep=",")
