#!/usr/bin/env python3

"""
Save a csv file with the format state,y (state, heuristic value) from 
downward log files.

Use: $ ./print_state_h.py [log_files]
"""

from sys import argv

def save_csv(data: dict, filename: str):
    with open(filename, "w") as f:
        f.write("state,y\n")
        for key, value in data.items():
            f.write("%s,%s\n" % (key, value))

for log_file in argv[1:]:
    state_values = {}
    log_split = log_file.split('/')
    out_dir = '/'.join(log_split[:-2])
    filename = out_dir + "/" + log_split[-1].replace('log','csv')
    print(filename)
    with open(log_file, "r") as f:
        for ln in f:
            if ln.startswith("[S] "):
                l = ln.split()
                h = l[1]
                s = l[2]
                state_values[s] = h

    save_csv(state_values, filename)
