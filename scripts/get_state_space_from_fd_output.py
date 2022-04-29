#!/usr/bin/env python3

"""
usage: ./get_state_space_from_fd_output fd.output
"""

from sys import argv

with open(argv[1]) as f:
    samples = [x.strip().split(" ", 1)[1] for x in f.readlines() if x.startswith("[S] ")]

# order by h
d = {}
for s in samples:
    h = s.split(";")[0]
    if h not in d:
        d[h] = []
    d[h].append(s)
for h in range(1000):
    h = str(h)
    if h in d:
        for s in d[h]:
            print(s)
