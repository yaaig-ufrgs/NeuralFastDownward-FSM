#!/usr/bin/env python3

from sys import argv

with open(argv[1], "r") as f:
    lines = f.readlines()

states_map = {}
for line in lines:
    if line[0] == "#" or line[0] == "\n":
        continue
    if line[-1] == "\n":
        line = line[:-1]
    h, s = line.split(";", 1)
    if (s not in states_map) or (int(h) < int(states_map[s])):
        states_map[s] = h

for line in lines:
    if line[-1] == "\n":
        line = line[:-1]
    if line[0] == "#":
        print(line)
    else:
        _, s = line.split(";", 1)
        print(f"{states_map[s]};{s}")
