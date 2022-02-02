#!/usr/bin/env python3

"""
For each partial sample in `partial_state_file`, find the best (by h-value)
complete sample in `full_state_file`.

Usage: ./partial_to_full_state.py partial_state_file full_state_file
"""

from sys import argv
from random import choice

def read_pairs(file_path: str) -> [(str, str)]:
    with open(file_path, "r") as file:
        return [l.strip().split(";") for l in file.readlines() if l and l[0] != '#']

def is_substate(partial: str, complete: str) -> bool:
    partial_idxs = [i for i, b in enumerate(partial) if b == "1"]
    for idx in partial_idxs:
        if complete[idx] == "0":
            return False
    return True

ps_pairs = read_pairs(argv[1])
fs_pairs = read_pairs(argv[2])
for _, ps_sample in ps_pairs:
    best = [None, []]
    for fs_h, fs_sample in fs_pairs:
        fs_h = int(fs_h)
        if is_substate(ps_sample, fs_sample):
            if best[0] == None or best[0] < fs_h:
                best = [fs_h, [fs_sample]]
            elif fs_h == best[0]:
                best[1].append(fs_sample)
    print(f"{best[0]};{choice(best[1])}")
