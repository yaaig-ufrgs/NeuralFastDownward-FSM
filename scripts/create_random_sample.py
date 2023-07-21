#!/usr/bin/env python3

"""
This script creates a random % of samples from the full statespace of an instance.

Usage:   $ ./create_random_sample.py <max_seed> <sample_file>
Example: $ ./create_random_sample.py 9 statespace_blocks_probBLOCKS-7-0_hstar
"""

from math import ceil
from sys import argv
import os
import random


def random_sample_statespace(
    statespace: str,
    directory: str,
    min_seed: int = 0,
    max_seed: int = 9,
    pcts: [float] = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],
):
    max_seed += 1

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(statespace, "r") as f:
        samples = [s.strip() for s in f.readlines() if s and s[0] != "#"]

    sample_name = statespace.split("/")[-1]

    print("> Creating random samples from the statespace:")
    for pct in pcts:
        for seed in range(min_seed, max_seed):
            curr_samples = samples.copy()
            random.seed(seed)
            random.shuffle(curr_samples)

            n = ceil(len(curr_samples) * pct)
            percentage = int(pct * 100)

            filename = f"{directory}/{sample_name}_{percentage}pct_ss{seed}"
            print(f"  > Saving {filename}...")
            with open(filename, "w") as f:
                for s in curr_samples[:n]:
                    f.write(s + "\n")


if __name__ == "__main__":
    random_sample_statespace(int(argv[1]), argv[2])
