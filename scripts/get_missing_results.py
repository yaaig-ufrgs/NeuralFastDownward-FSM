#!/usr/bin/env python3

# Checks if all tests ran successfully.
# Usage: ./get_missing_results.py samples_folder results_folder

from sys import argv
import os

ns = list(range(10))

results_folder = argv[2]
samples = os.listdir(argv[1])
results = [f"{results_folder}/{f}" for f in os.listdir(results_folder)]

# remove incomplete tests
for result in results:
    if not os.path.exists(f"{result}/tests/nfd_test/test_results.json"):
        os.rmdir(result)

for sample in samples:
    for s in ns:
        dir = f"nfd_train.{sample}.ns{s}"
        if not os.path.exists(f"{results_folder}/{dir}"):
            print(dir)
