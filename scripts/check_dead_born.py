#!/usr/bin/env python3

"""
usage: ./check_dead_born.py weight_initialization num_seeds pddl_file [pddl_file ...]
"""

import re
import subprocess
from sys import argv
from random import choice, randint
from os import remove
from shutil import rmtree

_FD_PATH = "../fast-downward.py"
_TRAIN_PATH = "../train.py"

def get_domain_and_problem(pddl_file: str) -> (str, str):
    return re.findall(".*/(.*)/(.*).pddl", pddl_file)[0]

def sampling(pddl_file: str, num_samples: int = 1000, seed: int = 0) -> str:
    domain, problem = get_domain_and_problem(pddl_file)
    sas_file = "output.sas"
    plan_file = f"yaaig_{domain}_{problem}_rw_{num_samples}_ss{seed}"
    command = [
        _FD_PATH,
        "--sas-file", sas_file,
        "--plan-file", plan_file,
        pddl_file,
        "--search", f"sampling_search_yaaig(astar(lmcut(transform=sampling_transform()), " \
            "transform=sampling_transform()), techniques=[gbackward_yaaig(searches=1, " \
            f"samples_per_search=200, max_samples={num_samples}, technique=rw, random_seed={seed})], " \
            f"state_representation=complete, random_seed={seed}, minimization=none, avi_k=0)"
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    assert f"Generated Entries: {num_samples}" in stdout.decode("utf-8")
    assert not stderr.decode("utf-8")
    remove(sas_file)
    return plan_file

def is_dead_born(sample_file: str, seed: int, weight_initialization: str) -> bool:
    assert weight_initialization in ["default", "sqrt_k", "1", "01", "xavier_uniform",
        "xavier_normal", "kaiming_uniform", "kaiming_normal", "rai"]
    command = [
        _TRAIN_PATH,
        sample_file,
        "-s", str(seed),
        "-wm", weight_initialization,
        "-e", "1",
        "-rst", "false",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, stderr = process.communicate()
    stderr = stderr.decode("utf-8")
    assert "Traceback (most recent call last):" not in stderr
    return "All predictions are 0 (born dead)" in stderr

def check_dead_born(pddl_file: int, weight_initialization: str, seeds: int, num_sample_files: int = 5):
    domain, problem = get_domain_and_problem(pddl_file)
    print("-", domain, problem)
    sample_files = [sampling(pddl_file, num_samples=1000+2000*i*i, seed=i) for i in range(num_sample_files)]
    for seed in range(seeds):
        dead_born = [is_dead_born(sample_files[i], seed, weight_initialization) for i in range(0, num_sample_files)]
        print(seed, "".join(["y" if x else "n" for x in dead_born]))
        rmtree('results', ignore_errors=True)
    for file in sample_files:
        remove(file)

if __name__ == '__main__':
    for pddl_file in argv[3:]:
        check_dead_born(pddl_file, argv[1], int(argv[2]))
