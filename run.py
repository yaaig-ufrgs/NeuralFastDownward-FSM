#!/usr/bin/env python3

"""
Run experimentes configured from JSON files.
./run.py [exp1.json exp2.json ... expn.json]

See exp.json files and parse_args.py.
"""

import sys
import os
from json import load

def build_args(d: dict, prefix: str) -> str:
    args = ""
    for k, v in exp.items():
        if v == "" or k == "ignore":
            continue
        if k == "samples":
            args += f"{v} "
        else:
            arg = prefix + k
            args += f"{arg} {v} "
    return args


def str2bool(s: str) -> bool:
    s = s.lower()
    if s == "no" or  s == "false":
        return False
    if s == "yes" or s == "true":
        return True


for exp_path in sys.argv[1:]:
    full_exp = {}
    with open(exp_path, 'r') as exp_file:
        full_exp = load(exp_file)

    exp = full_exp['experiment']
    train = full_exp['train']
    test = full_exp['test']
    evalu = full_exp['eval']
    sample = full_exp['sample']

    if not str2bool(exp['ignore']):
        args = f"./run_experiment.py "
        args += build_args(exp, "--")
        if not str2bool(train['ignore']):
            args += build_args(train, "--train-")
        if not str2bool(test['ignore']):
            args += build_args(test, "--test-")
           

    print(args)
                
                
        
