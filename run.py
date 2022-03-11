#!/usr/bin/env python3

"""
Run experimentes configured from JSON files.
./run.py [exp1.json exp2.json ... expn.json]

See exp.json files and `parse_args.py`.

In each JSON, arguments with empty string take the default values
from `default_args.py`.
"""

import sys
import os
from json import load


def build_args(d: dict, prefix: str) -> str:
    args = ""
    for k, v in d.items():
        if v == "" or k == "ignore":
            continue
        if k == "samples":
            args += f"{v} "
        elif k == "problem-pddls":
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
        only_train = str2bool(exp['exp-only-train'])
        only_test = str2bool(exp['exp-only-test'])
        only_eval = str2bool(exp['exp-only-eval'])
        args = f"./run_experiment.py "
        args += build_args(exp, "--")
        if not str2bool(evalu['ignore']) and only_eval:
            args += build_args(evalu, "--eval-")
        else:
            if not str2bool(train['ignore']) and not only_test:
                args += build_args(train, "--train-")
            if not str2bool(test['ignore']) and not only_train:
                args += build_args(test, "--test-")

    print(args)
    print()
    os.system(args)
    # TODO WHILE
    # TODO SAMPLE
