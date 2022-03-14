#!/usr/bin/env python3

"""
Run experimentes configured from JSON files.
./run.py [exp1.json exp2.json ... expn.json]

See exp_example.json files and `src/pytorch/utils/parse_args.py`.

In each JSON, arguments with empty string take the default values
from `src/pytorch/utils/default_args.py`.
"""

from distutils.command.build import build
from sys import argv
import os
import time
from json import load


def build_args(d: dict, prefix: str) -> str:
    args = ""
    for k, v in d.items():
        if v == "" or k == "exp-only-sampling":
            continue
        if k in ["samples", "problem-pddls", "method", "instances_dir"]:
            args += f" {v}"
        else:
            arg = prefix + k
            args += f" {arg} {v}"
    return args


def str2bool(s: str) -> bool:
    s = s.lower()
    if s in ["no", "false", "0"]:
        return False
    if s in ["yes", "true", "1"]:
        return True
    raise Exception(f"{s} isn't a boolean!")


def main(exp_paths: [str]):
    for exp_path in exp_paths:
        full_exp = {}
        with open(exp_path, "r") as exp_file:
            full_exp = load(exp_file)

        exp = full_exp["experiment"] if "experiment" in full_exp else None
        train = full_exp["train"] if "train" in full_exp else None
        test = full_exp["test"] if "test" in full_exp else None
        evalu = full_exp["eval"] if "eval" in full_exp else None
        sampling = full_exp["sampling"] if "sampling" in full_exp else None

        if not exp:
            continue

        only_sampling = str2bool(exp["exp-only-sampling"])
        only_train = str2bool(exp["exp-only-train"])
        only_test = str2bool(exp["exp-only-test"])
        only_eval = str2bool(exp["exp-only-eval"])

        if sampling and not any([only_train, only_test, only_eval]):
            args = "./fast_sample.py"
            args += build_args(sampling, "--")
            print(args, end="\n\n")
            os.system(args)

        if not only_sampling:
            args = "./run_experiment.py"
            args += build_args(exp, "--")
            if evalu and only_eval:
                args += build_args(evalu, "--eval-")
            else:
                if train and not only_test:
                    args += build_args(train, "--train-")
                if test and not only_train:
                    args += build_args(test, "--test-")
            print(args, end="\n\n")
            os.system(args)

        time.sleep(2)
        while True:
            p = os.popen("tsp")
            out = p.read()
            p.close()
            if "queued" not in out and "running" not in out:
                if "skipped" in out:
                    print(f"{exp_path}: Some task(s) were skipped.")
                break
            time.sleep(180)

        out = ""
        while "queued" in out and "running" in out:
            time.sleep(180)
            p = os.popen("tsp")
            out = p.read()
            p.close()


if __name__ == "__main__":
    main(argv[1:])
