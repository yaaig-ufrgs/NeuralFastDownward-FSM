#!/usr/bin/env python3

"""
Run experimentes configured from JSON files.
./run.py [exp1.json exp2.json ... expn.json]

See exp_example.json files and `src/pytorch/utils/parse_args.py`.

In each JSON, arguments with empty string take the default values
from `src/pytorch/utils/default_args.py`.
"""

import os
import time
import re
from sys import argv
from json import load
from glob import glob
from natsort import natsorted
from scripts.create_random_sample import random_sample_statespace


def build_args(d: dict, prefix: str) -> str:
    args = ""
    for k, v in d.items():
        if v == "" or k == "exp-only-sampling" or k == "modify-sample":
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


def wait(secs: int, exp_path: str):
    while True:
        p = os.popen("tsp")
        out = p.read()
        p.close()
        if "queued" not in out and "running" not in out:
            if "skipped" in out:
                print(f"{exp_path}: Some task(s) were skipped.")
            break
        time.sleep(secs)


def remove_leftover_files(output_dir: str):
    sas_files = glob(f'{output_dir}/*-output.sas')
    for sf in sas_files:
        if os.path.isfile(sf):
            os.remove(sf)


def sort_list_intercalate(files: [str]) -> [str]:
    ret = []
    d = {}
    files = natsorted(files)
    for f in files:
        f_split = f.split('/')[-1].split('_')
        domain = f_split[0]
        if domain not in d:
            d[domain] = []
        d[domain].append(f)

    count = 0
    while count < len(d):
        for k in d:
            if not d[k]:
                continue
            ret.append(d[k].pop(0))
            if not d[k]:
                count += 1

    return ret


def do_sample_mod(mod: str, samples_dir: str, statespace: str, min_seed: int, max_seed: int):
    if mod == "random-sample":
        pct = float(re.findall(".*-(\d*)pct.*", samples_dir)[0]) * 0.01
        random_sample_statespace(statespace, samples_dir, min_seed, max_seed, [pct])

def main(exp_paths: [str]):
    intercalate = False
    if intercalate:
        exp_paths = sort_list_intercalate(exp_paths) 
    for exp_path in exp_paths:
        full_exp = {}
        with open(exp_path, "r") as exp_file:
            full_exp = load(exp_file)

        exp = full_exp["experiment"] if "experiment" in full_exp else None
        if exp is None:
            continue
        train = full_exp["train"] if "train" in full_exp else None
        test = full_exp["test"] if "test" in full_exp else None
        evalu = full_exp["eval"] if "eval" in full_exp else None
        sampling = full_exp["sampling"] if "sampling" in full_exp else None
        if sampling is not None:
            sampling["threads"] = exp["exp-threads"]
            sampling["output-dir"] = exp["samples"]
        if train is not None and test is not None:
            test["model-dir"] = train["output-folder"]

        only_sampling = str2bool(exp["exp-only-sampling"])
        only_train = str2bool(exp["exp-only-train"])
        only_test = str2bool(exp["exp-only-test"])
        only_eval = str2bool(exp["exp-only-eval"])

        mod_sample = "default" if "modify-sample" not in exp else exp["modify-sample"]
        if mod_sample != "default":
            if type(exp['exp-sample-seed']) == str and ".." in exp['exp-sample-seed']:
                min_seed, max_seed = [int(n) for n in exp['exp-sample-seed'].split('..')]
            else:
                min_seed = int(exp['exp-sample-seed'])
                max_seed = min_seed
            do_sample_mod(mod_sample, exp["samples"], sampling["statespace"], min_seed, max_seed)

        if sampling and not any([only_train, only_test, only_eval]) and mod_sample == "default":
            args = "./fast_sample.py"
            args += build_args(sampling, "--")
            print(args, end="\n\n")
            os.system(args)

            time.sleep(2)
            wait(10, exp_path)

            remove_leftover_files(exp["samples"])

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
            wait(180, exp_path)

if __name__ == "__main__":
    main(argv[1:])
