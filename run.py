#!/usr/bin/env python3

"""
Run experimentes configured from JSON files.
./run.py [exp1.json exp2.json ... expn.json]

See exp_example.json files and `src/training/utils/parse_args.py`.

In each JSON, arguments with empty string take the default values
from `src/training/utils/default_args.py`.

CAUTION: if using multiple JSONs, make sure they have the same number of 'exp-cores'.
"""

import os
import time
import re
from sys import argv
from json import load
from glob import glob
from scripts.create_random_sample import random_sample_statespace


def build_args(d: dict, prefix: str) -> str:
    args = ""
    for k, v in d.items():
        if v == "" or k == "exp-only-sampling" or k == "modify-sample":
            continue
        if k in ["samples", "problem-pddls", "method", "instance"]:
            args += f" {v}"
        else:
            if k == "evaluator":
                v = '"' + v + '"'
            if k == "max-expansions" and v == "auto":
                v = -1
            arg = prefix + k
            if isinstance(v, list) and k != "output-model":
                v = ":".join(v)
            if k == "output-model":
                v = " ".join(v.split(","))
            args += f" {arg} {v}"
    return args


def str2bool(s: str) -> bool:
    s = s.lower()
    if s in ["no", "false", "0"]:
        return False
    if s in ["yes", "true", "1"]:
        return True
    raise Exception(f"{s} isn't a boolean!")


def wait(secs: int):
    while True:
        p = os.popen("tsp")
        out = p.read()
        p.close()
        if "queued" not in out and "running" not in out:
            if "skipped" in out:
                print(f"Some task(s) were skipped.")
            break
        time.sleep(secs)


def remove_leftover_files(output_dir: str):
    sas_files = glob(f"{output_dir}/*-output.sas")
    for sf in sas_files:
        if os.path.isfile(sf):
            os.remove(sf)


def do_sample_mod(
    mod: str, samples_dir: str, statespace: str, min_seed: int, max_seed: int
):
    if mod == "random-sample":
        pct = float(re.findall(".*-(\d*)pct.*", samples_dir)[0]) * 0.01
        random_sample_statespace(statespace, samples_dir, min_seed, max_seed, [pct])


def get_exp_dicts(full_exp):
    exp = full_exp["experiment"] if "experiment" in full_exp else None
    if exp is None:
        print("Error: invalid experiment.")
        exit(1)
    exp_defaults = {
        "exp-type": "all",
        "exp-only-sampling": "no",
        "exp-only-train": "no",
        "exp-only-test": "no",
        "exp-only-eval": "no",
        "trained-model": "",
        "samples": "None",
        "exp-sample-seed": "0",
        "exp-net-seed": "0",
        "exp-cores": "1",
        "unit-cost": "no",
    }
    for key in exp_defaults:
        if key not in exp:
            exp[key] = exp_defaults[key]

    results_bckp = exp["results"]
    train = full_exp["train"] if "train" in full_exp else None
    test = full_exp["test"] if "test" in full_exp else None
    evalu = full_exp["eval"] if "eval" in full_exp else None
    sampling = full_exp["sampling"] if "sampling" in full_exp else None
    if sampling:
        sampling["cores"] = exp["exp-cores"]
        sampling["output-dir"] = exp["samples"]
        sampling["seed"] = exp["exp-sample-seed"]
        if "unit-cost" in exp:
            sampling["unit-cost"] = exp["unit-cost"]
    if train:
        train["output-folder"] = exp["results"]
    if test:
        test["model-dir"] = exp["results"]
        if "unit-cost" in exp:
            test["unit-cost"] = exp["unit-cost"]

    for remove_label in ["results", "unit-cost"]:
        if remove_label in exp:
            del exp[remove_label]

    if train and test:
        if "save-git-diff" in train:
            test["save-git-diff"] = train["save-git-diff"]

    return (exp, sampling, train, test, evalu, results_bckp)


def main(exp_paths: [str]):
    full_exps = []
    for exp_path in exp_paths:
        try:
            full_exp = {}
            with open(exp_path, "r") as exp_file:
                full_exps.append(load(exp_file))
        except Exception as e:
            print(f"Error with {exp_path}: {e}")

    # 1st pass: perform sampling.
    sample_locations = []
    all_exps = []
    for full_exp in full_exps:
        exp, sampling, train, test, evalu, results_bckp = get_exp_dicts(full_exp)
        all_exps.append((exp, sampling, train, test, evalu))

        only_train = str2bool(exp["exp-only-train"]) or (
            not sampling and not test and not evalu
        )
        only_test = str2bool(exp["exp-only-test"]) or (
            not sampling and not train and not evalu
        )
        only_eval = str2bool(exp["exp-only-eval"]) or (
            not sampling and not train and not test
        )

        if sampling and not any([only_train, only_test, only_eval]):
            mod_sample = (
                "default" if "modify-sample" not in exp else exp["modify-sample"]
            )
            if mod_sample == "default":
                args = "./fast-sample.py"
                args += build_args(sampling, "--")

                try:
                    with open("PID", "r") as f:
                        pid = f.readline().strip()
                        if pid == "":
                            pid = 0
                except Exception:
                    pid = 0

                args += f" --pid {pid}"
                print("run.py [sampling]:", args, end="\n\n")
                os.system(args)

                sample_locations.append(exp["samples"])
            else:
                if (
                    type(exp["exp-sample-seed"]) == str
                    and ".." in exp["exp-sample-seed"]
                ):
                    min_seed, max_seed = [
                        int(n) for n in exp["exp-sample-seed"].split("..")
                    ]
                else:
                    min_seed = int(exp["exp-sample-seed"])
                    max_seed = min_seed
                do_sample_mod(
                    mod_sample,
                    exp["samples"],
                    sampling["statespace"],
                    min_seed,
                    max_seed,
                )

    with open("PID", "w") as f:
        f.write("0")

    wait(5)

    for sample_loc in sample_locations:
        remove_leftover_files(sample_loc)

    print(all_exps)
    # 2nd pass: do rest.
    for (exp, sampling, train, test, evalu) in all_exps:
        only_sampling = str2bool(exp["exp-only-sampling"]) or (
            not train and not test and not evalu
        )
        only_train = str2bool(exp["exp-only-train"]) or (
            not sampling and not test and not evalu
        )
        only_test = str2bool(exp["exp-only-test"]) or (
            not sampling and not train and not evalu
        )
        only_eval = str2bool(exp["exp-only-eval"]) or (
            not sampling and not train and not test
        )

        if evalu and "trained-model" not in evalu:
            evalu["trained-model"] = results_bckp + "/*/models/*.pt"

        if not only_sampling:
            args = "./run-experiment.py"
            args += build_args(exp, "--")
            if evalu and only_eval:
                args += build_args(evalu, "--eval-")
            else:
                if train and not only_test:
                    args += build_args(train, "--train-")
                if test and not only_train:
                    args += build_args(test, "--test-")
                if evalu and not only_test:
                    args += build_args(evalu, "--eval-")
            try:
                with open("PID", "r") as f:
                    pid = f.readline().strip()
                    if pid == "":
                        pid = 0
            except Exception:
                pid = 0

            args += f" --pid {pid}"
            print("run.py [train/test]:", args, end="\n\n")
            os.system(args)

    if os.path.isfile("PID"):
        os.remove("PID")


if __name__ == "__main__":
    main(argv[1:])
