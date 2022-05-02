#!/usr/bin/env python3

"""
usage: ./gen_big_experiments.py output_folder ../tasks/ferber21/test_states/*/*/*
"""

from sys import argv
from os.path import exists, join
import pathlib

_MAX_SEED = 4

_SAMPLING_TIME = 10 * 60
_SAMPLING_MEMORY = 2 * 1024

_TRAIN_TIME = 30 * 60
_TRAIN_BATCH = 64

_TEST_TIME = 5 * 60
_TEST_MEMORY = 2 * 1024
_TEST_INITIAL_STATES = 25

base = """
{
    "experiment": {
        "samples": "samples/{domain}/{difficult}/samples_{domain}_{problem}_bfsrw",
        "exp-type": "all",
        "exp-net-seed": "0..{max_seed}",
        "exp-sample-seed": "0..{max_seed}",
        "exp-threads": 10,
        "exp-only-sampling": "no",
        "exp-only-train": "no",
        "exp-only-test": "no",
        "exp-only-eval": "no",
        "trained-model": ""
    },
    "sampling": {
        "instance": "tasks/ferber21/training_tasks/{domain}/{problem}.pddl",
        "method": "yaaig",
        "technique": "rw",
        "subtechnique": "percentage",
        "state-representation": "fs",
        "max-time": {sampling_time},
        "mem-limit": {sampling_memory},
        "seed": "0..{max_seed}",
        "allow-dups": "interrollout",
        "restart-h-when-goal-state": "yes",
        "minimization": "both",
        "avi-k": 1,
        "avi-its": 9999,
        "bound": "propositions_per_mean_effects",
        "minimization-before-avi": "no"
    },
    "train": {
        "model": "resnet",
        "max-training-time": {train_time},
        "training-size": 0.9,
        "output-folder": "results/{domain}/{difficult}/results_{domain}_{problem}_bfsrw",
        "restart-no-conv": "yes",
        "hidden-layers": 2,
        "hidden-units": 250,
        "batch-size": {train_batch},
        "patience": 100,
        "activation": "relu",
        "loss-function": "mse",
        "weights-method": "kaiming_uniform",
        "shuffle": "yes",
        "learning-rate": 0.0001,
        "normalize-output": "no",
        "output-layer": "regression",
        "use-gpu": "no"
    },
    "test": {
        "search-algorithm": "eager_greedy",
        "max-search-time": {test_time},
        "max-search-memory": {test_memory},
        "auto-tasks-n": {test_ntasks}
    }
}
"""

output_folder = argv[1]
for task_folder in argv[2:]:
    _, domain, difficult, problem = task_folder.rsplit("/", 3)
    assert difficult in ["moderate", "hard"]
    assert exists(f"../tasks/ferber21/training_tasks/{domain}/{problem}.pddl")
    folder = join(output_folder, domain, difficult)
    pathlib.Path(folder).mkdir(parents=True, exist_ok=True) 
    with open(join(folder, f"{domain}_{problem}.json"), "w") as f:
        f.write(
            base\
                .replace("{domain}", domain)\
                .replace("{problem}", problem)\
                .replace("{difficult}", difficult)\
                .replace("{max_seed}", str(_MAX_SEED))\
                .replace("{sampling_time}", str(_SAMPLING_TIME))\
                .replace("{sampling_memory}", str(_SAMPLING_MEMORY))\
                .replace("{train_time}", str(_TRAIN_TIME))\
                .replace("{train_batch}", str(_TRAIN_BATCH))\
                .replace("{test_time}", str(_TEST_TIME))\
                .replace("{test_memory}", str(_TEST_MEMORY))\
                .replace("{test_ntasks}", str(_TEST_INITIAL_STATES))
        )
