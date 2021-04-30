#! /usr/bin/env python
import base_experiment
from base_experiment import REGRESSION, CLASSIFICATION, ORDINAL_CLASSIFICATION

import math
import os
import sys


if len(sys.argv) > 2 and sys.argv[1] == "extra_options":
    extra_options = sys.argv[2]
    sys.argv = [sys.argv[0]] + sys.argv[3:]
else:
    extra_options = None

UNARY_THRESHOLD = [0.01]

PREFIXES = [
    "ocls_ns_ubal_h3_sigmoid_inter_gen_sat_drp0_Kall_pruneOff_",
]


filter_domains = base_experiment.DOMAIN_DIRS_AAAI20_REDUCED_REDUCED

exp = base_experiment.get_base_experiment(
    False, filter_benchmarks=".*(%s).*" % "|".join(filter_domains),
    extra_options=extra_options,
    filter_task_regenerated="replaced_pddls_files.json",)


PARAMS = [
    base_experiment.get_network_param_from_prefix(prefix)
    for prefix in PREFIXES
]


for params in PARAMS:

    TYPE = params[base_experiment.Param.NETWORK_TYPE]
    STATE_LAYER = params[base_experiment.Param.INPUT_STATE]
    GOAL_LAYER = params[base_experiment.Param.INPUT_GOAL]

    CONFIGURATIONS = [
        list(base_experiment.convert_network_param_to_search_configuration(params))
    ]

    BASE_NAME = CONFIGURATIONS[0][0]
    for unary_threshold in UNARY_THRESHOLD:
        CONFIGURATIONS[0][0] = (
                BASE_NAME +
                ("_unary_threshold_%%.%if" % abs(math.log10(unary_threshold))) %
                unary_threshold)

        print(CONFIGURATIONS)
        base_experiment.add_nn_algorithm(
            exp, base_experiment.get_fd_network_type(TYPE),
            STATE_LAYER, GOAL_LAYER, CONFIGURATIONS,
            unary_threshold=unary_threshold,
            algorithm=base_experiment.get_algorithm_eager_greedy(
                [], [base_experiment.HEURISTIC_HFF_UNIFORM])
        )

exp.run_steps()

