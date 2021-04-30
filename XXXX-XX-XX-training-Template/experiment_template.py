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

UNARY_THRESHOLD = {FORMAT_UNARY_THRESHOLDS}  # [0.01]

PREFIXES = {FORMAT_PREFIXES}  # [OCLS_....]

filter_domains = {FORMAT_FILTER_DOMAINS}  # [domain1, domain2, ...]

skip_tasks = {FORMAT_SKIP_TASKS}  # set([t1, T2, ...])

exp = base_experiment.get_base_experiment(
    False, filter_benchmarks=filter_domains,
    skip_domains=skip_tasks,
    extra_options=extra_options)

for params in [base_experiment.get_network_param_from_prefix(prefix)
               for prefix in PREFIXES]:
    TYPE = params[base_experiment.Param.NETWORK_TYPE]
    STATE_LAYER = params[base_experiment.Param.INPUT_STATE]
    GOAL_LAYER = params[base_experiment.Param.INPUT_GOAL]
    assert (TYPE != base_experiment.CLASSIFICATION or
            (len(UNARY_THRESHOLD) == 1 and UNARY_THRESHOLD[0] == 0))
    
    CONFIGURATIONS = [list(
        base_experiment.convert_network_param_to_search_configuration(params))]

    BASE_NAME = CONFIGURATIONS[0][0]
    for unary_threshold in UNARY_THRESHOLD:
        CONFIGURATIONS[0][0] = (
                BASE_NAME +
                ("" if unary_threshold == 0 else
                 ("_unary_threshold_%%.%if" % abs(
                     math.log10(unary_threshold))) %
                 unary_threshold)
        )

        print(CONFIGURATIONS)
        base_experiment.add_nn_algorithm(
            exp, base_experiment.get_fd_network_type(TYPE),
            STATE_LAYER, GOAL_LAYER, CONFIGURATIONS,
            unary_threshold=unary_threshold)

exp.run_steps()
