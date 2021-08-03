#!/bin/bash
#
# Generate from a given task a new task (e.g. by changing the initial state)
#
# Use: $ ./fast-sample.sh DOMAIN PROBLEM NB_OF_TASKS_TO_GENERATE MIN_WALK_LENGTH MAX_WALK_LENGTH [forward | backward] [random_state | entire_plan | init_state |]
# e.g. $ ./fast-sample.sh domain.pddl p01.pddl 5 10 20
#      $ ./fast-sample.sh domain.pddl p01.pddl 50 10 20 backward entire_plan

DOMAIN=$1
PROBLEM=$2
NB_OF_TASKS_TO_GENERATE=$3
MIN_WALK_LENGTH=$4
MAX_WALK_LENGTH=$5

TECHNIQUE="iforward" # iforward or gbackward
SELECT_STATE="random_state"
if [[ $# == 7 ]];
then
    TECHNIQUE=$6
    SELECT_STATE=$7

    if [[ $TECHNIQUE == "forward" ]]; then TECHNIQUE="iforward"
    elif [[ $TECHNIQUE == "backward" ]]; then TECHNIQUE="gbackward"
    fi
fi

$(dirname $0)/fast-downward.py --build debug $DOMAIN $PROBLEM \
                   --search "sampling_search_ferber(eager_greedy([ff(transform=sampling_transform())], \
                             transform=sampling_transform()), \
                             techniques=[${TECHNIQUE}_none($NB_OF_TASKS_TO_GENERATE, \
                                distribution=uniform_int_dist($MIN_WALK_LENGTH, $MAX_WALK_LENGTH))],
                                select_state_method=$SELECT_STATE)"
