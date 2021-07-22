#!/bin/bash
#
# Generate from a given task a new task (e.g. by changing the initial state)
#
# Use: $ ./fast-sample.sh DOMAIN PROBLEM NB_OF_TASKS_TO_GENERATE MIN_WALK_LENGTH MAX_WALK_LENGTH [forward | backward]
# e.g. $ ./fast-sample.sh domain.pddl p01.pddl 5 10 20
# e.g. $ ./fast-sample.sh domain.pddl p01.pddl 50 10 20 backward

DOMAIN=$1
PROBLEM=$2
NB_OF_TASKS_TO_GENERATE=$3
MIN_WALK_LENGTH=$4
MAX_WALK_LENGTH=$5

TECHNIQUE="iforward" # iforward or gbackward
if [[ $# == 6 ]];
then
    TECHNIQUE=$6

    if [[ $TECHNIQUE == "forward" ]]; then TECHNIQUE="iforward"
    elif [[ $TECHNIQUE == "backward" ]]; then TECHNIQUE="gbackward"
    fi
fi

$(dirname $0)/fast-downward.py --build debug $DOMAIN $PROBLEM \
                   --search "sampling_search_simple(astar(lmcut(transform=sampling_transform()), \
                             transform=sampling_transform()), \
                             techniques=[${TECHNIQUE}_none($NB_OF_TASKS_TO_GENERATE, \
                                distribution=uniform_int_dist($MIN_WALK_LENGTH, $MAX_WALK_LENGTH))])"
