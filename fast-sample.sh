#!/bin/bash
#
# Generate sample states from a set of instances.
#
# Usage:
# $ ./fast-sample.sh [fukunaga|ferber] [rw|dfs] [fs|ps] searches samples_per_search n_seeds problem_dir output_dir
#
# Example:
# $ ./fast-sample.sh fukunaga dfs fs 500 200 5 tasks/IPC/blocks samples
#
# Beware: don't insert slashes on the directories.

METHOD=$1
TECHNIQUE=$2
STATE=$3
SEARCHES=$4
SAMPLES_PER_SEARCH=$5
N_SEEDS=$6
PROBLEM_DIR=$7
OUTPUT_DIR=$8
STATE_REPRESENTATION="complete"
MATCH_HEURISTICS="true"

if [ ! $TECHNIQUE = "rw" ] && [ ! $TECHNIQUE = "dfs" ]; then
    echo "Invalid search technique. Choose between rw (random walk) or dfs (depth-first search)."
    exit 1
fi

if [ $STATE = "fs" ] || [ $STATE = "complete" ]; then
    STATE_REPRESENTATION="complete"
    STATE="fs"
elif [ $STATE = "ps" ] || [ $STATE = "partial" ]; then
    STATE_REPRESENTATION="partial"
    STATE="ps"
elif [ $STATE = "us" ] || [ $STATE = "undefined" ]; then
    STATE_REPRESENTATION="undefined"
    STATE="us"
else
    echo "Invalid state representation. Choose between complete, partial, or undefined."
    exit 1
fi

if [ ! -d $OUTPUT_DIR ]; then
    mkdir $OUTPUT_DIR
fi

if [ $METHOD = "fukunaga" ]; then
    files=($PROBLEM_DIR/*.pddl)
    domain_name=${PROBLEM_DIR##*/}
    for file in ${files[@]}; do
        prob_name=${file#*${PROBLEM_DIR}/};
        prob_name=${prob_name%%.pddl*}
        if [ $prob_name != "domain" ]; then
            for seed in $(seq 1 $N_SEEDS); do
                ./fast-downward.py --plan-file $OUTPUT_DIR/${METHOD}_${domain_name}_${prob_name}_${TECHNIQUE}_${STATE}_${SEARCHES}x${SAMPLES_PER_SEARCH}_ss${seed} \
                    --build release $file \
                    --search "sampling_search_fukunaga(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()), \
                    techniques=[gbackward_fukunaga(searches=$SEARCHES, samples_per_search=$SAMPLES_PER_SEARCH, \
                    technique=$TEHNIQUE, random_seed=$seed)], state_representation=$STATE_REPRESENTATION, random_seed=$seed, match_heuristics=$MATCH_HEURISTICS)"
                exit 0
            done
        fi
    done
else
    echo "TODO FERBER"
fi
