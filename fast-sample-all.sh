#!/bin/bash
#
# Generate sample states from a set of instances.
#
# Usage:
# $ ./fast-sample-all.sh [fukunaga|ferber] [rw|dfs] [fs|ps] searches samples_per_search seed problem_dir output_dir
#
# Example:
# $ ./fast-sample-all.sh fukunaga dfs fs 10 10 1 tasks/IPC/blocks testdir
#
# Beware: don't insert slashes on the directories.

METHOD=$1
TECHNIQUE=$2
STATE=$3
SEARCHES=$4
SAMPLES_PER_SEARCH=$5
SEED=$6
PROBLEM_DIR=$7
OUTPUT_DIR=$8
USE_DFS="true"
USE_FULL_STATE="true"

if [ $TECHNIQUE = "rw" ]; then
    USE_DFS="false"
fi

if [ $STATE = "ps" ]; then
    USE_FULL_STATE="false"
fi


mkdir $OUTPUT_DIR

if [ $METHOD = "fukunaga" ]; then
    files=($PROBLEM_DIR/*.pddl)
    for file in ${files[@]}; do
        prob_name=${file#*${PROBLEM_DIR}/};
        prob_name=${prob_name%%.pddl*}
        if [ $prob_name = "domain" ]; then
            continue
        fi
        ./fast-downward.py --plan-file $OUTPUT_DIR/${METHOD}_${prob_name}_${TECHNIQUE}_${STATE}_${SEARCHES}x${SAMPLES_PER_SEARCH}_ss${SEED} \
            --build debug $file \
            --search "sampling_search_fukunaga(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()), \
            techniques=[gbackward_fukunaga(searches=$SEARCHES, samples_per_search=$SAMPLES_PER_SEARCH, \
            use_dfs=$USE_DFS, random_seed=$SEED)], use_full_state=$USE_FULL_STATE)"
    done
else
    echo "TODO FERBER"
fi
