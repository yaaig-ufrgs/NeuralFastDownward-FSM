#!/bin/bash

# Test with a given configuration.
# Requirements: tsp, taskset
#
# Usage:
# $ ./fast-test.sh <thread|mult> <cores> <trained models dir> <tasks dir>
#
# Example:
# $ ./fast-test.sh multi 10 results tasks/ferber21/test_states/blocks/*/p*.pddl
# $ ./fast-test.sh 0 -1 results tasks/ferber21/test_states/blocks/moderate/probBLOCKS-14-0/p*.pddl
#

TEST_TYPE=$1
CORES=$2
MODELS=$3
TASKS=${@:4}
task_files=(${TASKS})

if [ $TEST_TYPE = "multi" ] ; then
    tsp -S $CORES
    count=0
    # don't count domain.pddl and atoms.json
    for value in "${task_files[@]}" ; do
        f="$(cut -d '/' -f7 <<< ${value})"
        if [ $f = "atoms.json" ] || [ $f = "domain.pddl" ] || [ $f = "data_set_sizes.json" ] ; then
            count=$((count+1))
        fi
    done

    runs=1
    task_files_len=$(((${#task_files[@]}-$count)*${runs}))
    max_per_thread=$((($task_files_len+$CORES-1)/$CORES))
    COUNTER=0
    THREAD_ID=-1
    #c=0

    for file in ${task_files[@]} ; do
        echo $file
        domain_pddl="${file%/*}/domain.pddl"
        domain="$(cut -d '/' -f4 <<< ${file})"
        problem="$(cut -d '/' -f6 <<< ${file})"
        pddl="$(cut -d '/' -f7 <<< ${file})"
        if [ $pddl = "atoms.json" ] || [ $pddl = "domain.pddl" ] || [ $f = "data_set_sizes.json" ] ; then
            continue
        fi
        instance_trained_on="${domain}-${problem}"
        if [ $domain = "blocks" ] ; then
            instance_trained_on="blocks_${problem}"
        fi
        echo $instance_trained_on
        train_dir=$(find "results/" -maxdepth 1 -name \*${instance_trained_on}\* -type d -print | head -n1)/
        echo $train_dir
        facts=$(find "samples/" -maxdepth 1 -name \*${instance_trained_on}\*_facts\* -type f -print | head -n1)
        defaults=$(find "samples/" -maxdepth 1 -name \*${instance_trained_on}\*_defaults\* -type f -print | head -n1)
        echo $facts
        echo $defaults
        echo ""

        if [ $(($COUNTER%$max_per_thread)) = 0 ]; then
            THREAD_ID=$((THREAD_ID+1))
            tsp taskset -c ${THREAD_ID} ./test.py ${train_dir} ${file} -ffile ${facts} -dfile ${defaults} -t 360 -a eager_greedy -pt best
        else
            tsp -D $((COUNTER-1)) taskset -c ${THREAD_ID} ./test.py ${train_dir} ${file} -ffile ${facts} -dfile ${defaults} -t 360 -a eager_greedy -pt best
        fi

        #c=$((c+1))
        COUNTER=$((COUNTER+1))
    done

else
    THREAD_ID=$TEST_TYPE
    file=${task_files[0]}
    domain="$(cut -d '/' -f4 <<< ${file})"
    problem="$(cut -d '/' -f6 <<< ${file})"
    instance_trained_on="${domain}-${problem}"
    if [ $domain = "blocks" ] ; then
        instance_trained_on="blocks_${problem}"
    fi
    echo $instance_trained_on
    train_dir=$(find "results/" -maxdepth 1 -name \*${instance_trained_on}\* -type d -print | head -n1)/
    echo $train_dir
    facts=$(find "samples/" -maxdepth 1 -name \*${instance_trained_on}\*_facts\* -type f -print | head -n1)
    defaults=$(find "samples/" -maxdepth 1 -name \*${instance_trained_on}\*_defaults\* -type f -print | head -n1)
    echo $facts
    echo $defaults
    if [ $CORES = -1 ]; then
        tsp taskset -c ${THREAD_ID} ./test.py ${train_dir} $TASKS -ffile ${facts} -dfile ${defaults} -t 360 -a eager_greedy -pt best
    else
        tsp -D $CORES taskset -c ${THREAD_ID} ./test.py ${train_dir} $TASKS -ffile ${facts} -dfile ${defaults} -t 360 -a eager_greedy -pt best
    fi
    echo ""
fi
