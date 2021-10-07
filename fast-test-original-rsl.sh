#!/bin/bash

# Test with a given configuration.
# Requirements: tsp, taskset
#
# Usage:
# $ ./fast-test.sh <cores> <trained models dir> <tasks dir>
#
# Example:
# $ ./fast-test.sh 10 results tasks/ferber21/test_states/blocks/*/*.pddl
#

CORES=$1
MODELS=$2
TASKS=${@:3}

tsp -S $CORES

task_files=(${TASKS})

count=0
# don't count domain.pddl and atoms.json
for value in "${task_files[@]}" ; do
    f="$(cut -d '/' -f7 <<< ${value})"
    if [ $f = "atoms.json" ] || [ $f = "domain.pddl" ] ; then
        count=$((count+1))
    fi
done

runs=1
task_files_len=$(((${#task_files[@]}-$count)*${runs}))
max_per_thread=$((($task_files_len+$CORES-1)/$CORES))
COUNTER=0
THREAD_ID=-1

for file in ${task_files[@]} ; do
    echo $file
    domain_pddl="${file%/*}/domain.pddl"
    domain="$(cut -d '/' -f4 <<< ${file})"
    problem="$(cut -d '/' -f6 <<< ${file})"
    pddl="$(cut -d '/' -f7 <<< ${file})"
    if [ $pddl = "atoms.json" ] || [ $pddl = "domain.pddl" ] ; then
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

    COUNTER=$((COUNTER+1))
done
