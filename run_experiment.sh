#!/bin/sh

# Train & test with a given configuration.
# Requirements: tsp, taskset
#
# Usage:
# $ ./run_experiment.sh [fukunaga|ferber] [500x200|30K|100K] [change_all|fixed_net|fixed_sample] problem seed
#
# Example:
# $ ./run_experiment.sh fukunaga 500x200 fixed_net blocks 1

# run_general <sample_seed> <net_seed> <rw|dfs> <thread>
run_experiment() {
    sample_seed=$1
    net_seed=$2
    sample_type=$3
    thread=$4
    files=(../samples/${TECHNIQUE}/${PROBLEM}/${TECHNIQUE}_*_${sample_type}_fs_${SAMPLE_SIZE}_seed${sample_seed})
    for file in ${files[@]} ; do
        prob=${file#*${TECHNIQUE}_};
        prob=${prob%%_*}
        #echo $file
        #echo $prob.pddl
        #tsp taskset -c $thread ./train.py -o regression -f 1 -hl 1 -hu 16 -t 1800 -a sigmoid -s $net_seed file
        # TODO get model folder
        #tsp -D $COUNTER taskset -c $thread ./test.py results/algum_nome_seed${net_seed}/ ../downward-benchmarks/${PROBLEM}/${prob}.pddl -t 300 -a eager_greedy
        COUNTER=$((COUNTER+1))
    done
}

tsp -K
tsp -S 10

TECHNIQUE=$1
SAMPLE_SIZE=$2
EXPERIMENT=$3
PROBLEM=$4
SEED=$5 # for fixed seed experiments

if [ $TECHNIQUE = "fukunaga" ]; then

    COUNTER=0

    if [ $EXPERIMENT = "fixed_net" ]; then
        # run_experiment <sample_seed> <net_seed> <sample_type> <thread>
        run_experiment 1 $SEED "rw" 0
        run_experiment 2 $SEED "rw" 1
        run_experiment 3 $SEED "rw" 2
        run_experiment 4 $SEED "rw" 3
        run_experiment 5 $SEED "rw" 4

        run_experiment 1 $SEED "dfs" 5
        run_experiment 2 $SEED "dfs" 6
        run_experiment 3 $SEED "dfs" 7
        run_experiment 4 $SEED "dfs" 8
        run_experiment 5 $SEED "dfs" 9
    elif [ $EXPERIMENT = "fixed_sample" ]; then
        run_experiment $SEED 1 "rw" 0
        run_experiment $SEED 2 "rw" 1
        run_experiment $SEED 3 "rw" 2
        run_experiment $SEED 4 "rw" 3
        run_experiment $SEED 5 "rw" 4

        run_experiment $SEED 1 "dfs" 5
        run_experiment $SEED 2 "dfs" 6
        run_experiment $SEED 3 "dfs" 7
        run_experiment $SEED 4 "dfs" 8
        run_experiment $SEED 5 "dfs" 9
    elif [ $EXPERIMENT = "change_all" ]; then
        run_experiment 1 1 "rw" 0
        run_experiment 2 2 "rw" 1
        run_experiment 3 3 "rw" 2
        run_experiment 4 4 "rw" 3
        run_experiment 5 5 "rw" 4

        run_experiment 1 1 "dfs" 5
        run_experiment 2 2 "dfs" 6
        run_experiment 3 3 "dfs" 7
        run_experiment 4 4 "dfs" 8
        run_experiment 5 5 "dfs" 9
    fi
else
    echo "TODO FERBER"
fi

