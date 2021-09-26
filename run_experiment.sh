#!/bin/bash

# Train & test with a given configuration.
# Requirements: tsp, taskset
#
# Usage:
# $ ./run_experiment.sh [fukunaga|ferber] [dfs|rw] [fs|ps|us] [500x200|30K|100K] [change_all|fixed_net|fixed_sample|single] cores
# OR
# $ ./run_experiment.sh rsl [countBoth|countAdd|countDel] [100000|10000] [change_all|fixed_net|fixed_sample|single] cores
#
# Example:
# $ ./run_experiment.sh fukunaga dfs fs 500x200 single 10

# run_general <sample_seed> <net_seed> <thread> <runs>
run_experiment_fukunaga() {
    sample_seed=$1
    net_seed=$2
    cores=$3
    runs=$4
    files=(samples/${METHOD}_*_*_${TECHNIQUE}_${STATE_REPRESENTATION}_${SAMPLE_SIZE}_ss${sample_seed})
    files_len=$((${#files[@]}*${runs}))
    max_per_thread=$((($files_len+$cores-1)/$cores))
    for file in ${files[@]} ; do
        problem_file=${file#*"samples"/*}
        domain=${problem_file##${METHOD}_}
        domain=${domain%%_*}
        problem=${problem_file##${METHOD}_${domain}_}
        problem=${problem%%_*}
        if [ $(($COUNTER%$max_per_thread)) = 0 ]; then
            THREAD_ID=$((THREAD_ID+1))
            tsp taskset -c ${THREAD_ID} ./train-and-test.sh "-o regression -f 1 -hl 1 -hu 16 -t 99999 -e -1 -a relu -s $net_seed $file" \
                "results/nfd_train.${problem_file}.ns${net_seed}/ tasks/IPC/${domain}/${problem}.pddl -t 99999 -e -1 -a eager_greedy -pt all"
       else
            tsp -D $((COUNTER-1)) taskset -c ${THREAD_ID} ./train-and-test.sh "-o regression -f 1 -hl 1 -hu 16 -t 99999 -e -1 -a relu -s $net_seed $file" \
                "results/nfd_train.${problem_file}.ns${net_seed}/ tasks/IPC/${domain}/${problem}.pddl -t 99999 -e -1 -a eager_greedy -pt all"
        fi

        COUNTER=$((COUNTER+1))
    done
}

run_experiment_rsl() {
    sample_seed=$1
    net_seed=$2
    cores=$3
    runs=$4
    files=(samples/${METHOD}_*_*_${TECHNIQUE}_${SAMPLE_SIZE}_ss${sample_seed})
    files_len=$((${#files[@]}*${runs}))
    max_per_thread=$((($files_len+$cores-1)/$cores))
    for file in ${files[@]} ; do
        facts="${file}_facts.txt"
        defaults="${file}_defaults.txt"
        problem_file=${file#*"samples"/*}
        domain=${problem_file##${METHOD}_}
        domain=${domain%%_*}
        problem=${problem_file##${METHOD}_${domain}_}
        problem=${problem%%_*}
        if [ $(($COUNTER%$max_per_thread)) = 0 ]; then
            THREAD_ID=$((THREAD_ID+1))
            tsp taskset -c ${THREAD_ID} ./train-and-test.sh "-o regression -f 1 -hl 1 -hu 16 -t 99999 -e -1 -a relu -s $net_seed $file" \
                "results/nfd_train.${problem_file}.ns${net_seed}/ tasks/IPC/${domain}/${problem}.pddl -ffile ${facts} -dfile ${defaults} -t 99999 -e -1 -a eager_greedy -pt all"
       else
            tsp -D $((COUNTER-1)) taskset -c ${THREAD_ID} ./train-and-test.sh "-o regression -f 1 -hl 1 -hu 16 -t 99999 -e -1 -a relu -s $net_seed $file" \
                "results/nfd_train.${problem_file}.ns${net_seed}/ tasks/IPC/${domain}/${problem}.pddl -ffile ${facts} -dfile ${defaults} -t 99999 -e -1 -a eager_greedy -pt all"
        fi

        COUNTER=$((COUNTER+1))
    done
}

METHOD=$1
TECHNIQUE=$2
#CORES=$6
CORES=${@:$#}
SEED="1" # for fixed seed experiments

THREAD_ID=-1
COUNTER=0

tsp -K
tsp -S $CORES

if [ $METHOD = "fukunaga" ]; then
    STATE_REPRESENTATION=$3
    SAMPLE_SIZE=$4
    EXPERIMENT=$5

    if [ $EXPERIMENT = "single" ]; then
        # run_experiment_fukunaga <sample_seed> <net_seed> <sample_type> <thread>
        run_experiment_fukunaga 1 1 $CORES 1
    elif [ $EXPERIMENT = "fixed_net" ]; then
        # TODO: for 1..5, dfs rw
        # run_experiment_fukunaga 1 $SEED $CORES 5
        run_experiment_fukunaga 2 $SEED $CORES 4
        run_experiment_fukunaga 3 $SEED $CORES 4
        run_experiment_fukunaga 4 $SEED $CORES 4
        run_experiment_fukunaga 5 $SEED $CORES 4
    elif [ $EXPERIMENT = "fixed_sample" ]; then
        # run_experiment_fukunaga $SEED 1 $CORES 5
        run_experiment_fukunaga $SEED 2 $CORES 4
        run_experiment_fukunaga $SEED 3 $CORES 4
        run_experiment_fukunaga $SEED 4 $CORES 4
        run_experiment_fukunaga $SEED 5 $CORES 4
    elif [ $EXPERIMENT = "change_all" ]; then
        # run_experiment_fukunaga 1 1 $CORES 5
        run_experiment_fukunaga 2 2 $CORES 4
        run_experiment_fukunaga 3 3 $CORES 4
        run_experiment_fukunaga 4 4 $CORES 4
        run_experiment_fukunaga 5 5 $CORES 4
    fi
elif [ $METHOD = "rsl" ]; then
    # $ ./run_experiment.sh rsl [countBoth|countAdd|countDel] [100000|10000] [change_all|fixed_net|fixed_sample|single] cores
    SAMPLE_SIZE=$3
    EXPERIMENT=$4

    if [ $EXPERIMENT = "single" ]; then
        run_experiment_rsl 1 1 $CORES 1
    elif [ $EXPERIMENT = "fixed_net" ]; then
        # TODO: for 1..5, dfs rw
        # run_experiment_rsl 1 $SEED $CORES 5
        run_experiment_rsl 2 $SEED $CORES 4
        run_experiment_rsl 3 $SEED $CORES 4
        run_experiment_rsl 4 $SEED $CORES 4
        run_experiment_rsl 5 $SEED $CORES 4
    elif [ $EXPERIMENT = "fixed_sample" ]; then
        # run_experiment_rsl $SEED 1 $CORES 5
        run_experiment_rsl $SEED 2 $CORES 4
        run_experiment_rsl $SEED 3 $CORES 4
        run_experiment_rsl $SEED 4 $CORES 4
        run_experiment_rsl $SEED 5 $CORES 4
    elif [ $EXPERIMENT = "change_all" ]; then
        # run_experiment_rsl 1 1 $CORES 5
        run_experiment_rsl 2 2 $CORES 4
        run_experiment_rsl 3 3 $CORES 4
        run_experiment_rsl 4 4 $CORES 4
        run_experiment_rsl 5 5 $CORES 4
    fi
fi
