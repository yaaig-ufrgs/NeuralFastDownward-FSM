#!/bin/bash

TASKS=( \
    blocksworld_ipc/probBLOCKS-12-0 \
    depot_fix_goals/depot_p05 \
    grid_fix_goals/grid_prob03 \
    npuzzle_ipc/npuzzle_prob_n6_1 \
    pipesworld-notankage_fix_goals/pipes_nt_p19-net2-b18-g6 \
    rovers/rovers_p11 \
    scanalyzer-opt11-strips/scanalyzer11_p07 \
    storage/storage_p18 \
    transport-opt14-strips/transport_p05 \
    visitall-opt14-strips/visitall_p-1-12
)

for task in "${TASKS[@]}"; do
    MODEL="../../models/$task/ocls_ns_ubal_h3_sigmoid_inter_gen_sat_drp0_Kall_pruneOff_9_fold_model.pb"
    OUTPUT=output/$task
    for i in $(seq 1 20); do
        PROBLEM="../../tasks/$task/p$i.pddl"
        ../../fast-downward.py --overall-time-limit 30m --overall-memory-limit 3584M --build debug $PROBLEM \
            --search "eager_greedy([nh(network=sgnet(path=$MODEL,type=classification,unary_threshold=0.00001,state_layer=input_1_1,goal_layer=input_2_1,output_layers=[dense_4_1/Sigmoid],atoms={PDDL_ATOMS_FLEXIBLE},defaults={PDDL_INITS_FLEXIBLE},batch_size=50))],cost_type=one)" \
            >> $OUTPUT/p$i.log
        echo >> $OUTPUT/p$i.log

        if [ -f sas_plan ]; then
            mv sas_plan $OUTPUT/p${i}_sas_plan
        fi
    done
done
