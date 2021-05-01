#!/bin/sh

# Enter a value between 1 and 200 to select the task to attempt to solve.

# This test in particular takes a ton of time (more than 30 minutes) to solve,
# because it represents the "hardest" blocksworld problems.

MODEL="17_ocls_ns_ubal_h3_sigmoid_inter_gen_sat_drp0_Kall_pruneOff_9_fold_model.pb"
DOMAIN="../../tasks/blocksworld_ipc/probBLOCKS-17-0/domain.pddl"
PROBLEM="../../tasks/blocksworld_ipc/probBLOCKS-17-0/p$1.pddl"

#../../fast-downward.py --build debug $PROBLEM \
#  --search "eager_greedy([nh(network=sgnet(type=classification, path=${MODEL}, state_layer=input_1_1, \
#  goal_layer=input_2_1, output_layers=[dense_4_1/Sigmoid]))], cost_type=ONE)"

../../fast-downward.py --overall-time-limit 30m --overall-memory-limit 3584M --build debug $PROBLEM \
  --search "eager_greedy([nh(network=sgnet(path=$MODEL,type=classification,unary_threshold=0.00001,state_layer=input_1_1,goal_layer=input_2_1,output_layers=[dense_4_1/Sigmoid],atoms={PDDL_ATOMS_FLEXIBLE},defaults={PDDL_INITS_FLEXIBLE},batch_size=50))],cost_type=one)"
