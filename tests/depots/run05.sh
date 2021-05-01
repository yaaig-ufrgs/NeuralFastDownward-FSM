#!/bin/sh

# Enter a value between 1 and 200 to select the task to attempt to solve.

MODEL="05_ocls_ns_ubal_h3_sigmoid_inter_gen_sat_drp0_Kall_pruneOff_9_fold_model.pb"
DOMAIN="../../tasks/depot_fix_goals/depot_p05/domain.pddl"
PROBLEM="../../tasks/depot_fix_goals/depot_p05/p$1.pddl"

../../fast-downward.py --overall-time-limit 30m --overall-memory-limit 3584M --build debug $PROBLEM \
  --search "eager_greedy([nh(network=sgnet(path=$MODEL,type=classification,unary_threshold=0.00001,state_layer=input_1_1,goal_layer=input_2_1,output_layers=[dense_4_1/Sigmoid],atoms={PDDL_ATOMS_FLEXIBLE},defaults={PDDL_INITS_FLEXIBLE},batch_size=50))],cost_type=one)"
