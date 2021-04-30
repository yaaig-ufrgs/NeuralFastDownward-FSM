#!/bin/sh

# Enter a value between 1 and 200 to select the task to attempt to solve.

MODEL="17_ocls_ns_ubal_h3_sigmoid_inter_gen_sat_drp0_Kall_pruneOff_9_fold_model.pb"
DOMAIN="../../tasks/blocksworld_ipc/probBLOCKS-17-0/domain.pddl"
PROBLEM="../../tasks/blocksworld_ipc/probBLOCKS-17-0/p$1.pddl"

../../fast-downward.py --build debug $PROBLEM \
  --search "eager_greedy([nh(network=snet(type=classification, path=${MODEL}, state_layer=dense_3_input, \
  output_layers=[output_layer/Softmax]))], cost_type=ONE)"

# From src/search/neural_networks/state_network.cc
# state_layer: "Name of the input layer in the computation graph to insert the current state."

# The error seems to be on the input data. How to discover the name of input layer in the trained model?
