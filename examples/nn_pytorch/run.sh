#!/bin/sh

../../fast-downward.py --build debug instances/blocks-domain.pddl instances/blocks-b4-01.pddl --search "astar(nh(torch_sampling_network(path=traced.pt,blind=false,unary_threshold=0.01)))"
#../../fast-downward.py --build debug instances/gripper-domain.pddl instances/gripper-prob01.pddl --search "astar(nh(torch_sampling_network(path=traced.pt,blind=false,unary_threshold=0.01)))"
#../../fast-downward.py --build debug instances/npuzzle-domain.pddl instances/n-puzzle-3x3-s44.pddl --search "astar(nh(torch_sampling_network(path=traced.pt,blind=false,unary_threshold=0.01)))"
