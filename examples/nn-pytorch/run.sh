#!/bin/sh

../../fast-downward.py --build debug blocks-domain.pddl blocks-b4-01.pddl --search "astar(nh(torch_sampling_network(path=traced.pt)))"
