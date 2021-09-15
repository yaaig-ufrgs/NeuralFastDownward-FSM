#!/bin/bash
../../fast-downward.py --build debug ~/Home/downward-benchmarks/gripper/prob01.pddl --search "astar(nh(ttest(path=traced.pt)))"

