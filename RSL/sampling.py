#!/usr/bin/env python3

import numpy as np
import copy
from simulator import Simulator
from tarski.syntax.formulas import unwrap_conjunction_or_atom, land
import pickle
import argparse
from sklearn.model_selection import train_test_split
import time
import gc
import random
from util import fix_seed_and_possibly_rerun

def rsl_sampling(out_dir, instance, numTrainStates, checkStateInvars, trial, maxLenOfDemo, num_demos, seed, random_sample_percentage, regression_method):
    startTime = time.perf_counter()
    print("seed is {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)

    instance_split = instance.split('/')
    instance_name = instance_split[-1].split('.')[0]
    instance_domain = instance_split[-2] + "/domain.pddl"
    instance_mutexes = instance + ".mutexes"

    env = Simulator(instance_domain, instance, None, seed)
    with open('blocks/blocks_probBLOCKS-4-0.pddl.mutexes', 'rb') as input:
        state_mutexes = pickle.load(input)

    state_mutexes_for_environment = []
    for mutex_group in state_mutexes:
        mutex_group_set = set()
        for atom in mutex_group:
            mutex_group_set.add(atom.replace(", ", ","))
        state_mutexes_for_environment.append(mutex_group_set)
    env.set_state_mutexes(state_mutexes_for_environment)

    sampledStatesAll = []
    sampledStateHeurAll = []

    allPlansPreimages = []
    maxPlanLength = 0
    startTime_get_demos = time.perf_counter()
    print(env.atomToInt)
    for demoNum in range(num_demos):
        print("get demo {}".format(demoNum))
        env.reset_to_initial_state()
        plan = env.get_random_regression_plan(maxLenOfDemo, regression_method)
        maxPlanLength = max(len(plan), maxPlanLength)

        formula = copy.deepcopy(set(unwrap_conjunction_or_atom(env.problem.goal)))
        formulaInts = set()
        for atom in formula:
            formulaInts.add(env.atomToInt[str(atom)])

        preImageFormulas = [(formula, formulaInts)]
        env.set_state_mutexes(state_mutexes_for_environment)

        for op in plan:
            print(op)
            formula = env.preimage_set(copy.deepcopy(formula), op)
            formulaInts = set()
            for atom in formula:
                if str(atom) in env.atomToInt.keys():
                    #print(atom)
                #else:
                    formulaInts.add(env.atomToInt[str(atom)])
            preImageFormulas.append((formula, formulaInts))
        allPlansPreimages.append(preImageFormulas)
    endTime_get_demos = time.perf_counter()
    startTime_sample_states = time.perf_counter()
    maxLength = 0
    for preImagedSets in allPlansPreimages:
        if len(preImagedSets) > maxLength:
            maxLength = len(preImagedSets)

    numTrainStatesPerPreimage = int(numTrainStates / maxLength)  # maxLength = number of pre image steps
    print(numTrainStatesPerPreimage)

    heurValue = 0
    maxDemoLengthNotMet = True
    preImageSetsCurrent = []
    #preImageOrginialSets = []
    while maxDemoLengthNotMet:
        # append all demos sets together that have heur = to current heur
        maxDemoLengthNotMet = False
        for preImagedSets in allPlansPreimages:
            if len(preImagedSets) > heurValue:
                if not maxDemoLengthNotMet:
                    maxDemoLengthNotMet = True
                preImageSetsCurrent.append(preImagedSets[heurValue])

        if maxDemoLengthNotMet:
            print("getting data for heur {}".format(heurValue))
            sampledStates = []
            sampledStateHeur = []
            maxTime1 = 0
            maxTime2 = 0
            maxTime3 = 0
            for state_num in range(numTrainStatesPerPreimage):  # X samples from each BDD
                startTime = time.perf_counter()
                randHeurSet, randHeurSetInts  = random.choice(preImageSetsCurrent)
                if random.random() > random_sample_percentage/100: # sample random every second data point
                    assignedIndexes = list(randHeurSetInts)
                else:
                    assignedIndexes = []

                state = np.random.randint(0, 2, size=env.numAtoms)

                for atom_indx in assignedIndexes:
                    state[atom_indx] = 1
                maxTime1 += time.perf_counter() - startTime
                startTime = time.perf_counter()

                if checkStateInvars:
                    trueAtoms = np.nonzero(state)[0]
                    numGroups = env.state_mutexes.shape[0]
                    groupOrder = list(range(numGroups))
                    random.shuffle(groupOrder)
                    intersections = np.hstack((env.state_mutexes, np.tile(trueAtoms, (env.state_mutexes.shape[0])).reshape((env.state_mutexes.shape[0], trueAtoms.shape[0]))))
                    intersections.sort(axis=1)

                    indexOfIntersection = intersections[:, 1:] == intersections[:, :-1]
                    allnegOnes = intersections[:, 1:] != -1
                    indexOfIntersection = allnegOnes & indexOfIntersection

                    for mutexListIndx in groupOrder:
                        mutexList = intersections[mutexListIndx][:-1][indexOfIntersection[mutexListIndx]]#np.intersect1d(env.state_mutexes[mutexListIndx], trueAtoms, assume_unique = True)

                        if len(mutexList) > 1:
                            assignedOne = False
                            areOnes = []
                            for atomIndx in mutexList:
                                if state[atomIndx] == 1:
                                    areOnes.append(atomIndx)
                                    if not assignedOne and atomIndx in assignedIndexes:
                                        assignedOne = True
                                    else:
                                        state[atomIndx] = 0
                            if not assignedOne and len(areOnes) > 0:
                                state[random.choice(areOnes)] = 1
                maxTime2 += time.perf_counter() - startTime
                startTime = time.perf_counter()
                sampledStates.append(state)  # Can just give heuristic from preimage sampled from as an upper bound
                sampledStateHeur.append(heurValue)

            sampledStatesAll.extend(sampledStates)
            sampledStateHeurAll.extend(sampledStateHeur)
            heurValue += 1
            maxTime3 += time.perf_counter() - startTime
            startTime = time.perf_counter()
        #print("a {} b {} c {}".format(maxTime1, maxTime2, maxTime3))
    print("got data now checking what label")
    # Check which preimage set sampled state is in starting from goal backwards
    endTime_sample_states = time.perf_counter()
    startTime_check_state_membership = time.perf_counter()

    sampledStateHeurAll = []
    numberChecked = 0
    print("checking membership")
    for state in sampledStatesAll:
        numberChecked += 1
        heur = 0
        foundPreImage = False
        setOfStateIndxs = set(state.nonzero()[0])
        while not foundPreImage and heur < maxLength:
            for preImagedSets in allPlansPreimages:
                if len(preImagedSets) > heur and preImagedSets[heur][1].issubset(setOfStateIndxs):
                    sampledStateHeurAll.append(heur)
                    foundPreImage = True
                    break
            heur += 1

        if not foundPreImage: # if not in any preimage just assign highest heur value (highest preimage heur + 1)
            sampledStateHeurAll.append(heur)
        if numberChecked % 100 == 0:
            print(numberChecked)

    gc.collect()
    endTime_check_state_membership = time.perf_counter()
    startTime_check_train_NN = time.perf_counter()

    print("len states: ", len(sampledStatesAll))
    print("len heurs:  ", len(sampledStateHeurAll))

    print(sampledStatesAll[0])
    print(sampledStateHeurAll[0])

    sample_filename = f"rsl_{instance_name}_{regression_method}_{numTrainStates}_ss{seed}"
    csv_file = out_dir+"/"+sample_filename
    print(f"Saving sampled states to {csv_file}")
    with open(csv_file, "w") as f:
        f.write("#cost;state")
        for i in range(len(sampledStatesAll)):
            f.write("%s;%s\n" % (sampledStateHeurAll[i], ''.join(map(str, sampledStatesAll[i]))))

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    if not fix_seed_and_possibly_rerun():
        parser = argparse.ArgumentParser()

        parser.add_argument('--out_dir', default="samples")
        parser.add_argument('--instance', default=None)
        parser.add_argument('--one_step_method', default=None)
        parser.add_argument('--num_train_states', type=int, default=10000)
        parser.add_argument('--check_state_invars', type=str2bool, nargs='?',
                        const=True, default=False)
        parser.add_argument('--trialNum', type=int, default=0)
        parser.add_argument('--num_demos', type=int, default=1)
        parser.add_argument('--maxLenOfDemo', type=int, default=1)
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--random_sample_percentage', type=int, default=0)
        parser.add_argument('--regression_method', default=None)

        args = parser.parse_args()
        #rsl_sampling(args.out_dir, args.instance, args.num_train_states, args.check_state_invars, args.trialNum, args.maxLenOfDemo, args.num_demos, args.seed, args.random_sample_percentage, args.regression_method)
        rsl_sampling("samples", "blocks/blocks_probBLOCKS-4-0.pddl", 100000, True, 0, 200, 1, 2, 50, "countBoth")
