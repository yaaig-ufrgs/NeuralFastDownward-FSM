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


def rsl_sampling(
    out_dir,
    instance,
    numTrainStates,
    checkStateInvars,
    maxLenOfDemo,
    num_demos,
    seed,
    random_sample_percentage,
    regression_method,
    range_contrasting,
):

    ### (0) prepare
    startTime = time.perf_counter()
    print("> Sampling for {}".format(instance))
    print("> Seed is {}".format(seed))
    random.seed(seed)
    np.random.seed(seed)

    instance_split = instance.split("/")
    domain = instance_split[-2]
    instance_name = instance_split[-1].split(".")[0]
    instance_domain = "/".join(instance_split[:-1]) + "/domain.pddl"
    instance_mutexes = instance + ".mutexes"

    ### (1) simulator
    # Instantiates the "Simulator" and loads the mutexes for an instance.
    # A "Simulator" instance performs various functions imported from the "Tarski" library, and acts kind of
    # like a black box (but not really).
    env = Simulator(instance_domain, instance, None, seed)
    with open(instance_mutexes, "rb") as input:
        state_mutexes = pickle.load(input)


    ## (2) prepare mutex groups
    state_mutexes_for_environment = []
    # A mutex group is a group of variable/value pairs of which no two can be simultaneously true.
    for mutex_group in state_mutexes:
        mutex_group_set = set()
        # Add each atom [format: on(a,b), holding(i), etc.] of the mutex group to mutex_group_set.
        for atom in mutex_group:
            mutex_group_set.add(atom.replace(", ", ","))
        # Append the current mutex_group_set to the state_mutexes list.
        state_mutexes_for_environment.append(mutex_group_set)
    env.set_state_mutexes(state_mutexes_for_environment)

    ## (3) various initialization
    sampledStatesAll = []    # List of sampled states.
    sampledStateHeurAll = [] # List of estimated heuristics for each sampled state.

    # An "image" of a function is the set of all output values it may produce.
    # A "preimage" (inverse image) are a set of values that can produce an image (intuitive explanation).
    # e.g. f(x) = x^2, the preimage of {4} is {-2, 2}, i.e., the "Xs".

    # A state S_i-1 that can reach S_i is a "preimage" of S_i.
    allPlansPreimages = []
    maxPlanLength = 0
    startTime_get_demos = time.perf_counter()
    # print(env.atomToInt)

    ## (4) perform `num_demos` rollouts
    for demoNum in range(num_demos):
        # print("get demo {}".format(demoNum))
        env.reset_to_initial_state()
        # Gets a plan from a regression with a certain maximum_length and regression method (countBoth, countDel, count Add).
        # countAdd: count the number of atoms that are true in a partial state for the first time.
        # countDel: count the number of atoms that have become undetermined but were true for all previously generated partial states.
        # countBoth: countAdd + countDel.
        # randomWalk (none?): randomly select actions for which applying the regression aperator is valid. μ = 0.
        plan = env.get_random_regression_plan(maxLenOfDemo, regression_method)
        maxPlanLength = max(len(plan), maxPlanLength)

        ## (4.1) add the goal
        formula = copy.deepcopy(set(unwrap_conjunction_or_atom(env.problem.goal)))
        formulaInts = set()
        for atom in formula:
            # In the simulator, each atom has an integer corresponding to it, e.g. on(d,c) -> 36, on(a,g) -> 63
            formulaInts.add(env.atomToInt[str(atom)])

        # [({on(b,a), on(g,i), on(j,e), on(h,b), on(e,h), on(d,c), on(a,g), on(c,f), on(f,j)}, {0, 64, 98, 36, 5, 75, 44, 56, 63})]
        preImageFormulas = [(formula, formulaInts)]
        env.set_state_mutexes(state_mutexes_for_environment)

        ## (4.2) add all states in the plan
        for op in plan:
            # print(op)
            formula = env.preimage_set(copy.deepcopy(formula), op)
            formulaInts = set()
            for atom in formula:
                if str(atom) in env.atomToInt.keys():
                    #print(atom)
                    # else:
                    formulaInts.add(env.atomToInt[str(atom)])
            preImageFormulas.append((formula, formulaInts))

        ## (4.3) collect them all
        allPlansPreimages.append(preImageFormulas)

    endTime_get_demos = time.perf_counter()
    startTime_sample_states = time.perf_counter()

    maxLength = 0
    minLength = 9999

    ## (5) find maximum plan length and number of samples
    for preImagedSets in allPlansPreimages:
        if len(preImagedSets) < minLength:
            minLength = len(preImagedSets)
        if len(preImagedSets) > maxLength:
            maxLength = len(preImagedSets)

    # seems to make little sense to sample uniformly (MR)
    numTrainStatesPerPreimage = int(numTrainStates / maxLength)  # maxLength = number of pre image steps

    ## (6) go over all plans in increasing length (=heuristic value)
    heurValue = 0
    maxDemoLengthNotMet = True
    preImageSetsCurrent = []
    # preImageOrginialSets = []
    while maxDemoLengthNotMet:
        ## (6.1) append all demos sets together that have heur = to current heur
        maxDemoLengthNotMet = False
        for preImagedSets in allPlansPreimages:
            if len(preImagedSets) > heurValue:
                if not maxDemoLengthNotMet:
                    maxDemoLengthNotMet = True
                preImageSetsCurrent.append(preImagedSets[heurValue])

        if maxDemoLengthNotMet:
            # print("getting data for heur {}".format(heurValue))
            sampledStates = []
            sampledStateHeur = []
            maxTime1 = 0
            maxTime2 = 0
            maxTime3 = 0
            for state_num in range(numTrainStatesPerPreimage):  # X samples from each BDD
                startTime = time.perf_counter()
                # randHeurSet: {on(g,i), on(f,j), on(j,e), on(b,a), on(e,h), on(h,b), on(d,c), on(a,g), on(c,f)}
                # randHeurSetInts: {0, 64, 98, 36, 5, 75, 44, 56, 63}
                randHeurSet, randHeurSetInts = random.choice(preImageSetsCurrent)

                ## (6.2) select a random state (=no defined proposition) with probablity `random_sample_percentage`
                if (random.random() > random_sample_percentage / 100):
                    assignedIndexes = list(randHeurSetInts)
                else:
                    assignedIndexes = []

                state = np.random.randint(0, 2, size=env.numAtoms) # Random binary state.

                for atom_indx in assignedIndexes:
                    state[atom_indx] = 1
                maxTime1 += time.perf_counter() - startTime
                startTime = time.perf_counter()

                # A state invariant is a logical formula over the fluents of a state that must hold
                # in every state that may belong to a solution path (Alcázar and Torralba, 2015).
                if checkStateInvars:
                    trueAtoms = np.nonzero(state)[0] # Return the indices of the elements that are non-zero.
                    numGroups = env.state_mutexes.shape[0] # e.g. 32
                    groupOrder = list(range(numGroups)) # e.g. [0, 1, 2, ..., 31]
                    random.shuffle(groupOrder)

                    # I think this is for getting the mutexes in the for below.
                    intersections = np.hstack( # Stack arrays in sequence horizontally (column wise).
                        (
                            env.state_mutexes,
                            np.tile(trueAtoms, (env.state_mutexes.shape[0])).reshape(
                                (env.state_mutexes.shape[0], trueAtoms.shape[0])
                            ),
                        )
                    )
                    intersections.sort(axis=1)

                    indexOfIntersection = intersections[:, 1:] == intersections[:, :-1]
                    allnegOnes = intersections[:, 1:] != -1
                    indexOfIntersection = allnegOnes & indexOfIntersection

                    for mutexListIndx in groupOrder:
                        mutexList = intersections[mutexListIndx][:-1][indexOfIntersection[mutexListIndx]]
                        # np.intersect1d(env.state_mutexes[mutexListIndx], trueAtoms, assume_unique = True)
                        ## (6.3.1) if a proposition in this mutex group has been assigned keep it (precisely: the first assigned proposition); otherwise: select a random one
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

                # "RSL uses the tightest upper bound found for the state's goal distance derived through the
                # partial states visited by a number of regressions. As RSL searches and labels goal distance
                # estimates over partial states, it samples many different training states from each partial state."

                sampledStates.append(state) # Can just give heuristic from preimage sampled from as an upper bound

                # labeled heuristic = distance to the closest state set in R to the goal according to the regression.
                sampledStateHeur.append(heurValue)

            sampledStatesAll.extend(sampledStates)
            sampledStateHeurAll.extend(sampledStateHeur)

            # heuristic value is incremented on each while iteration, until len(preImagedSets) <= heurValue:
            heurValue += 1

            maxTime3 += time.perf_counter() - startTime
            startTime = time.perf_counter()
        # print("a {} b {} c {}".format(maxTime1, maxTime2, maxTime3))
    # print("got data now checking what label")
    # Check which preimage set sampled state is in starting from goal backwards
    endTime_sample_states = time.perf_counter()
    startTime_check_state_membership = time.perf_counter()

    sampledStateHeurAll = []
    numberChecked = 0
    # print("checking membership")
    for state in sampledStatesAll:
        numberChecked += 1
        heur = 0
        foundPreImage = False
        setOfStateIndxs = set(state.nonzero()[0])
        while not foundPreImage and heur < maxLength:
            for preImagedSets in allPlansPreimages:
                if len(preImagedSets) > heur and preImagedSets[heur][1].issubset(
                    setOfStateIndxs
                ):
                    sampledStateHeurAll.append(heur)
                    foundPreImage = True
                    break
            heur += 1

        if (
            not foundPreImage
        ):  # if not in any preimage just assign highest heur value (highest preimage heur + 1)
            if range_contrasting:
                rnd_x = random.randint(minLength, heur)
                sampledStateHeurAll.append(rnd_x)
            else:
                sampledStateHeurAll.append(heur)
        if numberChecked % 100 == 0:
            pass
            # print(numberChecked)

    gc.collect()
    endTime_check_state_membership = time.perf_counter()
    startTime_check_train_NN = time.perf_counter()

    sampling_filename = (
        f"rsl_{domain}_{instance_name}_{regression_method}_{numTrainStates}_ss{seed}"
    )
    if out_dir[-1] != "/":
        out_dir += "/"

    save_sampling(sampling_filename, out_dir, sampledStatesAll, sampledStateHeurAll)
    save_facts_order_and_default_values(sampling_filename, out_dir, env.getGroundedDicts())


def save_sampling(sampling_filename, out_dir, sampledStatesAll, sampledStateHeurAll):
    out_file = out_dir + sampling_filename
    print(f"> Saving sampled states to {out_file}")
    with open(out_file, "w") as f:
        f.write("#cost;state\n")
        for i in range(len(sampledStatesAll)):
            f.write(
                "%s;%s\n"
                % (sampledStateHeurAll[i], "".join(map(str, sampledStatesAll[i])))
            )


def save_facts_order_and_default_values(filename, out_dir, env):
    string_atom_order = ""
    string_defaults = ""

    for key, item in enumerate(env[1]):
        string_atom_order += "Atom " + item + ";"
        string_defaults += "0;"

    string_atom_order = string_atom_order[:-1]
    string_defaults = string_defaults[:-1]
    string_atom_order += ""
    string_defaults += ""

    out_file_facts = out_dir + filename + "_facts.txt"
    out_file_defaults = out_dir + filename + "_defaults.txt"
    with open(out_file_facts, "w") as f:
        f.write(string_atom_order)
    with open(out_file_defaults, "w") as text_file:
        text_file.write(string_defaults)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    if not fix_seed_and_possibly_rerun():
        parser = argparse.ArgumentParser()

        parser.add_argument("--out_dir", default="samples")
        parser.add_argument("--instance", default=None)
        parser.add_argument("--one_step_method", default=None)
        parser.add_argument("--num_train_states", type=int, default=10000) # Nt
        parser.add_argument("--check_state_invars", type=str2bool, nargs="?", const=True, default=False)
        parser.add_argument("--num_demos", type=int, default=1) # Nr
        parser.add_argument("--max_len_demo", type=int, default=1) # L
        parser.add_argument("--seed", type=int, default=1)
        parser.add_argument("--random_sample_percentage", type=int, default=0) # Pr
        parser.add_argument("--regression_method", default=None)
        parser.add_argument("--range_contrasting", type=str2bool, nargs="?", const=True, default=False)

        args = parser.parse_args()
        rsl_sampling(
            args.out_dir,
            args.instance,
            args.num_train_states,
            args.check_state_invars,
            args.max_len_demo,
            args.num_demos,
            args.seed,
            args.random_sample_percentage,
            args.regression_method,
            args.range_contrasting,
        )
