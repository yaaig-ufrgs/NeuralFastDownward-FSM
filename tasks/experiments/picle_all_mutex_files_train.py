#!/usr/bin/env python3

import os
import pickle

def getMutexGroups(file):
    with open(file) as f:
        lines = f.readlines()
        startedVariable = False
        skippedLines = 0
        mutexGroups = []
        mutexGroup = []
        for line in lines:
            if line == "begin_variable\n":
                # print(line)
                startedVariable = True
            elif line == "end_variable\n":
                skippedLines = 0
                # print(mutexGroup)
                mutexGroups.append(mutexGroup)
                mutexGroup = []
                # print(line)
                startedVariable = False
            elif startedVariable:
                if skippedLines < 3:
                    skippedLines += 1
                else:
                    mutexGroup.append(line.replace("\n", "").replace("Atom ", ""))

        # Get the additional mutex grouops
        startedMutex = False
        for line in lines:
            if line == "begin_mutex_group\n":
                startedMutex = True
            elif line == "end_mutex_group\n":
                skippedLines = 0
                # print(mutexGroup)
                mutexGroups.append(mutexGroup)
                mutexGroup = []
                # print(line)
                startedMutex = False
            elif startedMutex:
                if skippedLines < 1:
                    skippedLines += 1
                else:
                    group, indx = line.replace("\n", "").split(" ")
                    mutexGroup.append(mutexGroups[int(group)][int(indx)])

        for i in range(len(mutexGroups)):
            for j in range(len(mutexGroups[i])):
                if "Negated" in mutexGroups[i][j]:
                    mutexGroups[i][j] = "<none of those>"
    return mutexGroups

for domainName in os.listdir():
    print(domainName)
    if os.path.isdir("{}".format(domainName)):
        for problemName in os.listdir("{}/".format(domainName)):
            print("problem name is {}".format(problemName))
            if ".pddl" in problemName and "domain" not in problemName and ".sas" not in problemName and ".mutexes" not in problemName and ".json" not in problemName:
                mutexGroups = getMutexGroups("{}/{}.sas".format(domainName, problemName))
                with open('{}/{}.mutexes'.format(domainName, problemName), 'w') as output:
                    # pickle.dump(mutexGroups, output, pickle.HIGHEST_PROTOCOL)
                    output.write("\n".join([str(x) for x in mutexGroups])+"\n")