#!/usr/bin/env python3

"""
Create pddl files from sampling output.

Usage: ./bool2pddl.py sas_plan pddl_base
    sas_plan = Sampling output file
    pddl_base = PDDL used to generate the samples
"""

from sys import argv

with open(argv[1],) as sas_file:
    sas_lines = sas_file.readlines()
with open(argv[2],) as pddl_base:
    pddl = pddl_base.read()

# <State> is on the 3rd line
atoms = sas_lines[2].split(";")
atoms[0] = atoms[0].split("=")[1] # remove #<State>=
atoms[-1] = atoms[-1][:-1] # remove \n

# Convert "Atom ontable(a)" to "ontable a"
atoms = [x[5:] for x in atoms] # remove "Atom " prefix

for i in range(len(atoms)):
    pred, objs = atoms[i].split("(")
    objs = objs[:-1].replace(",", "")
    # print(f"{atoms[i]} -> {pred} {objs}")
    atoms[i] = pred+" "+objs

problem_name = pddl.split("(problem ")[1].split(")")[0]
sample_id = 0
for i in range(5, len(sas_lines)-1):
    # If it is the last state of the sample
    if sas_lines[i+1][0] == "#":
        values = sas_lines[i].split(";")[1:len(atoms)+1]

        # Get init from boolean values
        init = []
        for i, v in enumerate(values):
            if v == "1":
                init.append(atoms[i])

        # Create the new pddl by changing the initial state of pddl_base
        new_problem_name = problem_name+"-s"+str(sample_id)
        sample_id += 1

        pddl1 = pddl.split(":INIT" if ":INIT" in pddl else ":init")[0]
        pddl1 = pddl1.replace(problem_name, new_problem_name)
        pddl2 = ":init"
        for i in init:
            pddl2 += "\n\t("+i+")"
        pddl2 += ")\n"
        pddl3 = "(:goal" + pddl.split(":goal" if ":goal" in pddl else ":GOAL")[1]

        with open(new_problem_name+".pddl", "w") as output:
            output.write(pddl1 + pddl2 + pddl3)
