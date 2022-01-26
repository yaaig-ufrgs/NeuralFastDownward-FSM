#!/usr/bin/env python3

"""
Convert a boolean to state atoms.

Usage: ./boolean2state.py facts boolean
  e.g. ./boolean2state.py "Atom holding(e);Atom on(e, a);Atom..." "0010..."
"""

from sys import argv

if argv[1][-1] == ";":
    argv[1] = argv[1][:-1]
if argv[1][8] == "=":
    argv[1] = argv[1].split("=", 1)[1]
atoms = argv[1].split(";")
boolean = argv[2]

assert len(atoms) == len(boolean)
state = []
for i in range(len(boolean)):
    if boolean[i] == "1":
        state.append(atoms[i])

for s in state:
    s_ = s.split("Atom ")[1].replace("()","").replace("("," ").replace(",","").replace(")","")
    print(f"({s_})", end=" ")
print()
