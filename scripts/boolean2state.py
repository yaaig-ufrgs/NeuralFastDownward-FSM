#!/usr/bin/env python3

"""
Convert a boolean to state atoms.

Usage: ./boolean2state.py facts boolean
  e.g. ./boolean2state.py "Atom holding(e);Atom on(e, a);Atom..." "0010..."
OR
Usage: ./boolean2state.py samples_file
  e.g. ./boolean2state.py ../samples/yaaig_blocks_probBLOCKS-7-0_rw_ps_500x200_ss0
"""

from sys import argv

def bool2state(facts, boolean):
    atoms = [f for f in facts.split("=")[-1].replace("Atom ", "").split(";") if f]
    assert len(atoms) == len(boolean)
    state = ["("+atoms[i].replace("(", " ").replace(",", " ").replace("  ", " ") for i, x in enumerate(boolean) if x == "1"]
    return " ".join(state)

if __name__ == "__main__":
    if len(argv) == 2:
        with open(argv[1], "r") as f:
            lines = f.readlines()
            assert lines[1][:len("#<State>=")] == "#<State>="
            states = [bool2state(lines[1], l.split(";")[1]) for l in lines if l[0] != "#"]
            print("\n".join(states))
    elif len(argv) == 3:
        print(bool2state(argv[1], argv[2]))
    else:
        raise Exception("invalid arguments")
