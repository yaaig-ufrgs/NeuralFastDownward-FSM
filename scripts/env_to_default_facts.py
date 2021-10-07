#!/usr/bin/env python3

"""
Splits *.env files to *_defaults.txt and *_facts.txt files.

Usage: ./env_to_default_facts.py <directory>
"""

import sys
import os
import pickle
from glob import glob


def gen_default_facts(data):
    for d in data:
        d_split = d.split('/')
        directory = '/'.join(d_split[:-1])
        file = d_split[-1]
        problem = file.split('-')
        #print(problem[0])
        if problem[0] == 'blocks':
            problem_name = "blocks_probBLOCKS-" + '-'.join(problem[2:])[:-4]

        else:
            problem_name = file[:-4]

        technique = d_split[-2].split('_')[-1][2:]
        num_states = d_split[-2].split('_')[1][2:]
        filename = f"rsl_{problem_name}_{technique}_{num_states}_ss1337"
        print(filename)

        with open(d, 'rb') as fileinput:
            env = pickle.load(fileinput)
            save_facts_order_and_default_values(filename, directory+'/', env)


def save_facts_order_and_default_values(filename, out_dir, env):
    string_atom_order = ""
    string_defaults = ""

    for _key, item in enumerate(env[1]):
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



rsl_dir = sys.argv[1]
if rsl_dir[-1] != '/':
        rsl_dir += '/'


blocks = glob(rsl_dir+"*blocks*.env")
visitall = glob(rsl_dir+"*visitall*.env")
transport = glob(rsl_dir+"*transport*.env")
storage = glob(rsl_dir+"*storage*.env")
scanalyzer = glob(rsl_dir+"*scanalyzer*.env")
rovers = glob(rsl_dir+"*rovers*.env")
pipesworld = glob(rsl_dir+"*pipesworld*.env")
npuzzle = glob(rsl_dir+"*npuzzle*.env")
grid = glob(rsl_dir+"*grid*.env")
depot = glob(rsl_dir+"*depot*.env")

gen_default_facts(blocks)
gen_default_facts(visitall)
gen_default_facts(transport)
gen_default_facts(storage)
gen_default_facts(scanalyzer)
gen_default_facts(rovers)
gen_default_facts(pipesworld)
gen_default_facts(npuzzle)
gen_default_facts(grid)
gen_default_facts(depot)

#blocksdat = glob(rsl_dir+"*blocks*.dat")
#with open(blocksdat[0], 'rb') as fileinput:
#    dat = pickle.load(fileinput)
#print(dat)
