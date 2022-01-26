#!/usr/bin/env python3

"""
Create reference file from result folders.

usage: ./get_references.py expansions_secs train_folders
  e.g. ./get_references.py 600 ../results/*
"""

from sys import argv
from json import load

expansions_secs = int(argv[1])
print("domain,problem,epochs,expansions")
for folder in argv[2:]:
    try:
        with open(folder+"/train_args.json",) as f:
            data = load(f)
            domain = data["domain"]
            problem = data["problem"]
        with open(folder+"/nfd.log",) as f:
            lines = f.readlines()
            epochs = ""
            for line in lines:
                if " | avg_train_loss=" in line:
                   epochs = line.split("Epoch ")[1].split(" |")[0]
        with open(folder+"/tests/nfd_test/test_results.json",) as f:
            data = load(f)
            expansions = int(data["statistics"]["traced_0.pt"]["avg_expansion_rate"] * expansions_secs)
        print(domain, problem, epochs, expansions, sep=",")
    except Exception as e:
        print(f"error ({folder}): {e}")
        continue
