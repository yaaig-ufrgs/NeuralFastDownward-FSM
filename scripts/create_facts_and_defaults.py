#!/usr/bin/env python3

"""
Create the facts and defaults from pddl. Save in the same folder as pddl.

Usage: ./create_facts_and_defaults.py [--facts] [--defaults] pddls
  e.g. ./create_facts_and_defaults.py --facts ../tasks/ferber21/training_tasks_used/*/*/*.pddl
       ./create_facts_and_defaults.py --defaults ../tasks/ferber21/test_states/*/*/*/*.pddl
"""

from sys import argv
from subprocess import check_output, CalledProcessError
from os import remove

_FD = "../fast-downward.py"

FACTS_FILENAME_FORMAT = "{problem_name}_facts.txt"
DEFAULTS_FILENAME_FORMAT = "{problem_name}_defaults.txt"

def get_facts(pddl: str) -> [str]:
    SAS_FILE = "output.sas"
    PLAN_FILE = "sas_plan"

    # e.g. ../fast-downward.py --sas-file output.sas --plan-file sas_plan ../tasks/IPC/blocks/probBLOCKS-10-0.pddl
    # --search "sampling_search_yaaig(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()),
    # techniques=[gbackward_yaaig(searches=1, samples_per_search=1, technique=yaaig)])"
    cl = [
        _FD,
        "--sas-file",
        SAS_FILE,
        "--plan-file",
        PLAN_FILE,
        pddl,
        "--search",
        "sampling_search_yaaig(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()), "
        "techniques=[gbackward_yaaig(searches=1, samples_per_search=1, technique=yaaig)])",
    ]

    try:
        returncode = 0
        output = check_output(cl)
    except CalledProcessError as e:
        returncode = e.returncode
        output = e.output.decode("utf-8")

    if returncode != 12:
        raise Exception("get_facts: Return code should be 12")
    if "Generated Entries: 1" not in output:
        raise Exception("get_facts: Sample was not generated.")
    
    facts = []
    # PLAN_FILE format:
    #    #<PlanCost>=single integer value
    #    #<State>=Atom holding(i);Atom on(i, a);...;Atom ontable(j);
    #    0;0000100...
    with open(PLAN_FILE, "r") as f:
        facts = f.readlines()[1][len("#<State>="):-len(";\n")].split(";")

    remove(SAS_FILE)
    remove(PLAN_FILE)

    if not facts:
        raise Exception("get_facts: facts is empty")
    return facts

def get_defaults(pddl: str, facts) -> [int]:
    init = None
    with open(pddl, "r") as f:
        pddl_text = f.read().lower()
        init = pddl_text.split(":init")[1].split(":goal")[0]

    # Atom on(i, a) -> (on i a)
    modified_facts = []
    for fact in facts:
        f = fact.replace("Atom ", "")                   # Atom on(i, a) -> on(i, a)
        f = f.replace(", ", ",").replace(",", " ")       # on(i, a) -> on(i a)
        f = f"({f.split('(')[0]} {f.split('(')[1]}"   # on(i a) -> (on i a)
        f = f.replace(" )", ")") # facts without objects (handempty ) -> (handempty)
        modified_facts.append(f)

    defaults = []
    for fact in modified_facts:
        value = "1" if fact in init else "0"
        defaults.append(value)

    if not defaults:
        raise Exception("get_defaults: defaults is empty")
    return defaults

if __name__ == "__main__":
    facts_on, defaults_on = False, False
    for arg in argv[1:min(len(argv), 3)]:
        if arg == "--facts":
            facts_on = True
        elif arg == "--defaults":
            defaults_on = True
    if not facts_on and not defaults_on:
        raise Exception("Use --facts arg to create facts file and/or --defaults for defaults file")

    for pddl in argv[1:]:
        nono = False
        for a in "blocks depot grid npuzzle".split(" "):
            if a in pddl:
                nono = True
        if nono:
            continue
        if pddl.split("/")[-1] == "domain.pddl" or pddl[:2] == "--":
            continue

        facts = get_facts(pddl)
        if facts_on:
            with open(FACTS_FILENAME_FORMAT.format(problem_name=pddl.replace(".pddl", "")), "w") as f:
                f.write(";".join(facts) + "\n")
        if defaults_on:
            defaults = get_defaults(pddl, facts)
            with open(DEFAULTS_FILENAME_FORMAT.format(problem_name=pddl.replace(".pddl", "")), "w") as f:
                f.write(";".join(defaults) + "\n")
