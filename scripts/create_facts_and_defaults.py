#!/usr/bin/env python3

"""
Create the facts and defaults from pddl.
Save in the same folder as pddl.

usage: ./create_facts_and_defaults.py pddl facts
       ./create_facts_and_defaults.py ../tasks/ferber21/training_tasks_used/*/*/*.pddl
"""

from sys import argv
from subprocess import check_output, CalledProcessError
from os import remove

_FD = "../fast-downward.py"

FACTS_FILENAME_FORMAT = "{problem_pddl}_facts.txt"
DEFAULTS_FILENAME_FORMAT = "{problem_pddl}_defaults.txt"

def get_facts(pddl: str) -> [str]:
    SAS_FILE = "output.sas"
    PLAN_FILE = "sas_plan"

    # e.g. ../fast-downward.py --sas-file output.sas --plan-file sas_plan ../tasks/IPC/blocks/probBLOCKS-10-0.pddl
    # --search "sampling_search_fukunaga(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()),
    # techniques=[gbackward_fukunaga(searches=1, samples_per_search=1, technique=fukunaga)])"
    cl = [
        _FD,
        "--sas-file",
        SAS_FILE,
        "--plan-file",
        PLAN_FILE,
        pddl,
        "--search",
        "sampling_search_fukunaga(astar(lmcut(transform=sampling_transform()), transform=sampling_transform()), "
        "techniques=[gbackward_fukunaga(searches=1, samples_per_search=1, technique=fukunaga)])",
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

def get_facts_and_defaults(pddl: str) -> ([str], [int]):
    facts = get_facts(pddl)
    defaults = get_defaults(pddl, facts)
    return facts, defaults

if __name__ == "__main__":
    for pddl in argv[1:]:
        if pddl.split("/")[-1] == "domain.pddl":
            continue
        facts, defaults = get_facts_and_defaults(pddl)
        with open(FACTS_FILENAME_FORMAT.format(problem_pddl=pddl), "w") as f:
            f.write(";".join(facts) + "\n")
        with open(DEFAULTS_FILENAME_FORMAT.format(problem_pddl=pddl), "w") as f:
            f.write(";".join(defaults) + "\n")
