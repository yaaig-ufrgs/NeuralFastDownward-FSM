#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import re


PATTERN_VARIABLE = re.compile(r"begin_variable.*?end_variable", re.DOTALL)
PATTERN_STATE = re.compile(r"begin_state.*?end_state", re.DOTALL)


def get_block_range(content, keys, start=0, inc_level=["("], dec_level=[")"]):
    """

    :param content: string in which the block is searched
    :param keys: keys which determines the start of the block
    :param start: position in content to start searching
    :return: (position of first char in block [inclusive key],
              position of last char)
    """

    tuple_start = [(content.find(key, start), key) for key in keys]
    tuple_start = [x for x in tuple_start if x[0] != -1]
    assert len(tuple_start) == 1, str(tuple_start)

    idx_start, used_key = tuple_start[0]
    idx_end = idx_start + len(used_key)
    level = 1
    while len(content) > idx_end:
        if content[idx_end] in inc_level:
            level += 1
        elif content[idx_end] in dec_level:
            level -= 1
        idx_end += 1
        if level == 0:
            break

    assert level == 0
    return idx_start, idx_end


SAS_ATOM_PREFIX = "Atom "
def convert_sas2pddl_atoms(atom):
    if atom == "<none of those>" or atom.startswith("NegatedAtom"):
        return None
    assert atom.startswith(SAS_ATOM_PREFIX), "Err: %s" % atom
    atom = atom[len(SAS_ATOM_PREFIX):]
    atom = atom.replace(",", "")
    idx_bracket = atom.find("(")
    atom = ("(" + atom[:idx_bracket] +
            (" " if atom[idx_bracket + 1:].strip() != ")" else "") +
            atom[idx_bracket + 1:])

    return atom


def get_sas_variables(sas):
    variables = []
    for var in PATTERN_VARIABLE.finditer(sas):
        var = var.group().split("\n")
        variables.append(var[4:-1])
    return variables


def get_new_init(sas, pddl_init, variables):
    sas_init = PATTERN_STATE.findall(sas)
    assert len(sas_init) == 1
    sas_init = [int(x) for x in sas_init[0].split("\n")[1:-1]]
    sas_init = [variables[idx_var][idx_value]
                for idx_var, idx_value in enumerate(sas_init)
                if (variables[idx_var][idx_value] is not None and
                    not variables[idx_var][idx_value].startswith("Negated"))]
    sas_init = sorted(sas_init)

    for variable in variables:
        for value in variable:
            if value is None:
                continue
            pddl_init = pddl_init.replace(value, "")

    pddl_init = "(:init\n" + pddl_init[6:]  # enforce a linebreak
    pddl_init = pddl_init.split("\n")
    pddl_init = [pddl_init[0]] + sas_init + pddl_init[1:]
    pddl_init = ["\t\t" + x.strip() for x in pddl_init if x.strip() != ""]
    return "\n".join(pddl_init)


def run(args):
    assert len(args) == 2, ("args have to be [original pddl] "
                            "[sas file with init]")

    path_pddl = args[0]
    path_sas = args[1]

    assert os.path.isfile(path_pddl)
    assert os.path.isfile(path_sas)

    with open(path_pddl, "r") as f:
        pddl = f.read().lower()
    with open(path_sas, "r") as f:
        sas = f.read()

    init_start, init_end = get_block_range(pddl, ["(:init", "(:INIT"])
    pddl_init = pddl[init_start: init_end]

    sas_variables = get_sas_variables(sas)
    sas_variables = [[convert_sas2pddl_atoms(atom) for atom in variable]
                     for variable in sas_variables]

    new_init = get_new_init(sas, pddl_init, sas_variables)

    print(pddl[:init_start] + new_init + pddl[init_end:])


if __name__ == "__main__":
    run(sys.argv[1:])
