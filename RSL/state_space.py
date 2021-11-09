from math import ceil, prod
import numpy as np

def binary_to_fdr_state(binary: str, ranges: [int]):
    # binary 01000 1 0 0 0 1 00010 00001 00010
    # ranges [5, 1, 1, 1, 1, 1, 5, 5]
    # fdr [1, 0, 1, 1, 1, 0, 3, 4, 3]
    fdr = [-1] * len(ranges)
    ini = 0
    for i, range in enumerate(ranges):
        range -= 1
        subbin = binary[ini:ini+range]
        if subbin.count("1") > 1:
            return None
        fdr[i] = subbin.index("1") if "1" in subbin else range
        ini += range
    return fdr

def add_state_to_valid_states(valid_states, fdr: [int]):
    if len(fdr) == 1:
        valid_states[fdr[0]] = True
    else:
        add_state_to_valid_states(valid_states[fdr[0]], fdr[1:])

def check_partial_state_in_valid_states(valid_states, fdr: [int]):
    if fdr == None:
        return False
    if len(fdr) == 0:
        return valid_states
    if fdr[0] == len(valid_states)-1:
        for i in range(len(valid_states)):
            if check_partial_state_in_valid_states(valid_states[i], fdr[1:]):
                return True
    return check_partial_state_in_valid_states(valid_states[fdr[0]], fdr[1:])

def check_state_in_valid_states(valid_states, fdr: [int]):
    if fdr == None:
        return False
    if len(fdr) == 1:
        return valid_states[fdr[0]]
    return check_state_in_valid_states(valid_states[fdr[0]], fdr[1:])

def get_state_space_from_file(state_space_file: str):
    try:
        with open(state_space_file,) as f:
            lines = [l if l[-1] != "\n" else l[:-1] for l in f.readlines()]
        atoms = lines[0].split(";")
    except:
        return None

    # e.g. atoms = ['Atom A1', 'Atom A2', '', 'Atom B1', '', 'Atom C1', 'Atom C2', 'Atom C3']
    #      ranges = [3, 2, 4]
    ranges = []
    i, decr = 0, 0
    while i < len(atoms):
        i = atoms.index("", i) + 1 if "" in atoms[i:] else len(atoms) + 1
        ranges.append(i - 1 - decr + 1) # +1 because it needs value for <none of the options>
        decr = i
    num_atoms = ["Atom" in x for x in atoms].count(True)

    # Create a n-dimensions array (n = total of fdr variables)
    valid_states = np.reshape([False]*prod(ranges), tuple(ranges)).tolist()

    # Set True for all states \in state_space_file
    # state [3, 2, 4] = valid_states[3][2][4] is True
    for line in lines[1:]:
        add_state_to_valid_states(
            valid_states,
            binary_to_fdr_state(
                converter(line, num_atoms),
                ranges
            )
        )
    
    return valid_states, ranges

def converter(line_state: str, length: int):
    decimals = line_state.split(" ")
    assert ceil(length / 64) == len(decimals)
    binary = ""
    for i in range(len(decimals)):
        b = str(bin(int(decimals[i])))[2:]
        zeros = 64 - len(b)
        if i == 0 and length % 64 > 0:
            zeros = length % 64 - len(b)
            assert zeros >= 0
        binary += ("0" * zeros) + b
    return binary
