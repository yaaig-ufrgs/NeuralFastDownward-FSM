#!/usr/bin/env python3

"""
Usage:
    $ ./modify_json.py <section> <key> <value> [*.json]
Examples:
    $ ./modify_json.py sampling minimization none experiments/*.json
    $ ./modify_json.py train output-folder "+_hstar" experiments/*/*.json
"""

import json
from sys import argv


def is_float(s: str) -> bool:
    if '.' in s:
        try:
            float(s)
            return True
        except ValueError:
            return False
    return False


def fix_input(s: str):
    if is_float(s):
        val = float(s)
    else:
        val = int(s) if s.isdigit() else s
    return val


section = argv[1]
key = argv[2]

add_to_value = True if argv[3][0] == '+' else False

if add_to_value:
    v = argv[3][1:]
    value = fix_input(v)
else:
    value = fix_input(argv[3])

for json_file in argv[4:]:
    with open(json_file) as jf:
        json_decoded = json.load(jf)

    if add_to_value:
        json_decoded[section][key] += value
    else:
        json_decoded[section][key] = value

    print(f"{json_file}[{section}][{key}] = {json_decoded[section][key]}\n")

    with open(json_file, 'w') as jf:
        json.dump(json_decoded, jf, indent=4)
