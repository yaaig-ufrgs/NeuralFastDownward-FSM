#!/usr/bin/env python3

"""
Usage:
    $ ./modify_json.py <section> <key> <value> [*.json]
Example:
    $ ./modify_json.py sampling minimization none experiments/*.json
"""

import json
from sys import argv

section = argv[1]
key = argv[2]
value = argv[3]

for json_file in argv[4:]:
    with open(json_file) as jf:
        json_decoded = json.load(jf)

    json_decoded[section][key] = value
    print(f"{json_file}[{section}][{key}] = {json_decoded[section][key]}\n")

    with open(json_file, 'w') as jf:
        json.dump(json_decoded, jf, indent=4)
