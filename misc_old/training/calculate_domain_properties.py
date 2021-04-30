#!/usr/bin/env python
"""
Calculate the domain_properties.json and atoms.json for the files in a given directory.
"""
from __future__ import print_function
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.training.misc import DomainProperties

import json
import re

NO_STATICS = False
FORCE_RECALCULATE = False
PRELOAD_TASKS = False

SKIP_DIRECTORTIES = [re.compile(".*nomystery.*")]

def run(argv):
    dirs = argv
    if len(dirs) == 0:
        print(__file__)
        todo = ["."]
        while len(todo) > 0:
            path = todo.pop()
            has_domain = False
            for item in os.listdir(path):
                path_item = os.path.join(path, item)
                if os.path.isdir(path_item):
                    if not any(p.match(path_item) for p in SKIP_DIRECTORTIES):
                        todo.append(path_item)
                elif os.path.isfile(path_item) and item.find("domain") > -1:
                    has_domain = True
            if has_domain:
                dirs.append(path)

    for d in dirs:
        if not os.path.isdir(d):
            continue

        print("Directory: %s" % d)
        if NO_STATICS:
            path_store = os.path.join(d, "domain_properties_no_statics.json")
        else:
            path_store = os.path.join(d, "domain_properties.json")
        path_atoms = os.path.join(d, "atoms.json")

        # Delete invalid json file
        for file_json in [path_store, path_atoms]:
            if os.path.exists(path_store):
                try:
                    with open(file_json, "r") as f:
                        json.load(f)
                except ValueError:
                    os.remove(file_json)
                    print("\tInvalid json: %s" % (file_json))


        previously_computed = (os.path.exists(path_store) and
                               os.path.exists(path_atoms))


        if ((not previously_computed) or FORCE_RECALCULATE):
            x = DomainProperties.get_property_for(
                d,
                preload_tasks=PRELOAD_TASKS,
                no_gnd_static=NO_STATICS, verbose=6, parallize=True,
                load=(path_store
                      if (not FORCE_RECALCULATE and os.path.exists(path_store))
                      else None),
                store=path_store, store_atoms=path_atoms)
        else:
            print("\tPreviously computed.")



if __name__ == "__main__":
    run(sys.argv[1:])
