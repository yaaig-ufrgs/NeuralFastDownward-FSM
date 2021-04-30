#!/usr/bin/env python

import os
import string

DIR_BLOCKSWORLD = "blocksworld_ipc"
DIR_DEPOT = "depot_fix_goals"
DIR_GRID = "grid_fix_goals"
DIR_PIPESWORLD_NO_TANKAGE = "pipesworld-notankage_fix_goals"
DIR_TRANSPORT = "transport-opt14-strips"
DIR_SCANALYZER08 = "scanalyzer-08-strips"
DIR_SCANALYZER11 = "scanalyzer-opt11-strips"
DIR_STORAGE = "storage"

DIRS_AAAI20_MAIN = [
    DIR_BLOCKSWORLD, DIR_DEPOT, DIR_PIPESWORLD_NO_TANKAGE,
    DIR_SCANALYZER08, DIR_SCANALYZER11, DIR_TRANSPORT
]

DIRS_AAAI20_MAIN_EXTENDED = DIRS_AAAI20_MAIN + [DIR_GRID]
DIRS_AAAI20_MAIN_EXTENDED2 = DIRS_AAAI20_MAIN_EXTENDED + [DIR_STORAGE]


def get_repo_base(path):
    path = os.path.abspath(path)
    path = os.path.dirname(path) if os.path.isdir(path) else path
    while os.path.dirname(path) != path:
        if os.path.exists(os.path.join(path, ".hg")):
            return path
        path = os.path.dirname(path)
    sys.exit("repo base could not be found")


class FormatPlaceholder:
    def __init__(self, key):
        self.key = key

    def __format__(self, spec):
        result = self.key
        if spec:
            result += ":" + spec
        return "{" + result + "}"


class FormatDict(dict):
    def __missing__(self, key):
        return FormatPlaceholder(key)


def partial_format(template, **kwargs):
    formatter = string.Formatter()
    mapping = FormatDict(**kwargs)
    return formatter.vformat(template, (), mapping)
