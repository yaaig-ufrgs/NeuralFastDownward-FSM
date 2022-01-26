#!/usr/bin/env python
"""
Extracts from the training parsed and summarized training output some stats and stores them in a json file in a dictionary
./extract_restrictions.py JOB_STATS_FILE --restriction-keys HOW_TO_STORE_KEYS
'$FOLD' train
"""

import argparse
import datetime
import json
import os
import sys
import re


KEY_CALL_STRING = "call_string"
KEY_TASK_SET_SIZE = "task_set_sizes"
KEY_DATA_SET_SIZE = "data_set_sizes"
KEY_TASK_SET_SIZE_TRAIN = "train"
KEY_TASK_SET_SIZE_VALIDATION = "validation"
KEY_TASK_SET_SIZE_TEST = "test"
KEY_FINISHED = "finished"
KEY_TIMESTAMP = "timestamp"

KEY_RESTRICTIONS_TASK_COUNT_12GB = "task_count_12gb"

REGEX_DIRECTORY = re.compile(r"--directory (\S+)")
REGEX_FOLD = re.compile(r"--prefix \S+_(\d+)_fold_")

RESTRICTION_KEY_FOLD = "$FOLD$"
RESTRICTION_KEY_TEST = "$REGEX_TEST$"
RESTRICTION_KEY_VALID = "$REGEX_VALID$"


def type_is_file(arg):
    assert os.path.isfile(arg)
    return arg


parser = argparse.ArgumentParser()
parser.add_argument("job_stats", type=type_is_file, nargs="+",
                    help="Path to the job_stats json files")
parser.add_argument("--restriction-keys", nargs="+", required=True,
                    help="keys under which the restriction will be added to "
                         "the restriction file. Either use some not magic"
                         " strings which will be used or used:"
                         "{RESTRICTION_KEY_FOLD} for the fold number, "
                         "{RESTRICTION_KEY_VALID} for the validation set"
                         "regex, {RESTRICTION_KEY_TEST} for the test set "
                         "regex".format(**locals()))
parser.add_argument("--stat-keys", nargs="+",
                    default=[KEY_DATA_SET_SIZE, KEY_TASK_SET_SIZE_TRAIN],
                    help="list of keys in the job_stats dict to get the "
                         "restriction value. By default selects size of "
                         "training data")
parser.add_argument("--local", action="store_true",
                    help="converts the task paths to my local system "
                         "(infai-> home)")


def job_stats_generator(files_job_stats):
    for file_job_stats in files_job_stats:
        with open(file_job_stats, "r") as f:
            all_job_stats = json.load(f)
            for job_stats in all_job_stats.values():
                yield job_stats


def get_regex(s, regex, clazz=None, count=1):
    results = regex.findall(s)
    assert len(results) == count
    return [x if clazz is None else clazz(x) for x in results]


def get_restrictions_file(dir_task):
    return os.path.join(dir_task, "sample_restrictions.json")


def load_restrictions(file_restrictions):
    if os.path.exists(file_restrictions):
        with open(file_restrictions, "r") as f:
            return json.load(f)
    else:
        return {}


def save_restrictions(file_restrictions, restrictions):
    with open(file_restrictions, "w") as f:
        json.dump(restrictions, f, sort_keys=True, indent=4)


def run(options):
    # {task_directory : {key1 : {key 2: ... {key N: (timestamp, restriction
    # value)} ... } } }
    to_save = {}

    for job_stats in job_stats_generator(options.job_stats):
        if job_stats[KEY_FINISHED] is False:
            continue

        # Get the restriction value to store
        restriction_value = job_stats
        for stat_key in options.stat_keys:
            if stat_key not in restriction_value:
                restriction_value = None
                break
            restriction_value = restriction_value[stat_key]
        if restriction_value is None:
            continue

        # Get the restriction key behind which it shall be stored
        restriction_keys = []
        fold = None
        for restriction_key in options.restriction_keys:
            if restriction_key == RESTRICTION_KEY_FOLD:
                if fold is None:
                    fold = get_regex(
                        job_stats[KEY_CALL_STRING], REGEX_FOLD)[0]
                restriction_key = fold
            elif restriction_key == RESTRICTION_KEY_VALID:
                restriction_key = job_stats["validation"]
            elif restriction_key == RESTRICTION_KEY_TEST:
                restriction_key = job_stats["test"]
            restriction_keys.append(restriction_key)

        task_directory = get_regex(job_stats[KEY_CALL_STRING],
                                   REGEX_DIRECTORY)[0]
        if options.local:
            task_directory = task_directory.replace("infai", "home")
        timestamp = datetime.datetime.strptime(job_stats[KEY_TIMESTAMP],
                                               '%Y-%m-%d %H:%M:%S.%f')

        restriction_keys = [task_directory] + restriction_keys
        curr_dict = to_save
        for restriction_key in restriction_keys[:-1]:
            if restriction_key not in curr_dict:
                curr_dict[restriction_key] = {}
            curr_dict = curr_dict[restriction_key]
        if (restriction_keys[-1] not in curr_dict or
                curr_dict[restriction_keys[-1]][0] < timestamp):
            curr_dict[restriction_keys[-1]] = (timestamp, restriction_value)

    nb_restrictions_added = 0
    for task_directory, restriction_keys_and_value in to_save.items():
        file_restrictions = get_restrictions_file(task_directory)
        if not os.path.isdir(task_directory):
            print("Domain directory missing: %s" % task_directory)
            continue
        restrictions = load_restrictions(file_restrictions)

        nb_restrictions_added += add_restriction_key_and_value(
            restrictions, restriction_keys_and_value)
        save_restrictions(file_restrictions, restrictions)
    print("Restriction added for {nb_restrictions_added} networks.".format(
        **locals()))


def add_restriction_key_and_value(restrictions, key_and_value):
    if isinstance(key_and_value, dict):
        nb_restrictions_added = 0
        for k, v in key_and_value.items():
            if isinstance(v, dict):
                if k not in restrictions:
                    restrictions[k] = {}
                nb_restrictions_added += add_restriction_key_and_value(
                    restrictions[k], v)
            else:
                assert isinstance(v, tuple)
                assert len(v) == 2
                restrictions[k] = v[1]
                nb_restrictions_added += 1

        return nb_restrictions_added
    else:
        assert False, type(key_and_value)


if __name__ == "__main__":
    run(parser.parse_args(sys.argv[1:]))
