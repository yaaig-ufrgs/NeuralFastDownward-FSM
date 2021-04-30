#!/usr/bin/env python
"""
analyse something with size of the data set. I have forgotten what this script was doing exactly :)
anaylse_dataset_size_slurm.sh is used to run this on a slurm grid
"""

from __future__ import print_function

import argparse
import gzip
import json
import os
import shutil
import sys

# Python 2/3 compatibility
if sys.version_info < (3,):
    import subprocess32 as subprocess
else:
    import subprocess

gzip_write_converter = lambda x: x.encode()
gzip_read_converter = lambda x: x.decode()
if sys.version_info[0] == 2:
    gzip_write_converter = lambda x: x
    gzip_read_converter = lambda x: x

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


EXITCODE_SUCCESS = 0
EXITCODE_GZIP_ERROR = 1

FIELD_PROBLEM_HASH = "problem_hash"
FIELD_MODIFICATION_HASH = "modification_hash"
FIELD_DELIMITER = "delimiter"


parser = argparse.ArgumentParser()
parser.add_argument("--debug", action="store_true",
                    help="Run the script in debug mode")
parser.add_argument("-d", "--directory", type=str, action="append", default=[],
                    help="Directory under which all directories containing"
                         "domain.pddl shall be analysed.")
parser.add_argument("-s", "--suffix", type=str, action="append", default=[],
                    help="Only files with a given suffix are read")
parser.add_argument("--reanalyse", action="store_true",
                    help="By default already analysed files are not analysed "
                         "again. Setting this flag deletes previously stored"
                         "analysis output and produces the output anew.")
parser.add_argument("--slurm", action="store_true",
                    help="For each detected directory to analyse submits a new "
                         "slurm job.")
parser.add_argument("--dry", action="store_true",
                    help="Shows only the directories it would analyse")


def print_debug(active, *args, **kwargs):
    if active:
        print("Debug> ", *args, **kwargs)


def get_path_output(dir_data, tmp=False):
    return os.path.join(dir_data,
                        "data_set_sizes.%s" % ("tmp" if tmp else "json"))


def find_domains(path):
    path_domains = []
    todo = [path]
    while len(todo) > 0:
        path_dir = todo.pop()
        path_domain_file = os.path.join(path_dir, "domain.pddl")
        if os.path.isfile(path_domain_file):
            path_domains.append(path_dir)
        for item in os.listdir(path_dir):
            path_item = os.path.join(path_dir, item)
            if os.path.isdir(path_item):
                todo.append(path_item)
    return path_domains


def split_meta(line):
    line = line.strip()
    if not line.startswith("{"):
        return None, line

    level = 1
    for i in range(1, len(line)):
        if line[i] == "{":
            level += 1
        elif line[i] == "}":
            level -= 1
            if level == 0:
                return json.loads(line[: i + 1]), line[i + 1:]

    raise ValueError("Entry has an opening, but not closing tag: " + line)


def get_data_files(dir_data, suffixes):
    suffix_to_file = {suffix: [] for suffix in suffixes}  # {suffix: [paths]}
    for item in os.listdir(dir_data):
        path_item = os.path.join(dir_data, item)
        for suffix in suffixes:
            if item.endswith(suffix):
                suffix_to_file[suffix].append(path_item)

    all_files = set()
    for suffix, files in suffix_to_file.items():
        all_files.update(files)
    return all_files


def load_previous_data(file_output, reanalyse):
    sizes = {}
    if os.path.exists(file_output):
        assert os.path.isfile(file_output), \
            "A directory is where the output file shall be stored."
        if reanalyse:
            os.remove(file_output)
        else:
            with open(file_output, "r") as f:
                sizes = json.load(f)
    return sizes


def analyse_data_file(file_data):
    print("\t%s" % file_data)
    problems = set()
    count_samples = 0
    try:
        with gzip.open(file_data, "r") as f:
            for line in f:
                line = gzip_read_converter(line)
                if line == "" or line.startswith("#"):
                    continue

                meta, entry = split_meta(line)
                assert meta is not None
                assert FIELD_PROBLEM_HASH in meta
                assert FIELD_MODIFICATION_HASH in meta
                assert FIELD_DELIMITER in meta

                problem_hash = meta[FIELD_PROBLEM_HASH]
                modification_hash = meta[FIELD_MODIFICATION_HASH]
                problems.add((problem_hash, modification_hash))
                count_samples += 1

    except IOError as e:
        print("IOError in: %s\n%s" % (file, str(e)))
        sys.exit(EXITCODE_GZIP_ERROR)

    return {"#problems": len(problems),
            "#samples": count_samples}


def analyse_data_dir(dir_data, suffixes, reanalyse=False, debug=False):
    file_output = get_path_output(dir_data, tmp=False)
    file_temporary = get_path_output(dir_data, tmp=True)

    sizes = load_previous_data(file_output, reanalyse)
    files_data = get_data_files(dir_data, suffixes)

    for file_data in files_data:
        file_key = os.path.basename(file_data)
        if file_key not in sizes:
            stats = analyse_data_file(file_data)
            sizes[file_key] = stats
            with open(file_temporary, "w") as f:
                json.dump(sizes, f)
            shutil.move(file_temporary, file_output)
        if debug:
            print_debug(debug, "Stop after first problem is analysed")
            break


def parse_argv(argv):
    options = parser.parse_args(argv)
    for path_dir in options.directory:
        assert os.path.isdir(path_dir)
    return options


def run(argv):
    options = parse_argv(argv)
    print_debug(options.debug, "Debug mode active")

    domains = set()
    for d in options.directory:
        domains.update(find_domains(d))

    if options.dry:
        print(domains)
        sys.exit()

    if options.slurm:
        command = (["sbatch", "--array=0-%i" % (len(domains) - 1),
                    os.path.join(os.path.dirname(__file__),
                                 "analyse_dataset_size_slurm.sh")] +
                   [x for x in sorted(domains)] +
                   (["--debug"] if options.debug else []) +
                   (["--reanalyse"] if options.reanalyse else []))
        for suffix in options.suffix:
            command += ["--suffix", suffix]
        print("Slurm Command:", command)
        subprocess.call(command)

    else:
        for dir_domain in sorted(domains):
            print("Domain: %s" % dir_domain)

            analyse_data_dir(dir_domain, options.suffix,
                             options.reanalyse, options.debug)
            if options.debug:
                print_debug(options.debug, "Stop after first directory")
                break

    print("Done.")


if __name__ == "__main__":
    run(sys.argv[1:])

