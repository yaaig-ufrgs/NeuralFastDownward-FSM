#!/usr/bin/env python

from __future__ import print_function
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from src.training import parser, parser_tools

from src.training.bridges import StateFormat, LoadSampleBridge
from src.training.misc import DomainProperties, StreamDefinition, StreamContext
from src.training.samplers import DirectorySampler

import argparse
import json
import re
import time

if sys.version_info < (3,):
    import subprocess32 as subprocess
    def decoder(x):
        return x.decode()
else:
    import subprocess
    def decoder(x):
        return x


SLURM_FILE = os.path.join(os.path.dirname(__file__), "convert_sample_format.sh")

CHOICE_STATE_FORMATS = []
for name in StateFormat.get_format_names():
    CHOICE_STATE_FORMATS.append(name)


""" PARSING FUNCTIONS """
def parse_file_exists(arg):
    assert os.path.exists(arg)
    return arg


def parse_state_format(arg):
    assert arg in CHOICE_STATE_FORMATS
    return StateFormat.get(arg)


def parse_stream(arg):
    return parser.construct(
        parser_tools.ItemCache(),
        parser_tools.main_register.get_register(StreamDefinition),
        arg)


aparser = argparse.ArgumentParser("Convert samples from a given format into "
                                 "another format")
aparser.add_argument("format", type=parse_state_format,
                     action="store",
                     help=("State format name into which the loaded data shall"
                           "be converted (if not given, the preferred of the"
                           "network is chosen)."))
aparser.add_argument("-d", "--directory", type=parse_file_exists,
                    action="append", default=[],
                    help="List of directories where samples shall be converted"
                         "(each directory is individually treated).")

aparser.add_argument("-r", "--regex", type=re.compile,
                     action="append", default=[],
                     help="Regex to filter which problems to choose from the"
                          "directories(the regexes are used for ALL "
                          "directories).")
aparser.add_argument("-i", "--input", type=parse_stream,
                    action="append", default=[],
                    help="Define an input stream for the loading of samples"
                         "(use this option multiple times for multiple). The "
                         " available streams can be checked in "
                         "training.misc.stream_contexts.py"
                         "(the way this is done is for every problem file of"
                         " which data shall be loaded the stream is asked,"
                         "where would you store data for this file and then"
                         "the data at the location is loaded).")

aparser.add_argument("-o", "--output", type=parse_stream,
                    action="append", default=[],
                    help="Define an output stream for the loading of samples"
                         "(use this option multiple times for multiple). The "
                         " available streams can be checked in "
                         "training.misc.stream_contexts.py"
                         "(the way this is done is for every problem file of"
                         " which data shall be loaded the stream is asked,"
                         "where would you store data for this file and then"
                         "the data at the location is loaded).")

aparser.add_argument("--slurm", action="store_true",
                     help="Submits for every problem detected a slurm job to "
                          "convert the associated data files.")

aparser.add_argument("--dry", action="store_true",
                     help="Does not execute the conversion")

def timing(old_time, msg):
    new_time = time.time()
    print(msg % (new_time-old_time))
    return new_time


def convert_data(directory, input_streams, output_streams, format,
                 regexes_problems, slurm, dry):
    """
    :param directory:
    :param streams:
    :param format:
    :return:
    """

    sampler = DirectorySampler(None, directory, filter_file=regexes_problems)

    if slurm:
        return sampler._iterable

    if dry:
        print(sampler._iterable)
        return

    start_time = time.time()
    print("Start analysing Domain:")
    path_domain = os.path.join(directory, "domain.pddl")
    path_load = os.path.join(
        os.path.dirname(path_domain), "domain_properties.json")
    path_store = path_load if not os.path.exists(path_load) else None
    path_load = path_load if path_store is None else None

    domain_properties = DomainProperties.get_property_for(
        path_domain=path_domain,
        paths_problems=sampler._iterable,
        no_gnd_static=False,
        load=path_load,
        store=path_store,
        verbose=1)
    _ = timing(start_time, "Domain analysising time: %ss")

    bridge = LoadSampleBridge(
        streams=StreamContext(streams=input_streams),
        write_streams=StreamContext(streams=output_streams),
        format=format, skip=True,
        domain_properties=domain_properties,
        provide=False,
        prune=False

    )

    sampler.sbridges = [bridge]
    sampler.initialize()
    sampler.sample()
    sampler.finalize()


def slurm_clean_argv(argv):
    i = 0
    new_args = []
    while i < len(argv):
        if argv[i] in ["-d", "--directory", "-r", "--regex"]:
            i += 2
        elif argv[i] in ["--slurm"]:
            i += 1
        else:
            new_args.append(argv[i])
            i += 1
    return new_args


def run(argv):
    options = aparser.parse_args(argv)
    problems = {}
    for directory in options.directory:
        files_problems = convert_data(
            directory, options.input, options.output,
            options.format, options.regex, options.slurm, options.dry)
        problems[directory] = files_problems

    if options.slurm:
        directory_problem_tuples = [(os.path.dirname(problem), problem)
                                    for directory, problems in problems.items()
                                    for problem in problems]
        command = (
                ["sbatch",
                 "--array=0-%i" % (len(directory_problem_tuples) - 1),
                 SLURM_FILE] +
                [x for t in directory_problem_tuples for x in t] +
                slurm_clean_argv(argv)
        )
        print(" ".join(command))
        if not options.dry:
            subprocess.call(command)

if __name__ == "__main__":
    run(sys.argv[1:])
