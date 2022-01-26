#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import re
import sys

import summarize_slurm_tools as tools
from summarize_slurm_tools import REGEX_FLOAT, get_search_pattern


CACHE_INTERVALS = 50
# Pattern for sample log file
PATTERN_TRAINING_LOG_FILE = re.compile(
    r"(.*/)?slurm_training_\d+.(out|err)"
)

# Pattern for the output file
PATTERN_SLURM_ARRAY_ID = re.compile(r"^Slurm array id: (\d+)/(\d+)$")
PATTERN_CALLSTRING = re.compile(r"Call: (.*)")
PATTERN_START_TIMESTAMP = re.compile(r"(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d(\.\d+)?)")
PATTERN_VALIDATION = re.compile(r"--validation\s([^\s]+)")
PATTERN_TEST = re.compile(r"--test\s([^\s]+)")
PATTERN_MAX_EPOCHS = re.compile(r"Epoch 1/(\d+)")
PATTERN_EPOCH_END = re.compile(r"- (\d+)s (\d+)ms/step(.*)")
PATTERN_DATASET_SIZES = re.compile(r"Data sizes: (\d+), (\d+), (\d+) \(train, validation, test\)")
PATTERN_TASKSET_SIZES = re.compile(r"Loaded from distrinct tasks: (\d+), (\d+), (\d+) \(test, validation, train\)")
PATTERN_TRAINING_OUTCOME = re.compile(r"Training Outcome: (NotStarted|Aborted|Finished|Failed)")
PATTERN_TIME_PARSING = re.compile(r"Parsing time: (%s)s" % REGEX_FLOAT)
PATTERN_TIME_DATA = re.compile(r"Loading data time: (%s)s" % REGEX_FLOAT)
PATTERN_TIME_NETWORK_INITIALIZATION = re.compile(r"Network initialization time: (%s)s" % REGEX_FLOAT)
PATTERN_TIME_TRAINING = re.compile(r"Network training time: (%s)s" % REGEX_FLOAT)
PATTERN_TIME_EVALUATION = re.compile(r"Network evaluation time: (%s)s" % REGEX_FLOAT)
PATTERN_TIME_ANALYSIS = re.compile(r"Network analysis time: (%s)s" % REGEX_FLOAT)
PATTERN_TIME_FINALIZATION = re.compile(r"Network finalization time: (%s)s" % REGEX_FLOAT)
PATTERNS_TIME = [
    ("option_parsing", PATTERN_TIME_PARSING, lambda x, y: True),
    ("data_loading", PATTERN_TIME_DATA, lambda x, y: True),
    ("network_initialization", PATTERN_TIME_NETWORK_INITIALIZATION, lambda x, y: True),
    ("network_training", PATTERN_TIME_TRAINING, lambda x, y: True),
    ("network_evaluation", PATTERN_TIME_EVALUATION,
     lambda _, training_succeeded: training_succeeded),
    ("network_analysis", PATTERN_TIME_ANALYSIS,
     lambda _, training_succeeded: training_succeeded),
    ("network_finalization", PATTERN_TIME_FINALIZATION,
     lambda _, training_succeeded: training_succeeded),
]

PATTERN_TENSORFLOW_ADDITIONAL_FEATURES_WARNING = re.compile(
    r"\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+: I tensorflow/core/platform/cpu_"
    r"feature_guard.cc:\d+] Your CPU supports instructions that this "
    r"TensorFlow binary was not compiled to use: AVX2 AVX512F FMA\n?"
)

PATTERN_TENSORFLOW_MEMORY_ALLOCATION_WARNING = re.compile(
    r"\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\.\d+: W tensorflow/core/framework/"
    r"allocator.cc:\d+] Allocation of \d+ exceeds \d+% of system memory.\n?"
)
PATTERN_GZIP_CRC_ERROR = re.compile(
    r"IOError: CRC check failed %s != %sL" %
    (tools.REGEX_HEXADECIMAL, tools.REGEX_HEXADECIMAL)
)

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, nargs="+",
                    help="List of directories where to summarize.")
#parser.add_argument("-tmp", "--temporary-directory", type=str, action="store",
#                    help="Path to a directory where to store the temporary "
#                         "data. The directory has to exist.")
parser.add_argument("-c", "--clean", action="store_true",
                    help="After summarizing a directory successfully, all"
                         "log files summarized are deleted.")
parser.add_argument("--cache-interval", type=int, action="store", default=None,
                    help="Positive integer. After reading n logs, the results"
                         "are written to the final location (if the execution "
                         "stops it can be proceeded from there)")


def get_metric_evolution(content):
    evolution = {"time": [], "time_per_step": []}
    first_round = True
    for m in PATTERN_EPOCH_END.finditer(content):
        time, time_per_step, metrics = m.groups()
        evolution["time"].append(time)
        evolution["time_per_step"].append(time_per_step)
        for metric in metrics.split(" - "):
            if metric == "":
                continue
            name, value = metric.split(": ")
            if name not in evolution:
                assert first_round, "New metric appeared after first epoch."
                evolution[name] = []
            evolution[name].append(float(value))
        first_round = False
    return evolution


def get_output_stats(path, has_error):
    assert os.path.exists(path)
    stats = {}
    with open(path, "r") as f:
        content = f.read()
        if PATTERN_SLURM_ARRAY_ID.match(content):
            stats["finished"] = False
            return stats, "Job terminated unexpectedly"
        stats["call_string"] = get_search_pattern(
            PATTERN_CALLSTRING, content, 0, allowed_none=has_error)
        stats["timestamp"] = get_search_pattern(
            PATTERN_START_TIMESTAMP, content, 0, allowed_none=has_error
        )
        stats["validation"] = get_search_pattern(
            PATTERN_VALIDATION, content, 0, allowed_none=has_error)
        stats["test"] = get_search_pattern(
            PATTERN_TEST, content, 0, allowed_none=has_error)

        if content.find("Other training still running: ") > -1:
            stats["skip_execution"] = "other_instance_running"
            stats["finished"] = True
            return stats, None
        if content.find("Skip at network flag: ") > -1:
            stats["skip_execution"] = "skip_flag_set"
            stats["finished"] = True
            return stats, None
        if content.find("Network previously trained: ") > -1:
            stats["skip_execution"] = "previously_trained"
            stats["finished"] = True
            return stats, None
        if content.find("Exit: Training data set is empty.") > -1:
            stats["skip_execution"] = "exit_few_samples"
            stats["finished"] = True
            return stats, None

        stats["epochs_max"] = get_search_pattern(
            PATTERN_MAX_EPOCHS, content, 0, allowed_none=has_error)
        stats["metric_evolution"] = get_metric_evolution(content)
        data_set_sizes = get_search_pattern(PATTERN_DATASET_SIZES, content,
                                            None, [0, 1, 2], has_error)
        stats["data_set_sizes"] = {
            "train": int(data_set_sizes[0]) if data_set_sizes is not None else 0,
            "validation": int(data_set_sizes[1]) if data_set_sizes is not None else 0,
            "test": int(data_set_sizes[2]) if data_set_sizes is not None else 0
        }
        train_set_sizes = get_search_pattern(PATTERN_TASKSET_SIZES, content,
                                             None, [0, 1, 2], True)
        stats["task_set_sizes"] = {
            "test": int(
                train_set_sizes[0]) if train_set_sizes is not None else 0,
            "validation": int(
                train_set_sizes[1]) if train_set_sizes is not None else 0,
            "train": int(train_set_sizes[2]) if train_set_sizes is not None else 0
        }


        stats["training_rounds "] = content.count("Epoch 1/")
        stats["training_outcome"] = get_search_pattern(
            PATTERN_TRAINING_OUTCOME, content, 0, allowed_none=has_error)
        training_succeeded = stats["training_outcome"] == "Finished"
        phase_times = {}
        for name, pattern, shall_parse in PATTERNS_TIME:
            if not shall_parse(has_error, training_succeeded):
                continue
            phase_time = get_search_pattern(
                pattern, content, 0, None, has_error)
            phase_times[name] = float(phase_time) if phase_time is not None else None
        stats["phase_times"] = phase_times
        stats["finished"] = any([content.endswith(suffix)
                                 for suffix in ["finished", "finished\n"]])

        stats["final_100_lines"] = tools.last_n_lines(content, 100)
        stats["exit_few_samples"] = (
                stats["final_100_lines"].endswith(
                    "Exit: At least one given data set is close to empty.\n")
                or stats["final_100_lines"].endswith(
            "Exit: Training data set is empty.\n"))
        stats["exit_below_sample_bound"] = (stats["final_100_lines"].find(
            "Exit: A minimum sample requirement cannot be satisfied") > -1)

    return stats, None


def error_unification_func(error):
    error = tools.unify_slurm_cancel_error(error)
    error = tools.unify_slurm_timeout_error(error)
    error = tools.unify_slurm_oom_killed_error(error)
    error = PATTERN_TENSORFLOW_ADDITIONAL_FEATURES_WARNING.sub("", error)
    error = PATTERN_TENSORFLOW_MEMORY_ALLOCATION_WARNING.sub("", error)
    error = PATTERN_GZIP_CRC_ERROR.sub("IOError: CRC check failed "
                                       "HEXADECIMAL1 != HEXADECIMAL2L", error)
    return error


def summarize_errors(path, stats, errors):
    for no, (err, job_ids) in enumerate(errors.items()):
        path_error = os.path.join(path, "error_%i.txt" % no)
        outcomes = {}
        have_finished = ([0], set())
        for job_id in job_ids:
            stat = stats[job_id]

            if stat["finished"]:
                have_finished[0][0] += 1
                have_finished[1].add(job_id)

        content = "finished: %i. Job_id: %s\n" % (
            have_finished[0][0], ", ".join(have_finished[1]))

        with open(path_error, "w") as f:
            f.write(content)
            f.write(err)


def summarize(path, temporary=None, clean=False, cache_intervals=CACHE_INTERVALS):
    if not os.path.isdir(path):
        raise ValueError("Given path to clean is no directory")
    assert temporary is None, "Not implemented"
    assert cache_intervals is None or cache_intervals == int(cache_intervals)
    assert cache_intervals is None or cache_intervals > 0


    # Setup/Reload the data storage
    print("Loading")
    summary_data = tools.get_summaries_data(path, tmp_dir=temporary)
    path_stats, path_errors, stats_mapping, error_mapping = summary_data[:4]
    if temporary is None:
        path_tmp_stats, path_tmp_errors = None, None
    else:
        path_tmp_stats, path_tmp_errors = summary_data[4:]

    if path_tmp_stats is not None and os.path.isfile(path_tmp_stats):
        os.remove(path_tmp_stats)
    if path_tmp_errors is not None and os.path.isfile(path_tmp_errors):
        os.remove(path_tmp_errors)
    print("Loaded")

    # Go through all files (skip if irrelevant or previously done)
    next_caching = -1 if cache_intervals is None else cache_intervals
    next_caching_errors = False
    log_filter = lambda x: PATTERN_TRAINING_LOG_FILE.match(x)
    log_generator = tools.get_log_generator(
        path, stats_mapping,
        get_file_group=tools.get_associated_log_files,
        delete=False, filter=log_filter)
    processed_files = []
    for counter, associated_files in enumerate(log_generator):
        processed_files.extend(associated_files)
        path_out, path_err = associated_files[:2]
        job_id = tools.get_job_id(path_out)
        print("%i: %s" % (counter, path_out))
        next_caching -= 1

        # Process error (nothing happens if no error)
        current_error = tools.get_error(path_err, error_unification_func)

        # Process stats
        current_stats, additional_error = get_output_stats(
            path_out, current_error is not None)
        stats_mapping[job_id] = current_stats
        if additional_error is not None:
            additional_error = error_unification_func(additional_error)
            current_error = (additional_error if current_error is None else
                             ("%s\n\n%s" % (current_error, additional_error)))


        valid_abort_options = ["exit_few_samples", "exit_below_sample_bound"]
        for abort_option in valid_abort_options:
            if (current_error is not None and
                    abort_option in current_stats and
                    current_stats[abort_option]):
                current_error = None

        if current_error is None and not (current_stats["finished"] or any(
                current_stats[abort_option] for abort_option in
                valid_abort_options)):
            current_error = "Missing final finished"
        if tools.update_error_summary(error_mapping, job_id, current_error):
            next_caching_errors = True


        if next_caching == 0:
            if next_caching_errors:
                tools.save_dict(path_errors, error_mapping)
            tools.save_dict(path_stats, stats_mapping)
            next_caching = -1 if cache_intervals is None else cache_intervals
            next_caching_errors = False

    if next_caching != cache_intervals:
        if next_caching_errors:
            tools.save_dict(path_errors, error_mapping)
        tools.save_dict(path_stats, stats_mapping)
    print("Summarize")
    summarize_errors(path, stats_mapping, error_mapping)

    if clean:
        print("Clean processed logs")
        for pfile in processed_files:
            if pfile is not None:
                os.remove(pfile)

    print("Done")


def parse_args(argv):
    options = parser.parse_args(argv)
    for path in options.directory:
        assert os.path.isdir(path), \
            "A directory to summarize does not exit: %s" % path
    #assert (options.temporary_directory is None
    #        or os.path.isdir(options.temporary_directory)), \
    #    ("The given directory for temporary data does not exist: %s"
    #     % options.temporary_directory)
    assert options.cache_interval is None or options.cache_interval > 0
    return options


def run(argv):
    options = parse_args(argv)
    for path in options.directory:
        summarize(path=path,
                  temporary=None, #options.temporary_directory,
                  clean=options.clean,
                  cache_intervals=options.cache_interval)


if __name__ == "__main__":
    run(sys.argv[1:])
