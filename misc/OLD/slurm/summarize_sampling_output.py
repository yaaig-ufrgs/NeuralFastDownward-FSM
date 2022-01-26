#!/usr/bin/env python
from __future__ import print_function

import summarize_slurm_tools as tools
from summarize_slurm_tools import REGEX_FLOAT, get_search_pattern

import argparse
import json
import os
import re
import shutil
import sys

MISSING_FD_LAST_TIME_ERROR = "Missing FD exit time"
REGEX_ACTUAL_SEARCH_TIME = (r"Actual search time: (%s)s \[t=(%s)s\]" %
                            (REGEX_FLOAT, REGEX_FLOAT))


CACHE_INTERVALS = 50
# Pattern for sample log file
PATTERN_SAMPLE_LOG_FILE = re.compile(
    r"(.*/)?slurm_sample_\d+.(out|err)"
)
PATTERN_STATS_ERROR_FILE = re.compile(
    r"(.*/)?(job_stats|errors)_\d+.json"
)
# Pattern for the error file
PATTERN_SAMPLING_FILE_PATH = re.compile(
    r"/scratch/ferber/slurm-job\.\d+/tmp/[^/]+\.tmp")

# Pattern for the output file
PATTERN_PDDL = re.compile(r"[a-zA-Z0-9\./_-]+\.pddl\s")
PATTERN_JOB_CALLSTRING = re.compile("SamplingCall: (.*)")
PATTERN_CALLSTRING = re.compile(r"callstring: .*")
PATTERN_CALLSTRING2 = re.compile(r"command line string: .*")
PATTERN_START_TIMESTAMP = re.compile(r"(\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d(\.\d+)?)")
PATTERN_ACTUAL_SEARCH_TIME = re.compile(r"%s(\n%s)?" %
                                        (REGEX_ACTUAL_SEARCH_TIME,
                                         REGEX_ACTUAL_SEARCH_TIME))
PATTERN_SOLUTION_FOUND = re.compile(r"Solution found!")
PATTERN_NO_SOLUTION_FOUND = re.compile(r"(Completely explored state space|"
                                       r"Time limit reached. Abort search.)")
PATTERN_EXIT_CODE = re.compile(r"(returned non-zero exit status (-?\d+)|"
                               r"search exit code: (-?\d+))")
PATTERN_INTERNAL_PLAN_FILE = re.compile(r"--internal-plan-file \S+")
PATTERN_HASH = re.compile(r"hash=[^, ]+")


parser = argparse.ArgumentParser()
parser.add_argument("directory", type=str, nargs="+",
                    help="List of directories where to summarize.")
parser.add_argument("-tmp", "--temporary-directory", type=str, action="store",
                    help="Path to a directory where to store the temporary "
                         "data. The directory has to exist.")
parser.add_argument("-c", "--clean", action="store_true",
                    help="After summarizing a directory successfully, all"
                         "log files summarized are deleted.")
parser.add_argument("--skip-creation-error-files", action="store_true",
                    help="Does not create for each distinct error a single file."
                         "with the error and the jobs having the error.")
parser.add_argument("-m", "--merge", action="store_true",
                    help="It does not parse the logs, but merges all"
                         "job_stats(_.*)?.json and errors(_.*)?.json files")

""" Methods associated to errors """
def error_unification_func(error):
    error = tools.unify_slurm_cancel_error(error)
    error = tools.unify_slurm_timeout_error(error)
    error = tools.unify_slurm_oom_killed_error(error)
    error = PATTERN_SAMPLING_FILE_PATH.sub(
        "/scratch/ferber/slurm-job.XYZ/tmp/pXYZ.tmp", error)
    return error


""" Methods associated to the stdout of slurm """


def get_pddl_input(content, has_error):
    pddls = [tools.sanitize_string(x.group())
             for x in PATTERN_PDDL.finditer(content)
             if x.group().find("_automatically_generated_problem.pddl") == -1
                and content[x.start() - 1] != "\n"]
    pddls = set(pddls)
    if len(pddls) != 2:
        if has_error:
            return None, None
        else:
            assert False, "Cannot savely determine the problem and domain file %s" % ", ".join(pddls)

    domain = None
    problem = None
    for x in pddls:
        if x.find("domain") > -1:
            assert domain is None, "Multiple domain file candidates: %s" % ", ".join(pddls)
            domain = x
        else:
            assert problem is None, "Multiple problem file candidates: %s" % ", ".join(pddls)
            problem = x

    assert domain is not None, "Unable to determine domain file for slurm job"
    assert problem is not None, "Unable to determine problem file for slurm job"
    return domain, problem


def get_call_strings(content, has_error):
    calls = PATTERN_CALLSTRING.findall(content) + PATTERN_CALLSTRING2.findall(content)
    translate = None
    fd = None
    for x in calls:
        count = 0
        if x.find("translate.py") > -1:
            assert translate is None or x == translate, "Multiple translate call candidates: %s" % "\n".join(calls)
            translate = x
            count += 1
        if x.find("downward") > -1:
            x = PATTERN_INTERNAL_PLAN_FILE.sub(
                "--internal-plan-file INTERNAL_PLAN_FILE", x)
            x = PATTERN_HASH.sub("hash=HASH", x)
            assert fd is None or x == fd, "Multiple fd call candidates: %s" % "\n".join(calls)
            fd = x
            count += 1

        assert count == 1, "Cannot determine tool for call string"
    assert translate is not None or has_error, "Unable to determine translate call"
    assert fd is not None or has_error, "Unable to determine fd call"
    return translate, fd


def get_search_stats(content):
    times = PATTERN_ACTUAL_SEARCH_TIME.findall(content)
    if len(times) == 0:
        return None, [], 0, 0

    nb_solutions = len(PATTERN_SOLUTION_FOUND.findall(content))
    nb_no_solutions = len(PATTERN_NO_SOLUTION_FOUND.findall(content))
    assert nb_solutions + nb_no_solutions == len(times)

    exit_times = [t[7] for t in times if t[7] != ""]
    if len(exit_times) == 0 or times[-1][7] == "":
        exit_times = None

    return exit_times, times, nb_solutions, nb_no_solutions


def get_exit_codes(content):
    exit_codes = PATTERN_EXIT_CODE.findall(content)
    converted_exit_codes = []
    for no, exit_code in enumerate(exit_codes):
        for code in exit_code[1:]:
            if code != "":
                converted_exit_codes.append(int(code))
                break
        assert len(converted_exit_codes) == no + 1, "Invalid exit code"
    return converted_exit_codes


def has_sampling_finished(content):
    for k in ["finished2\n", "finished2", "finished\n", "finished"]:
        if content.endswith(k):
            return True
    return False


def get_output_stats(path, has_error):
    assert os.path.exists(path)
    stats = {}
    with open(path, "r") as f:
        content = f.read()

        stats["call_string"] = get_search_pattern(
            PATTERN_JOB_CALLSTRING, content, 0, allowed_none=has_error
        )
        stats["timestamp"] = get_search_pattern(
            PATTERN_START_TIMESTAMP, content, 0, allowed_none=has_error
        )



        domain, problem = get_pddl_input(content, has_error)
        stats["pddl_domain"] = domain
        stats["pddl_problem"] = problem

        translate, fd = get_call_strings(content, has_error)
        stats["call_translate"] = translate
        stats["call_fd"] = fd

        exit_times, search_times, nb_solutions, nb_no_solutions = (
            get_search_stats(content))
        stats["last_time"] = exit_times
        stats["nb_problems"] = len(search_times)
        stats["nb_solutions"] = nb_solutions
        stats["nb_no_solutions"] = nb_no_solutions

        stats["error"] = has_error
        stats["exit_code"] = get_exit_codes(content)
        stats["finished"] = has_sampling_finished(content)

        stats["final_100_lines"] = tools.last_n_lines(content, 100)

    return stats


""" Managing methods """
def summarize_errors(path, stats, errors, temporary=None):
    paths = []
    for no, (err, job_ids) in enumerate(errors.items()):
        path_error = os.path.join(path, "error_%i.txt" % no)
        path_tmp_error = (None if temporary is None else
                          os.path.join(temporary, "error_%i.txt" % no))
        outcomes = {}
        have_finished = ([0], set())
        for job_id in job_ids:
            stat = stats[job_id]
            exit_codes = stat["exit_code"]

            if stat["finished"]:
                have_finished[0][0] += 1
                have_finished[1].add(job_id)

            for code in exit_codes:
                if code not in outcomes:
                    outcomes[code] = ([0], set())
                outcomes[code][0][0] += 1
                outcomes[code][1].add(job_id)

        content = "finished: %i. Job_id: %s\n" % (
            have_finished[0][0], ", ".join(have_finished[1]))
        for code, data in sorted(outcomes.items()):
            content += "%i: %i. Job_id: %s\n" % (code, outcomes[code][0][0],
                                                 ", ".join(outcomes[code][1]))

        with open(path_error if path_tmp_error is None
                  else path_tmp_error, "w") as f:
            f.write(content)
            f.write(err)
        paths.append((path_error, path_tmp_error))
    return paths


def load_previous_stats_and_prepare(path, temporary=None):
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
    return (path_stats, path_tmp_stats, path_errors, path_tmp_errors,
            stats_mapping, error_mapping)


def move_temporary_to_final(file_tuples):
    print("Moving from temporary to final location:")
    for trg, src in file_tuples:
        if src is not None and os.path.isfile(src):
            shutil.move(src, trg)

def humanify_list(l, process_value = lambda x: x):
    if len(l) == 0:
        return l
    humanified = []
    last_merge_entry=False
    curr_value = process_value(l[0])
    start_curr = -1

    def insert_value_change(change_idx, previous_idx, previous_value, is_merge_entry):
        if change_idx - previous_idx > 1:
            if is_merge_entry:
                humanified[-1] += "]"
            humanified.append("%i * [%s]" % (change_idx - previous_idx,
                                             str(previous_value)))
            return False
        else:
            if is_merge_entry:
                humanified[-1] += ", %s" % str(previous_value)
            else:
                humanified.append("[%s" % str(previous_value))
            return True

    for idx, value in enumerate(l[1:]):
        value = process_value(value)
        if value != curr_value:
            last_merge_entry = insert_value_change(
                idx, start_curr, curr_value, last_merge_entry)
            curr_value = value
            start_curr = idx
    last_merge_entry = insert_value_change(
        len(l) - 1, start_curr, curr_value, last_merge_entry)
    if last_merge_entry:
        humanified[-1] += "]"

    return " + ".join(humanified)





def summarize(path,
              temporary=None,
              clean=False,
              skip_error_summary=False,
              cache_intervals=CACHE_INTERVALS):
    if not os.path.isdir(path):
        raise ValueError("Given path to clean is no directory")
    assert cache_intervals is None or cache_intervals == int(cache_intervals)
    assert cache_intervals is None or cache_intervals > 0

    # Setup/Reload the data storage
    (path_stats, path_tmp_stats, path_errors, path_tmp_errors,
     stats_mapping, error_mapping) = load_previous_stats_and_prepare(
        path=path, temporary=temporary)

    # Go through all files (skip if irrelevant or previously done)
    next_caching = -1 if cache_intervals is None else cache_intervals
    next_caching_errors = False
    log_files = []
    log_filter = lambda x: PATTERN_SAMPLE_LOG_FILE.match(x)
    log_generator = tools.get_log_generator(path, stats_mapping,
                                            delete=False, filter=log_filter)
    for counter, (path_out, path_err) in enumerate(log_generator):
        print("%i: %s" % (counter, path_out))
        job_id = tools.get_job_id(path_out)
        next_caching -= 1

        # Process error (nothing happens if no error)
        current_error = tools.get_error(path_err, error_unification_func)
#        tools.update_and_save_error_summary(
#            path_errors if path_tmp_errors is None else path_tmp_errors,
#            error_mapping, job_id, current_error)

        # Process stats
        current_stats = get_output_stats(path_out, current_error is not None)
        stats_mapping[job_id] = current_stats
#        tools.save_dict(
#            path_stats if path_tmp_stats is None else path_tmp_stats,
#            stats_mapping)

        if current_stats["last_time"] is None and current_error is None:
            current_error = MISSING_FD_LAST_TIME_ERROR
        if tools.update_error_summary(error_mapping, job_id, current_error):
            next_caching_errors = True

        log_files.append(path_out)
        log_files.append(path_err)

        if next_caching == 0:
            if next_caching_errors:
                tools.save_dict(
                    path_errors if path_tmp_errors is None else path_tmp_errors,
                    error_mapping)
            tools.save_dict(
                path_stats if path_tmp_stats is None else path_tmp_stats,
                stats_mapping)
            next_caching = -1 if cache_intervals is None else cache_intervals
            next_caching_errors = False

    if next_caching != cache_intervals:
        if next_caching_errors:
            tools.save_dict(
                path_errors if path_tmp_errors is None else path_tmp_errors,
                error_mapping)
        tools.save_dict(
            path_stats if path_tmp_stats is None else path_tmp_stats,
            stats_mapping)
    print("Summarize")
    if not skip_error_summary:
        paths_error_files = summarize_errors(
            path, stats_mapping, error_mapping, temporary)

    if temporary is not None:
        move_temporary_to_final(
            [(path_stats, path_tmp_stats),
             (path_errors, path_tmp_errors)] + paths_error_files
        )

    if clean:
        print("Clean processed logs")
        for log_file in log_files:
            os.remove(log_file)

    print("Done")


def merge(path, clean=False, skip_error_summary=False):
    # Setup/Reload the data storage
    (path_stats, path_tmp_stats, path_errors, path_tmp_errors,
     stats_mapping, error_mapping) = load_previous_stats_and_prepare(
        path=path, temporary=None)

    stats_filter = lambda x: PATTERN_STATS_ERROR_FILE.match(x)
    log_generator = tools.get_log_generator(
        path, stats_mapping,
        get_file_group=tools.get_associated_stat_error_files,
        delete=False, filter=stats_filter)

    processed_files = []
    for counter, associated_files in enumerate(log_generator):
        processed_files.extend(associated_files)
        path_new_job_stats, path_new_errors = associated_files[:2]

        with open(path_new_job_stats, "r") as f:
            new_stats = json.load(f)
            for value in new_stats.values():
                value["exit_code"] = humanify_list(value["exit_code"])
                value["last_time"] = None
            stats_mapping.update(new_stats)
        if path_new_errors is not None:
            with open(path_new_errors, "r") as f:
                new_errors = json.load(f)
            for new_error_messages, new_error_jobs in new_errors.items():
                if new_error_messages not in error_mapping:
                    error_mapping[new_error_messages] = new_error_jobs
                else:
                    error_mapping[new_error_messages].extend(new_error_jobs)

    tools.save_dict(
        path_errors if path_tmp_errors is None else path_tmp_errors,
        error_mapping)
    tools.save_dict(
        path_stats if path_tmp_stats is None else path_tmp_stats,
        stats_mapping)

    if clean:
        print("Clean processed logs")
        for pfile in processed_files:
            if pfile is not None:
                os.remove(pfile)

    if not skip_error_summary:
        paths_error_files = summarize_errors(
            path, stats_mapping, error_mapping)


def parse_args(argv):
    options = parser.parse_args(argv)
    for path in options.directory:
        assert os.path.isdir(path), \
            "A directory to summarize does not exit: %s" % path
    assert (options.temporary_directory is None
            or os.path.isdir(options.temporary_directory)), \
        ("The given directory for temporary data does not exist: %s"
         % options.temporary_directory)
    return options


def run(argv):
    options = parse_args(argv)
    for path in options.directory:
        if options.merge:
            merge(path=path,
                  clean=options.clean,
                  skip_error_summary=options.skip_creation_error_files)
        else:
            summarize(path=path,
                      temporary=options.temporary_directory,
                      clean=options.clean,
                      skip_error_summary=options.skip_creation_error_files,
                      cache_intervals=None)


if __name__ == "__main__":
    run(sys.argv[1:])
