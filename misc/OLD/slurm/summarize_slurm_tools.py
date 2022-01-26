from __future__ import print_function

import json
import os
import re
import shutil

REGEX_FLOAT = r"\d+\.?\d*(e-?\d+)?"
REGEX_HEXADECIMAL = "0[xX][0-9a-fA-F]+"
SLURM_FILE_SUFFICES = [".out", ".err"]

PATTERN_JOB_ID = re.compile(r"(.*/)?(slurm_([^_]+)|job_stats|errors)_"
                            r"(\d+(_\d+)?)\.(err|out|json)")
PATTERN_SLURM_CANCELLED = re.compile(r"slurmstepd: error: \*\*\* JOB (\d+) "
                                     r"ON ase(\d+) "
                                     r"CANCELLED AT "
                                     r"(\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d) "
                                     r"\*\*\*")
PATTERN_SLURM_TIMEOUT = re.compile(
    r"slurmstepd: error: \*\*\* JOB (\d+) ON ase(\d+) CANCELLED AT "
    r"(\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d) DUE TO TIME LIMIT \*\*\*")
PATTERN_SLURM_KILLED = re.compile(r"/var/lib/slurm/slurmd/job(\d+)/slurm_"
                                  r"script: line (\d+): (\d+) Killed")


def remove(path, ok_missing=False):
    if os.path.exists(path):
        os.remove(path)
    elif not ok_missing:
        raise ValueError("Item to delete does not exist: %s" % path)


def get_job_id(path, throw_error=False):
    match = PATTERN_JOB_ID.match(path)
    if match is None and throw_error:
        raise ValueError("Given path does not contain a slurm job id: %s" %
                         path)
    return None if match is None else match.groups()[3]


def is_slurm_output(path):
    return get_job_id(path) is not None


def get_associated_stat_error_files(path):
    if not is_slurm_output(path):
        raise ValueError("Given file is not a slurm output file")

    if os.path.basename(path).find("job_stats") > -1:
        path_job_stats = path
        path_errors = os.path.join(
            os.path.dirname(path),
            os.path.basename(path).replace("job_stats", "errors"))
    elif os.path.basename(path).find("errors") > -1:
        path_errors = path
        path_job_stats = os.path.join(
            os.path.dirname(path),
            os.path.basename(path).replace("errors", "job_stats"))
    else:
        assert False

    job_id = get_job_id(path)
    log_stdout = os.path.join(os.path.dirname(path),
                              "slurm_sample_%s.out" % job_id)
    log_stderr = os.path.join(os.path.dirname(path),
                              "slurm_sample_%s.err" % job_id)
    return [f if os.path.exists(f) else None for f in
            [path_job_stats, path_errors, log_stdout, log_stderr]]


def get_associated_log_files(path):
    if not is_slurm_output(path):
        raise ValueError("Given file is not a slurm output file")
    base = path[:-4]
    return base + ".out", base + ".err"

def last_n_lines(content, lines):
    assert lines >= 0
    if lines == 0 or content == "":
        return ""

    if content[-1] == "\n":
        lines += 1

    counter = 0
    i = len(content) - 1
    while i >= 0:
        if content[i] == "\n":
            counter += 1
            if counter == lines:
                return content[i + 1:]
        i -= 1
    return content


def get_search_pattern(pattern, content, group_index=None, group_indices=None,
                       allowed_none=True):
    assert (group_index is None) ^ (group_indices is None)
    match = pattern.search(content)
    if match is None:
        if allowed_none:
            return None
        else:
            raise ValueError("Unable to match: %s" % pattern.pattern)
    if group_index is not None:
        return match.groups()[group_index]
    else:
        return [match.groups()[idx] for idx in group_indices]


""" Error related functions """


def unify_slurm_cancel_error(content):
    return PATTERN_SLURM_CANCELLED.sub(
        "slurmstepd: error: *** JOB ID ON aseID CANCELLED AT "
        "YYYY-MM-DDThh:mm:ss ***", content
    )


def unify_slurm_timeout_error(content):
    return PATTERN_SLURM_TIMEOUT.sub(
        "slurmstepd: error: *** JOB ID ON aseID CANCELLED AT "
        "YYYY-MM-DDThh:mm:ss DUE TO TIME LIMIT ***", content
    )


def unify_slurm_oom_killed_error(content):
    return PATTERN_SLURM_KILLED.sub("/var/lib/slurm/slurmd/jobID/slurm_script: "
                                    "line NB: SOMETHING Killed", content)

#
def get_error(path_err, error_unification_func=None):
    error = None
    if os.path.exists(path_err):
        with open(path_err, "r") as f:
            content = f.read()
            if (content.find("Traceback") > -1 or
                    content.find("traceback") > -1 or
                    content.find("error") > -1 or content.find("Error") > -1):
                error = (content if error_unification_func is None
                         else error_unification_func(content))
    return error

def update_error_summary(error_mapping, job_id, new_error):
    """

    :param error_mapping:
    :param job_id:
    :param new_error:
    :return: True is error_mapping was changed
    """
    if new_error is not None:
        if new_error not in error_mapping:
            error_mapping[new_error] = []
        error_mapping[new_error].append(job_id)
        return True
    return False


def update_and_save_error_summary(path_error_summary, error_mapping,
                                  job_id, new_error):
    if update_error_summary(error_mapping, job_id, new_error):
        save_dict(path_error_summary, error_mapping)


""" Caching/Output functions"""


def get_path_error_summary(path_dir):
    return os.path.join(path_dir, "errors.json")


def get_path_stats_summary(path_dir):
    return os.path.join(path_dir, "job_stats.json")


def load_summary(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        return {}


def get_summaries_data(path_dir, tmp_dir=None):
    path_err = get_path_error_summary(path_dir)
    errors = load_summary(path_err)
    path_stats = get_path_stats_summary(path_dir)
    stats = load_summary(path_stats)
    if tmp_dir is not None:
        path_tmp_err = get_path_error_summary(tmp_dir)
        path_tmp_stats = get_path_stats_summary(tmp_dir)
        return path_stats, path_err, stats, errors, path_tmp_stats, path_tmp_err
    else:
        return path_stats, path_err, stats, errors



def save_dict(path, d, path_tmp=None):
    path_tmp = (path + ".tmp") if path_tmp is None else path_tmp
    with open(path_tmp, "w") as f:
        json.dump(d, f, indent=4, sort_keys=True)
    shutil.move(path_tmp, path)

def get_log_generator(path_dir, previous_job_ids=None,
                      delete=False, filter=None,
                      get_file_group=get_associated_log_files):
    for item in sorted(os.listdir(path_dir)):
        path_item = os.path.join(path_dir, item)
        if not is_slurm_output(path_item):
            continue
        if filter is not None and not filter(path_item):
            continue

        file_group = get_file_group(path_item)
        job_id = get_job_id(path_item)

        # Previously processed?
        if previous_job_ids is None or job_id not in previous_job_ids:
            yield file_group

        if delete:
            for associated_file in file_group:
                remove(associated_file, ok_missing=True)


""" Miscellaneous functions """


def sanitize_string(s, strip=True, quotations=True):
    if strip:
        s = s.strip()
    if quotations:
        for c in ["'", '"']:
            if s[0] == c and s[-1] == c:
                s = s[1:-1]
                break
    return s
