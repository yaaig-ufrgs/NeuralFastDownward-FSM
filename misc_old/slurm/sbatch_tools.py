#!/usr/bin/env python
"""
This script contains the tools to submit jobs to slurm via sbatch, receive
their job ids and start them in a dependency order.
Additional other features for our use case are present.

"""
from __future__ import print_function

import os
import re
import sys

if sys.version_info < (3,):
    import subprocess32 as subprocess

    def decoder(x):
        return x.decode()
else:
    import subprocess

    def decoder(x):
        return x


PATH_NDOWNWARD = os.environ.get("DEEPDOWN")
assert PATH_NDOWNWARD is not None, \
    ("DEEPDOWN variable not set! Please point DEEPDOWN to your "
     "Neural Fast Downward repository")

PATH_MISC_SLURM = os.path.join(PATH_NDOWNWARD, "misc", "slurm")

PATTERN_SUBMITTED_BATCH_JOB = re.compile(r"Submitted batch job (\d*)")


def call_subprocess(command, multijobs=False, verbose=2):
    """
    Calls a command via subprocess and reads the slurm job_ids from stdout.
    :param command: command to submit
    :param multijobs: do we expect multiple jobs to be submitted by the command
    :param verbose: 0 nothing, 1 + jobids, 2 + command
    :return: job_ids (if multijobs as list otherwise a single job_id)
    """
    if verbose >= 2:
        print("Submit job:", command)

    out = subprocess.check_output(command)
    out = decoder(out)
    job_ids = [int(x) for x in PATTERN_SUBMITTED_BATCH_JOB.findall(out)]
    assert len(job_ids) > 0, \
        "Submitting job with sbatch failed: '{out}'".format(**locals())
    assert len(job_ids) == 1 or multijobs, \
        "Submitted multiple jobs, although that was not specified"

    if verbose >= 1:
        print("Submitted job:", ", ".join(str(x) for x in job_ids))
    return job_ids if multijobs else job_ids[0]


def submit_job(command, dependencies=None, multijobs=False, verbose=2):
    """
    Submits a job via slurm
    :param command: command to submit (requires sbatch as first argument)
    :param dependencies: slurm dependency to add (adds automatically
    --kill-on-invalid-dep=yes
    :param multijobs: do we expect multiple jobs to be submitted by the command
    :param verbose: 0 nothing, 1 + jobids, 2 + command
    :return: job_ids (if multijobs as list otherwise a single job_id)
    """
    assert command[0] == "sbatch"
    if dependencies is not None:
        command[1:1] = ["--dependency", dependencies,
                        "--kill-on-invalid-dep=yes"]

    return call_subprocess(command, multijobs=multijobs, verbose=verbose)



