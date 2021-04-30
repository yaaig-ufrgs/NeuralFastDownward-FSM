#!/usr/bin/env python
"""
This script replaces sbatch to submit sampling jobs. The script receives the
same arguments as sbatch would, but it reads the submitted job ids and adds an
additional job which summarizes the logs of the sampling runs AFTER all runs
have finished.
"""
from __future__ import print_function

import sbatch_tools as stools

import os
import sys


PATH_SUMMARIZE_SAMPLING = os.path.join(
    stools.PATH_MISC_SLURM, "summarize_sampling_output.sh")

try:
    # Submit sampling job
    job_id = stools.submit_job(["sbatch"] + sys.argv[1:])
    job_id = stools.submit_job(["sbatch", PATH_SUMMARIZE_SAMPLING, "."],
                               dependencies="afterany:%i" % job_id)
except:
    print("Failed to submit jobs.")

