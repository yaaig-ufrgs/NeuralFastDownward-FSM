#!/bin/bash

#SBATCH --job-name=fast_convert
#SBATCH --time=06:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH -o slurm_convert_%j.out # Standard output
#SBATCH -e slurm_convert_%j.err # Standard error
#SBATCH --partition infai_1

## Do not use --tmp! It will be set automatically

ulimit -Sv 3072000
STDOUT_SLURM_LOG=`pwd`"/slurm_convert_${SLURM_JOB_ID}.out"
STDOUT_LOCAL="$TMPDIR/stdout.log"

##LOAD MODULES
module load Python/2.7.11-goolf-1.7.20
source ~/bin/kerascpu/bin/activate


##SETUP PROBLEM TO RUN IF ARRAY JOB
if [ ! -z ${SLURM_ARRAY_TASK_ID+x} ]; then
	if [ -z ${SLURM_ARRAY_TASK_COUNT+x} ]; then
	    (>&2 echo "error: Error with slurm array variables");
	    exit 4;
	fi
    echo "Slurm array id: $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"

    FILE_TASK=$(($SLURM_ARRAY_TASK_ID + 1))
    FILE_TASK=${@:FILE_TASK:1}

    shift $SLURM_ARRAY_TASK_COUNT
    set -- --task-files "$FILE_TASK" -tmp "$TMPDIR" "$@"
fi

cd $TMPDIR

##RUN
$DEEPDOWN/fast-convert.py "$@" > $STDOUT_LOCAL
EXIT_CODE=$?
cat $STDOUT_LOCAL >> $STDOUT_SLURM_LOG

if [ "$EXIT_CODE" -eq "0" ]; then
	echo "finished"
else
	(>&2 echo "error: Unable to convert data")
fi
deactivate
