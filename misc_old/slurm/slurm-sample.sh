#!/bin/bash
## Usage:
## This scripts can ONLY call the sampling not the traversing block of
## ./fast-sample.py
## The environment variable TARGET_DIR is the directory to which to
## sample the data (replaces --target-folder). If no given, no target
## folder is specified. --target-file and --temporary-folder is NOT
## available.
## Provide arguments in the same way as for fast-sample.py EXCEPT:
## Do NOT provide
##  --temporary-folder
##  --target-file
##  --target-folder
##    The script sets them itself and cannot use the option to store them in a
##    target file (as multiple runs can be done in parallel on different
##    compute nodes.
## --fast-downward (this will be set to be in $DEEPDOWN)

#SBATCH --job-name=fast_sample
#SBATCH --time=06:00:00
#SBATCH --mem=3G
#SBATCH -o slurm_sample_%j.out # Standard output
#SBATCH -e slurm_sample_%j.err # Standard error
#SBATCH --partition infai_1

ulimit -Sf 52428800
ulimit -Sv 3072000
INITIAL_DIR=`pwd`
STDOUT_LOCAL="$TMPDIR/slurm_sample_${SLURM_JOB_ID}.out"
STDERR_LOCAL="$TMPDIR/slurm_sample_${SLURM_JOB_ID}.err"

##LOAD MODULES
module load Python/2.7.11-goolf-1.7.20
source ~/bin/kerascpu/bin/activate

##SETUP VARIABLE
if [ ! -z ${TARGET_DIR+x} ]; then
    echo "TARGET_DIR" $TARGET_DIR
	if [ ! -d "$TARGET_DIR" ]; then
		mkdir $TARGET_DIR
	fi
	if [ ! -d "$TARGET_DIR" ]; then
		(>&2 echo "error: Unable to make TARGET_DIR")
		exit 1
	fi

	PATH_DATA=$TMPDIR/data
	echo "PATH_DATA" $PATH_DATA
	if [ ! -d "$PATH_DATA" ]; then
		mkdir $PATH_DATA
	fi
	if [ ! -d "$PATH_DATA" ]; then
		(>&2 echo "error: Unable to make DATA DIR")
		exit 3
	fi
fi


PATH_TMP=$TMPDIR/tmp
echo "PATH_TMP" $PATH_TMP
if [ ! -d "$PATH_TMP" ]; then
    mkdir $PATH_TMP
fi
if [ ! -d "$PATH_TMP" ]; then
	(>&2 echo "error: Unable to make TMP DIR")
    exit 2
fi


##SETUP PROBLEM TO RUN IF ARRAY JOB
if [ ! -z ${SLURM_ARRAY_TASK_ID+x} ]; then
	if [ -z ${SLURM_ARRAY_TASK_COUNT+x} ]; then
	    (>&2 echo "error: Error with slurm array variables");
	    exit 4;
	fi
        echo "Slurm array id: $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"

	shift $SLURM_ARRAY_TASK_ID
	PROBLEM=$1
	shift $(($SLURM_ARRAY_TASK_COUNT-$SLURM_ARRAY_TASK_ID))
	set -- $PROBLEM "${@:1}"

	WD=`pwd`
	case $1 in
		/*) PROBLEM=$1 ;;
		*) PROBLEM="${WD}/${1}" ;;
	esac

	set -- $PROBLEM "${@:2}"
	cd $TMPDIR

fi
echo "SamplingCall: " $@ >> $STDOUT_LOCAL

##RUN
if [ ! -z ${TARGET_DIR+x} ]; then
        $DEEPDOWN/fast-sample.py -fd $DEEPDOWN/fast-downward.py -tmp $PATH_TMP -t $PATH_DATA "$@" 2>>$STDERR_LOCAL | tail -n 5000 >> $STDOUT_LOCAL
        EXIT_CODE=$?
	if [ "$EXIT_CODE" -eq "0" ]; then
		mv $PATH_DATA/* $TARGET_DIR
		echo "finished1" >> $STDOUT_LOCAL
	else
		echo "error: Unable to sample data1" >> $STDERR_LOCAL
	fi
else
        $DEEPDOWN/fast-sample.py -fd $DEEPDOWN/fast-downward.py -tmp $PATH_TMP "$@" 2>>$STDERR_LOCAL | tail -n 5000 >> $STDOUT_LOCAL
        EXIT_CODE=$?
	if [ "$EXIT_CODE" -eq "0" ]; then
		echo "finished2" >> $STDOUT_LOCAL
	else
		echo "error: Unable to sample data2"  >> $STDERR_LOCAL
	fi
fi


$DEEPDOWN/misc/slurm/summarize_sampling_output.py $TMPDIR --clean --skip-creation-error-files
ls

JSON_STATS="job_stats.json"
JSON_ERRORS="errors.json"
if [[ -e $JSON_ERRORS ]]; then
    cp $JSON_ERRORS "${INITIAL_DIR}/errors_${SLURM_JOB_ID}.json"
fi
if [[ -e $JSON_STATS ]]; then
    cp $JSON_STATS "${INITIAL_DIR}/job_stats_${SLURM_JOB_ID}.json"
fi
