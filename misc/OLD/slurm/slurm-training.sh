#!/bin/bash

#SBATCH --job-name=fast_train
#SBATCH --time=48:00:00
#SBATCH --mem=7G
#SBATCH --cpus-per-task=2
#SBATCH -o slurm_training_%j.out # Standard output
#SBATCH -e slurm_training_%j.err # Standard error
#SBATCH --partition infai_1

## If an array job is given, then it is expected that N regular 
## expressions are the first N parameters and the SLURM_ARRAY_TASK_ID
## determines which of those expression describes the test and which the
## validation problems.

ulimit -Sv 7168000
STDOUT_SLURM_LOG=`pwd`"/slurm_training_${SLURM_JOB_ID}.out"
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
    
    # Set test & validation problem regex filter
    fold_test=$(($SLURM_ARRAY_TASK_ID + 1))
    fold_valid=$(($SLURM_ARRAY_TASK_ID + 2))
    if (( $fold_valid >  $SLURM_ARRAY_TASK_COUNT )); then
		fold_valid=1
    fi
    fold_test=${@:$fold_test:1}
    fold_valid=${@:$fold_valid:1}
    shift $SLURM_ARRAY_TASK_COUNT
    set -- "$@" "--test" "$fold_test" "--validation" "$fold_valid"
    
    # Adapt prefix to include fold number
    FOUND_PREFIX=0
	for (( idx=1; idx<=$#; idx++ )); do
		if [[ ${@:$idx:1} == "-p" ]] || [[ ${@:$idx:1} == "--prefix" ]]; then
			idx_pref=$(( $idx + 1 ))
			set -- ${@:1:$idx} "${@:$idx_pref:1}${SLURM_ARRAY_TASK_ID}_fold_" ${@:(( $idx_pref + 1 ))}
			FOUND_PREFIX=1
			break
		fi
	done
	if [[ $FOUND_PREFIX == 0 ]]; then
		set -- ${@} "--prefix" "${SLURM_ARRAY_TASK_ID}_fold_"
	fi
fi

cd $TMPDIR

##RUN
$DEEPDOWN/fast-training.py "$@" > $STDOUT_LOCAL
EXIT_CODE=$?
cat $STDOUT_LOCAL >> $STDOUT_SLURM_LOG

if [ "$EXIT_CODE" -eq "0" ]; then
	echo "finished"
else
	(>&2 echo "error: Unable to train network(s)")
fi
deactivate
