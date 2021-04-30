#!/bin/bash

#SBATCH --job-name=convert_sample_format
#SBATCH --time=48:00:00
#SBATCH --mem=3G
#SBATCH --cpus-per-task=1
#SBATCH -o convert_sample_format_%j.out # Standard output
#SBATCH -e convert_sample_format_%j.err # Standard error
#SBATCH --partition infai_2

ulimit -Sv 3072000

##LOAD MODULES
module load Python/2.7.11-goolf-1.7.20
source ~/bin/kerascpu/bin/activate

##SETUP PROBLEM TO RUN IF ARRAY JOB
if [ -z ${SLURM_ARRAY_TASK_ID+x} ]; then
    (>&2 echo "error: Error with slurm array id variables");
    exit 4;
fi
if [ -z ${SLURM_ARRAY_TASK_COUNT+x} ]; then
    (>&2 echo "error: Error with slurm array count variables");
    exit 4;
fi
echo "Slurm array id: $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT"
    
shift $SLURM_ARRAY_TASK_ID
shift $SLURM_ARRAY_TASK_ID
DIRECTORY=$1
REGEX=$2
shift $(($SLURM_ARRAY_TASK_COUNT-$SLURM_ARRAY_TASK_ID))
shift $(($SLURM_ARRAY_TASK_COUNT-$SLURM_ARRAY_TASK_ID))

$DEEPDOWN/misc/training/convert_sample_format.py $@ --directory $DIRECTORY --regex $REGEX
deactivate
