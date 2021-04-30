#!/bin/bash

#SBATCH --job-name=analyse_dataset_size
#SBATCH --time=48:00:00
#SBATCH --mem=7G
#SBATCH --cpus-per-task=1
#SBATCH -o slurm_data_set_size_%j.out # Standard output
#SBATCH -e slurm_data_set_size_%j.err # Standard error
#SBATCH --partition infai_2

ulimit -Sv 7168000

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
DIR_DOMAIN=$1
shift $(($SLURM_ARRAY_TASK_COUNT-$SLURM_ARRAY_TASK_ID))

echo $DIR_DOMAIN
echo $@

$DEEPDOWN/misc/training/analyse_dataset_size.py --directory $DIR_DOMAIN $@
deactivate
