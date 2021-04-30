#!/bin/bash

#SBATCH --job-name=summarize_sampling
#SBATCH --time=48:00:00
#SBATCH --mem=7G
#SBATCH -o slurm_summarize_sampling_%j.out # Standard output
#SBATCH -e slurm_summarize_sampling_%j.err # Standard error
#SBATCH --partition infai_2

##LOAD MODULES
module load Python/2.7.11-goolf-1.7.20


##RUN
echo "START"
python summarize_sampling_output.py $1
echo "DONE"
