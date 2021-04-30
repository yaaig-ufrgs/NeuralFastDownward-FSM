#!/bin/bash
## Runs the training output summarization for all directories provided
## as parameter. Do not provide the parameter
## --clean, this scripts sets it automatically


#SBATCH --job-name=summarize_training_output
#SBATCH --time=48:00:00
#SBATCH --mem=3G
#SBATCH -o summarize_training_output_%j.out # Standard output
#SBATCH -e summarize_training_output_%j.err # Standard error
#SBATCH --partition infai_1

ulimit -Sv 3072000

##LOAD MODULES
module load Python/2.7.11-goolf-1.7.20

$DEEPDOWN/misc/slurm/summarize_training_output.py --clean $@
