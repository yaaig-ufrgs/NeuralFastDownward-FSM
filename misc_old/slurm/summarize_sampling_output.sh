#!/bin/bash
## Runs the sampling output summarization for all directories provided
## as parameter. Do not provide the parameter 
## --temporary-directory nor --clean, this scripts sets it automatically


#SBATCH --job-name=summarize_sampling_output
#SBATCH --time=48:00:00
#SBATCH --mem=3G
#SBATCH -o summarize_sampling_output_%j.out # Standard output
#SBATCH -e summarize_sampling_output_%j.err # Standard error
#SBATCH --partition infai_2

ulimit -Sv 3072000

##LOAD MODULES
module load Python/2.7.11-goolf-1.7.20

$DEEPDOWN/misc/slurm/summarize_sampling_output.py --clean --merge $@
