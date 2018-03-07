#!/bin/sh
#SBATCH -p standard
#SBATCH -o out_%j.txt
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 0-07:59:59
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -a 1-20
#SBATCH --mail-type=all
module load anaconda/5.0.1c
source deactivate
source activate thesis
python -OO launcher.py clustergrid.yml $SLURM_ARRAY_TASK_ID
