#!/bin/sh
#SBATCH -p standard
#SBATCH -o out_%j.txt
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 0-07:59:59
#SBATCH -n 1
#SBATCH -c 24
#SBATCH -a 0-3
#SBATCH --mail-type=all
python launcher_cma.py clustercoma.yml $SLURM_ARRAY_TASK_ID
