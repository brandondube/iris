#!/bin/sh
#SBATCH -p standard
#SBATCH -o out_$j.txt
#SBATCH --mem-per-cpu=1GB
#SBATCH -t 0-07:59:59
#SBATCH -n 1
#SBATCH -c 24
module load anaconda/5.0.1c
source deactivate
source activate thesis
python launcher.py clustergrid.yml $idx
