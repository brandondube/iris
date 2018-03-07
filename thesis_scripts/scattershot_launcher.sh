#!/bin/bash
module load anaconda/5.0.1c
source deactivate
source activate thesis
python make_costfunction_comparison_queues.py
sbatch launcher.sh
